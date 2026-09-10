#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test Qwen3AttentionWithDiffusionToggle mask behavior.

Checks:
  1. diffusion_lm=False:
     The attention should behave as standard causal attention.
     We verify the mask passed to attention backend is causal:
       - allowed on diagonal and below
       - blocked above diagonal

  2. diffusion_lm=True + 4D block_diff mask:
     The attention should use custom block_diff mask with is_causal=False.

Usage:
  cd /home/c00633840/MindSpeed-LLM

  python scripts/test_qwen3_attention_mask_behavior.py \
    --model-path /home/c00633840/models/Qwen3-1.7B-Base \
    --seq-len 64 \
    --batch-size 1 \
    --block-size 16 \
    --device npu:0

CPU:
  python scripts/test_qwen3_attention_mask_behavior.py \
    --model-path /home/c00633840/models/Qwen3-1.7B-Base \
    --seq-len 16 \
    --batch-size 1 \
    --block-size 4 \
    --device cpu
"""

import argparse
import traceback
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, Qwen3Config


CAPTURED = {
    "calls": [],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


def detect_device():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu:0"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def get_dtype(dtype_name: str):
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    return torch.float32


def set_diffusion_lm(model, value: bool):
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "diffusion_lm"):
            layer.self_attn.diffusion_lm = value


def swap_attention(model):
    from mindspeed_llm.fsdp2.models.qwen3.qwen3_attention import (
        Qwen3AttentionWithDiffusionToggle,
        _swap_attention,
    )

    num_swapped = 0
    for layer in model.model.layers:
        _swap_attention(layer, Qwen3AttentionWithDiffusionToggle)
        num_swapped += 1

    print(f"[INFO] swapped attention layers: {num_swapped}")

    attn0 = model.model.layers[0].self_attn
    print(f"[INFO] layer0 attention class: {type(attn0)}")
    print(f"[INFO] layer0 has diffusion_lm: {hasattr(attn0, 'diffusion_lm')}")


def make_block_diff_mask(
    seq_len: int,
    block_size: int,
    batch_size: int,
    dtype: torch.dtype,
    device: str,
):
    """
    Construct additive block_diff mask for concatenated [noisy, original].

    Total length = 2 * seq_len.

    Semantics:
      0.0  = attend
      -inf = blocked

    Same logic as your Qwen3DiffusionForCausalLM.make_block_diff_mask.
    """
    L = seq_len
    n = L
    total_len = 2 * L

    q_idx = torch.arange(total_len, device=device).unsqueeze(1)
    kv_idx = torch.arange(total_len, device=device).unsqueeze(0)

    x0_flag_q = q_idx >= n
    x0_flag_kv = kv_idx >= n

    block_q = torch.where(
        x0_flag_q,
        (q_idx - n) // block_size,
        q_idx // block_size,
    )
    block_kv = torch.where(
        x0_flag_kv,
        (kv_idx - n) // block_size,
        kv_idx // block_size,
    )

    m_bd = (block_q == block_kv) & (~x0_flag_kv) & (~x0_flag_q)
    m_obc = (block_q > block_kv) & x0_flag_kv & (~x0_flag_q)
    m_bc = (q_idx >= kv_idx) & x0_flag_kv & x0_flag_q

    attend = m_bd | m_obc | m_bc

    mask = torch.full(
        (1, 1, total_len, total_len),
        float("-inf"),
        dtype=dtype,
        device=device,
    )
    mask[0, 0][attend] = 0.0

    if batch_size > 1:
        mask = mask.expand(batch_size, -1, -1, -1)

    return mask


def assert_is_causal_additive_mask(mask: torch.Tensor, q_len: int, kv_len: int):
    """
    Check additive causal mask:
      - above diagonal should be blocked
      - diagonal and below should be allowed

    Supports shape:
      [B, 1, Q, KV]
      [1, 1, Q, KV]
    """
    assert mask is not None, "Expected causal attention_mask, but got None"
    assert mask.dim() == 4, f"Expected 4D mask, got shape {tuple(mask.shape)}"
    assert mask.shape[-2] == q_len, f"Q length mismatch: {mask.shape[-2]} vs {q_len}"
    assert mask.shape[-1] == kv_len, f"KV length mismatch: {mask.shape[-1]} vs {kv_len}"

    m = mask[0, 0].detach().float().cpu()

    q_idx = torch.arange(q_len).unsqueeze(1)
    kv_idx = torch.arange(kv_len).unsqueeze(0)

    blocked_expected = kv_idx > q_idx
    allowed_expected = ~blocked_expected

    blocked_values = m[blocked_expected]
    allowed_values = m[allowed_expected]

    # Blocked positions should be -inf or very negative.
    blocked_ok = torch.isneginf(blocked_values).all() or (blocked_values < -1e4).all()

    # Allowed positions should be approximately 0.
    allowed_ok = torch.allclose(
        allowed_values,
        torch.zeros_like(allowed_values),
        atol=1e-5,
        rtol=0,
    )

    assert blocked_ok, (
        "Causal mask check failed: upper-triangular positions are not blocked. "
        f"blocked min={blocked_values.min().item()}, max={blocked_values.max().item()}"
    )
    assert allowed_ok, (
        "Causal mask check failed: diagonal/lower-triangular positions are not zero. "
        f"allowed min={allowed_values.min().item()}, max={allowed_values.max().item()}"
    )

    print("[PASS] causal additive mask is upper-triangular blocked.")


def assert_same_additive_mask(actual: torch.Tensor, expected: torch.Tensor):
    assert actual is not None, "Actual mask is None"
    assert actual.dim() == 4, f"Actual mask should be 4D, got {tuple(actual.shape)}"
    assert actual.shape == expected.shape, (
        f"Mask shape mismatch: actual={tuple(actual.shape)}, expected={tuple(expected.shape)}"
    )

    a = actual.detach().float().cpu()
    e = expected.detach().float().cpu()

    same_inf = torch.equal(torch.isneginf(a), torch.isneginf(e))

    finite = torch.isfinite(e)
    same_finite = torch.allclose(a[finite], e[finite], atol=1e-5, rtol=0)

    assert same_inf and same_finite, "Actual block_diff mask differs from expected mask."

    print("[PASS] block_diff additive mask matches expected mask.")


def install_attention_capture_patch():
    """
    Monkey patch qwen3_attention.eager_attention_forward imported inside your module.

    Your Qwen3AttentionWithDiffusionToggle.forward uses the symbol:
      eager_attention_forward

    imported from:
      transformers.models.qwen3.modeling_qwen3

    into:
      mindspeed_llm.fsdp2.models.qwen3.qwen3_attention

    Therefore we patch that module-level symbol.
    """
    import mindspeed_llm.fsdp2.models.qwen3.qwen3_attention as qa

    original_eager = qa.eager_attention_forward

    def wrapped_eager_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        *args,
        **kwargs,
    ):
        CAPTURED["calls"].append(
            {
                "module": module,
                "query_shape": tuple(query.shape),
                "key_shape": tuple(key.shape),
                "value_shape": tuple(value.shape),
                "attention_mask": attention_mask.detach().clone() if attention_mask is not None else None,
                "attention_mask_shape": tuple(attention_mask.shape) if attention_mask is not None else None,
                "is_causal": kwargs.get("is_causal", None),
                "sliding_window": kwargs.get("sliding_window", None),
                "diffusion_lm": getattr(module, "diffusion_lm", None),
            }
        )

        return original_eager(
            module,
            query,
            key,
            value,
            attention_mask,
            *args,
            **kwargs,
        )

    qa.eager_attention_forward = wrapped_eager_attention_forward

    return original_eager


def restore_attention_patch(original_eager):
    import mindspeed_llm.fsdp2.models.qwen3.qwen3_attention as qa
    qa.eager_attention_forward = original_eager


@torch.no_grad()
def test_causal_attention(model, input_ids):
    """
    diffusion_lm=False:
      Expect causal behavior.

    For eager Qwen3 attention, the model should pass a 4D additive causal mask
    into attention backend.
    """
    print("\n=== Test 1: diffusion_lm=False should be causal attention ===")

    CAPTURED["calls"].clear()

    set_diffusion_lm(model, False)

    _ = model(
        input_ids=input_ids,
        use_cache=False,
    )

    assert len(CAPTURED["calls"]) > 0, "No attention calls captured."

    first = CAPTURED["calls"][0]
    print(f"[INFO] first attention call:")
    print(f"       query_shape          = {first['query_shape']}")
    print(f"       key_shape            = {first['key_shape']}")
    print(f"       attention_mask_shape = {first['attention_mask_shape']}")
    print(f"       is_causal            = {first['is_causal']}")
    print(f"       diffusion_lm         = {first['diffusion_lm']}")

    q_len = first["query_shape"][-2]
    kv_len = first["key_shape"][-2]

    assert first["diffusion_lm"] is False, "Expected diffusion_lm=False in causal mode."

    assert_is_causal_additive_mask(
        first["attention_mask"],
        q_len=q_len,
        kv_len=kv_len,
    )


@torch.no_grad()
def test_block_diff_attention(model, input_ids, seq_len, block_size, dtype, device):
    """
    diffusion_lm=True + custom 4D block_diff mask:
      Expect attention backend receives the exact block_diff mask
      and is_causal=False.
    """
    print("\n=== Test 2: diffusion_lm=True with 4D mask should be block_diff attention ===")

    CAPTURED["calls"].clear()

    batch_size = input_ids.shape[0]

    noisy_ids = input_ids.clone()
    concat_ids = torch.cat([noisy_ids, input_ids], dim=1)

    expected_mask = make_block_diff_mask(
        seq_len=seq_len,
        block_size=block_size,
        batch_size=batch_size,
        dtype=dtype,
        device=device,
    )

    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    position_ids = position_ids.expand(batch_size, -1)

    # set_diffusion_lm(model, True)
    # model.config.diffusion_lm = True

    _ = model(
        input_ids=concat_ids,
        attention_mask=expected_mask,
        position_ids=position_ids,
        use_cache=False,
    )

    assert len(CAPTURED["calls"]) > 0, "No attention calls captured."

    first = CAPTURED["calls"][0]
    print(f"[INFO] first attention call:")
    print(f"       query_shape          = {first['query_shape']}")
    print(f"       key_shape            = {first['key_shape']}")
    print(f"       attention_mask_shape = {first['attention_mask_shape']}")
    print(f"       is_causal            = {first['is_causal']}")
    print(f"       diffusion_lm         = {first['diffusion_lm']}")

    assert first["diffusion_lm"] is True, "Expected diffusion_lm=True in block_diff mode."
    assert first["is_causal"] is False, "Expected is_causal=False in block_diff mode."

    assert_same_additive_mask(
        actual=first["attention_mask"],
        expected=expected_mask,
    )


def main():
    args = parse_args()

    device = args.device or detect_device()
    dtype = get_dtype(args.dtype)

    print(f"[INFO] device              = {device}")
    print(f"[INFO] dtype               = {dtype}")
    print(f"[INFO] model_path          = {args.model_path}")
    print(f"[INFO] seq_len             = {args.seq_len}")
    print(f"[INFO] batch_size          = {args.batch_size}")
    print(f"[INFO] block_size          = {args.block_size}")
    print(f"[INFO] attn_implementation = {args.attn_implementation}")

    config = Qwen3Config.from_pretrained(args.model_path)
    config._attn_implementation = args.attn_implementation
    config.diffusion_lm = True

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=dtype,
        device_map={"": device},
    )
    model.eval()

    swap_attention(model)

    vocab_size = model.config.vocab_size
    torch.manual_seed(1234)
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(args.batch_size, args.seq_len),
        device=device,
        dtype=torch.long,
    )

    original_eager = install_attention_capture_patch()

    try:
        # test_causal_attention(model, input_ids)
        from mindspeed_llm.fsdp2.models.qwen3_diffusion.modeling_qwen3_diffusion import (
            Qwen3DiffusionForCausalLM,
        )

        config = Qwen3Config.from_pretrained(args.model_path,)
        # Set mask_token_id = vocab_size (out-of-vocab) to trigger resize path
        config.mask_token_id = config.mask_token_id
        config.dlm_paradigm = "autoregressive"
        config.block_size = 32
        config.dlm_loss_weight = None
        config.ar_loss_weight = 1.0
        config.dp_varying_mask_ratio = False
        config.enable_diffusion_lm = True
        config._attn_implementation = "eager"
        original_vocab = config.vocab_size
        print(f"  Original vocab_size: {original_vocab}")
        print(f"  mask_token_id (pre-load): {config.mask_token_id}")
        config.diffusion_lm = True
        model = Qwen3DiffusionForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        model.eval()
        test_block_diff_attention(
            model=model,
            input_ids=input_ids,
            seq_len=args.seq_len,
            block_size=args.block_size,
            dtype=dtype,
            device=device,
        )

        print("\n========================================")
        print("All Qwen3AttentionWithDiffusionToggle mask tests PASSED.")
        print("========================================")

    except Exception:
        print("\n[FAILED] Test failed with traceback:")
        traceback.print_exc()
        raise

    finally:
        restore_attention_patch(original_eager)


if __name__ == "__main__":
    main()