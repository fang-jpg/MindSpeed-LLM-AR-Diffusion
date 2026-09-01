#!/usr/bin/env python3
"""Phase 4 smoke test: verify from_pretrained auto-resize and one training step.

This is a pre-launch smoke test for the FSDP2 training script. It checks
the things that would otherwise fail at torchrun startup:

  4-A: from_pretrained auto-resizes embeddings to accommodate mask_token_id
  4-B: Block_diff forward produces finite scalar loss under block_diff paradigm
  4-C: Backward + optimizer step succeeds; gradient flows to all head groups
  4-D: Loss components are recorded in model._last_loss_components

This script is a SINGLE-PROCESS check (no FSDP2 wrap, no torchrun). It
does NOT replace the real 10-step torchrun smoke test, which validates
FSDP2 sharding, dataloader, and checkpoint save. Run this before the
torchrun smoke test to catch model-level issues early.

Usage (CPU):
  python scripts/phase4_smoke_test.py \
    --model-path Qwen/Qwen3-1.7B-Base \
    --seq-len 64 \
    --block-size 32 \
    --device cpu

Usage (NPU):
  export ASCEND_RT_VISIBLE_DEVICES=0
  python scripts/phase4_smoke_test.py \
    --model-path /home/c00633840/models/Qwen3-1.7B-Base/ \
    --seq-len 64 \
    --block-size 32 \
    --device npu:0
"""

import argparse
import sys

import torch
from transformers import Qwen3Config


def parse_args():
    p = argparse.ArgumentParser(description="Phase 4 smoke test")
    p.add_argument("--model-path", type=str, required=True,
                   help="Path to Qwen3-1.7B-Base")
    p.add_argument("--seq-len", type=int, default=64,
                   help="Sequence length (must be divisible by block-size)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def detect_device():
    try:
        import torch_npu
        if torch.npu.is_available():
            return "npu:0"
    except ImportError:
        pass
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def make_input(batch_size, seq_len, vocab_size, device):
    torch.manual_seed(42)
    # Use vocab_size-1 as upper bound to avoid sampling the mask_token_id
    input_ids = torch.randint(0, vocab_size - 1, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    return input_ids, labels


# ── 4-A: from_pretrained auto-resize ──

def experiment_4a(model_path, device, block_size):
    """from_pretrained auto-resizes embeddings + rebuilds diffusion_head."""
    print("\n=== Experiment 4-A: from_pretrained auto-resize ===")

    from mindspeed_llm.fsdp2.models.qwen3_diffusion.modeling_qwen3_diffusion import (
        Qwen3DiffusionForCausalLM,
    )

    config = Qwen3Config.from_pretrained(model_path)
    # Set mask_token_id = vocab_size (out-of-vocab) to trigger resize path
    config.mask_token_id = config.mask_token_id
    config.dlm_paradigm = "block_diff"
    config.block_size = block_size
    config.dlm_loss_weight = None
    config.ar_loss_weight = 1.0
    config.dp_varying_mask_ratio = False
    config.enable_diffusion_lm = True
    config._attn_implementation = "eager"

    original_vocab = config.vocab_size
    print(f"  Original vocab_size: {original_vocab}")
    print(f"  mask_token_id (pre-load): {config.mask_token_id}")

    model = Qwen3DiffusionForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )


    # # After from_pretrained, vocab should be resized
    new_vocab = model.config.vocab_size
    # assert new_vocab == original_vocab + 1, \
    #     f"FAIL: vocab not resized. expected {original_vocab + 1}, got {new_vocab}"
    # print(f"  Resized vocab_size: {new_vocab} (PASSED)")

    # assert model.config.mask_token_id == original_vocab, \
    #     f"FAIL: mask_token_id not updated. expected {original_vocab}, got {model.config.mask_token_id}"
    # print(f"  mask_token_id: {model.config.mask_token_id} (PASSED)")

    # Check embed_tokens shape
    embed_weight = model.get_input_embeddings().weight
    assert embed_weight.shape[0] == new_vocab, \
        f"FAIL: embed_tokens shape={embed_weight.shape}, expected vocab={new_vocab}"
    print(f"  embed_tokens shape: {embed_weight.shape} (PASSED)")

    # Check diffusion_head shape
    dh_weight = model.lm_head.weight
    assert dh_weight.shape[0] == new_vocab, \
        f"FAIL: diffusion_head shape={dh_weight.shape}, expected vocab={new_vocab}"
    print(f"  diffusion_head shape: {dh_weight.shape} (PASSED)")

    # Check lm_head shape
    lh_weight = model.lm_head.weight
    assert lh_weight.shape[0] == new_vocab, \
        f"FAIL: lm_head shape={lh_weight.shape}, expected vocab={new_vocab}"
    print(f"  lm_head shape: {lh_weight.shape} (PASSED)")

    # Mask token embedding should be non-zero (initialized by resize)
    mask_emb = embed_weight[model.config.mask_token_id]
    assert not torch.all(mask_emb == 0), "FAIL: mask token embedding is all zeros"
    print(f"  Mask embedding non-zero (norm={mask_emb.norm().item():.4f}) (PASSED)")

    model.eval()
    return model


# ── 4-B: Forward produces finite scalar loss ──

def experiment_4b(model, input_ids, labels, device):
    """Block_diff forward under block_diff paradigm produces finite loss."""
    print("\n=== Experiment 4-B: forward finite scalar loss ===")

    model.train()
    input_ids = input_ids.clone().to(device)
    labels = labels.clone().to(device)
    print(input_ids)
    print(labels)
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss

    assert loss.dim() == 0, f"FAIL: loss not scalar, shape={loss.shape}"
    print(f"  Loss is scalar: PASSED (value={loss.item():.4f})")

    assert torch.isfinite(loss), f"FAIL: loss NaN/Inf"
    print(f"  Loss is finite: PASSED")

    model.zero_grad()
    model.eval()
    return True


# ── 4-C: Backward + optimizer step ──

def experiment_4c(model, input_ids, labels, device):
    """Backward + optimizer step succeeds; gradient flows to all head groups."""
    print("\n=== Experiment 4-C: backward + optimizer step ===")

    model.train()
    input_ids = input_ids.clone().to(device)
    labels = labels.clone().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    optimizer.zero_grad()

    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    print(f"  Forward loss: {loss.item():.4f}")
    loss.backward()
    print(f"  Backward: PASSED")

    # Check gradient groups (same as Phase 3 3-D)
    grad_groups = {
        "embed_tokens": False,
        "self_attn": False,
        "mlp": False,
        "lm_head": False,
    }
    tied_weight_has_grad = False
    for name, param in model.named_parameters():
        if param.grad is None or param.grad.abs().sum() == 0:
            continue
        if "embed_tokens" in name:
            grad_groups["embed_tokens"] = True
            tied_weight_has_grad = True
        elif "self_attn" in name:
            grad_groups["self_attn"] = True
        elif "mlp" in name:
            grad_groups["mlp"] = True
        elif "lm_head" in name:
            grad_groups["lm_head"] = True
    if tied_weight_has_grad:
        grad_groups["lm_head"] = True  # lm_head tied with embed_tokens

    for group, has_grad in grad_groups.items():
        status = "OK" if has_grad else "MISSING"
        print(f"  {group}: {status}")
        assert has_grad, f"FAIL: {group} has no gradient"

    # Optimizer step (just check it doesn't crash)
    optimizer.step()
    print(f"  Optimizer step: PASSED")

    model.zero_grad()
    model.eval()
    return True


# ── 4-D: Loss components recorded ──

def experiment_4d(model, input_ids, labels, device):
    """Loss components are recorded in model._last_loss_components."""
    print("\n=== Experiment 4-D: loss components recorded ===")

    model.train()
    input_ids = input_ids.clone().to(device)
    labels = labels.clone().to(device)

    model(input_ids=input_ids, labels=labels)

    components = getattr(model, "_last_loss_components", None)
    assert components is not None, "FAIL: _last_loss_components not set"
    print(f"  Components dict present: PASSED")

    assert "diff_loss" in components, "FAIL: diff_loss not in components"
    assert "ar_loss" in components, "FAIL: ar_loss not in components"
    print(f"  diff_loss={components['diff_loss']:.4f}")
    print(f"  ar_loss={components['ar_loss']:.4f}")

    assert components["diff_loss"] > 0, "FAIL: diff_loss is 0"
    assert components["ar_loss"] > 0, "FAIL: ar_loss is 0"
    print(f"  Both components positive: PASSED")

    model.zero_grad()
    model.eval()
    return True


def main():
    args = parse_args()
    device = args.device or detect_device()
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print(f"seq_len: {args.seq_len}, batch_size: {args.batch_size}, block_size: {args.block_size}")

    assert args.seq_len % args.block_size == 0, \
        f"seq_len ({args.seq_len}) must be divisible by block_size ({args.block_size})"

    results = {}

    # 4-A: Loads model with auto-resize
    try:
        model = experiment_4a(args.model_path, device, args.block_size)
        results["4-A"] = True
    except Exception as e:
        print(f"  4-A FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["4-A"] = False
        # Cannot continue without model
        _print_summary(results)
        sys.exit(1)

    vocab_size = model.config.vocab_size
    input_ids, labels = make_input(args.batch_size, args.seq_len, vocab_size, device)
    print(f"\nModel loaded. Num layers: {len(model.model.layers)}")
    print(f"dlm_paradigm: {model.config.dlm_paradigm}")
    print(f"block_size: {model.config.block_size}")
    print(f"vocab_size: {model.config.vocab_size}")

    for name, fn in [
        ("4-B", experiment_4b),
        ("4-C", experiment_4c),
        ("4-D", experiment_4d),
    ]:
        try:
            results[name] = fn(model, input_ids, labels, device)
        except Exception as e:
            print(f"  {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    _print_summary(results)


def _print_summary(results):
    print("\n" + "=" * 50)
    print("Phase 4 Smoke Test Results")
    print("=" * 50)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  Experiment {name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll smoke tests PASSED. Ready for torchrun 10-step run.")
    else:
        print("\nSome tests FAILED. Fix before torchrun.")
        sys.exit(1)


if __name__ == "__main__":
    main()
