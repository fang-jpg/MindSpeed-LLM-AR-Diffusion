#!/usr/bin/env python3
"""Phase 1 experiment: verify Qwen3AttentionWithDiffusionToggle works correctly.

Experiments:
  1-A: causal vs bidirectional output differ, shape same, no NaN
  1-B: all 28 layers bidirectional forward, no OOM/NaN
  1-C: diffusion_lm=False matches original HF Qwen3 forward
  1-D: single-layer toggle independent from other layers

Usage (NPU):
  export ASCEND_RT_VISIBLE_DEVICES=0
  export ASCEND_LAUNCH_BLOCKING=1
  python scripts/phase1_test_attention_toggle.py \
    --model-path /home/c00633840/models/Qwen3-1.7B-Base/ \
    --seq-len 64 \
    --attn-implementation eager

Usage (CPU, for dev):
  python scripts/phase1_test_attention_toggle.py \
    --model-path Qwen/Qwen3-1.7B-Base \
    --seq-len 32 \
    --attn-implementation eager \
    --device cpu
"""

import argparse
import sys

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 attention toggle experiments")
    p.add_argument("--model-path", type=str, required=True,
                   help="Path to Qwen3-1.7B-Base (local dir or HF hub id)")
    p.add_argument("--seq-len", type=int, default=64,
                   help="Sequence length for test inputs")
    p.add_argument("--batch-size", type=int, default=2,
                   help="Batch size for test inputs")
    p.add_argument("--attn-implementation", type=str, default="eager",
                   choices=["eager", "sdpa"],
                   help="Attention implementation to use")
    p.add_argument("--device", type=str, default=None,
                   help="Device (auto-detected if not set)")
    p.add_argument("--skip-1c", action="store_true",
                   help="Skip experiment 1-C (original model comparison, slower)")
    return p.parse_args()


def detect_device():
    if torch.npu.is_available():
        return "npu:0"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def load_model(model_path, attn_impl, device):
    """Load Qwen3-1.7B and swap in our diffusion-toggleable attention class.

    AutoModelForCausalLM.from_pretrained() always uses the standard HF
    Qwen3ForCausalLM, so we must call _swap_attention manually after
    loading to inject Qwen3AttentionWithDiffusionToggle.
    """
    from transformers import AutoModelForCausalLM, Qwen3Config
    from mindspeed_llm.fsdp2.models.qwen3.qwen3_attention import (
        Qwen3AttentionWithDiffusionToggle,
        _swap_attention,
    )

    config = Qwen3Config.from_pretrained(model_path)
    config._attn_implementation = attn_impl

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    # Swap attention modules after pretrained weights are loaded
    num_swapped = 0
    for layer in model.model.layers:
        _swap_attention(layer, Qwen3AttentionWithDiffusionToggle)
        num_swapped += 1
    print(f"  Swapped {num_swapped} attention layers to Qwen3AttentionWithDiffusionToggle")

    # Verify swap succeeded
    assert hasattr(model.model.layers[0].self_attn, "diffusion_lm"), \
        "Swap failed: layer 0 self_attn has no diffusion_lm attribute"
    return model


def set_diffusion_lm(model, value):
    """Set diffusion_lm on all attention layers."""
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "diffusion_lm"):
            layer.self_attn.diffusion_lm = value


def make_input(batch_size, seq_len, vocab_size, device):
    torch.manual_seed(42)
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)


@torch.no_grad()
def experiment_1a(model, input_ids):
    """Experiment 1-A: causal vs bidirectional output differ."""
    print("\n=== Experiment 1-A: causal vs bidirectional ===")

    set_diffusion_lm(model, False)
    out_causal = model(input_ids).logits

    set_diffusion_lm(model, True)
    out_bidir = model(input_ids).logits

    # Shape check
    assert out_causal.shape == out_bidir.shape, \
        f"Shape mismatch: causal={out_causal.shape}, bidir={out_bidir.shape}"
    print(f"  Shape check PASSED: {out_causal.shape}")

    # Values differ
    max_diff = (out_causal - out_bidir).abs().max().item()
    mean_diff = (out_causal - out_bidir).abs().mean().item()
    assert not torch.allclose(out_causal, out_bidir, atol=1e-3), \
        "FAIL: causal and bidir outputs are too close"
    print(f"  Value diff check PASSED: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    # No NaN/Inf
    assert torch.isfinite(out_causal).all(), "FAIL: causal output has NaN/Inf"
    assert torch.isfinite(out_bidir).all(), "FAIL: bidir output has NaN/Inf"
    print(f"  NaN/Inf check PASSED")

    return True


@torch.no_grad()
def experiment_1b(model, input_ids):
    """Experiment 1-B: all layers bidirectional forward, no OOM/NaN."""
    print("\n=== Experiment 1-B: full-model bidirectional forward ===")

    set_diffusion_lm(model, True)
    out = model(input_ids).logits

    assert torch.isfinite(out).all(), "FAIL: bidir full-model output has NaN/Inf"
    print(f"  Shape: {out.shape}")
    print(f"  NaN/Inf check PASSED")

    return True


@torch.no_grad()
def experiment_1c(model_path, input_ids, attn_impl, device):
    """Experiment 1-C: diffusion_lm=False matches original HF Qwen3."""
    print("\n=== Experiment 1-C: patched (causal) vs original HF ===")

    from transformers import AutoModelForCausalLM, Qwen3Config

    # Load original model (no patching)
    config_orig = Qwen3Config.from_pretrained(model_path)
    config_orig._attn_implementation = attn_impl
    model_orig = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config_orig,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model_orig.eval()

    with torch.no_grad():
        out_orig = model_orig(input_ids).logits

    del model_orig

    # Load patched model (diffusion_lm=False = causal)
    config_patched = Qwen3Config.from_pretrained(model_path)
    config_patched._attn_implementation = attn_impl
    model_patched = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config_patched,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model_patched.eval()

    # Swap attention manually (same as load_model)
    from mindspeed_llm.fsdp2.models.qwen3.qwen3_attention import (
        Qwen3AttentionWithDiffusionToggle,
        _swap_attention,
    )
    for layer in model_patched.model.layers:
        _swap_attention(layer, Qwen3AttentionWithDiffusionToggle)

    with torch.no_grad():
        out_patched = model_patched(input_ids).logits

    del model_patched

    max_diff = (out_orig - out_patched).abs().max().item()
    mean_diff = (out_orig - out_patched).abs().mean().item()

    # bf16 tolerance: max diff should be small (weight copy + same computation)
    TOLERANCE = 0.01
    if max_diff < TOLERANCE:
        print(f"  PASSED: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} (< {TOLERANCE})")
    else:
        print(f"  WARNING: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} "
              f"(> {TOLERANCE}, may be acceptable for bf16)")

    return max_diff < TOLERANCE


@torch.no_grad()
def experiment_1d(model, input_ids):
    """Experiment 1-D: single-layer toggle independence."""
    print("\n=== Experiment 1-D: per-layer toggle ===")

    # All causal
    set_diffusion_lm(model, False)
    out_all_causal = model(input_ids).logits

    # Only layer 0 bidirectional
    set_diffusion_lm(model, False)
    model.model.layers[0].self_attn.diffusion_lm = True
    out_l0_bidir = model(input_ids).logits

    # All bidirectional
    set_diffusion_lm(model, True)
    out_all_bidir = model(input_ids).logits

    # Layer-0-only bidir should differ from all-causal
    diff_vs_causal = (out_all_causal - out_l0_bidir).abs().max().item()
    # Layer-0-only bidir should differ from all-bidir
    diff_vs_bidir = (out_l0_bidir - out_all_bidir).abs().max().item()

    assert diff_vs_causal > 1e-3, \
        f"FAIL: layer-0-only bidir too close to all-causal (diff={diff_vs_causal:.6f})"
    assert diff_vs_bidir > 1e-3, \
        f"FAIL: layer-0-only bidir too close to all-bidir (diff={diff_vs_bidir:.6f})"
    print(f"  vs all-causal max_diff={diff_vs_causal:.6f}")
    print(f"  vs all-bidir  max_diff={diff_vs_bidir:.6f}")
    print(f"  Per-layer toggle PASSED")

    # Reset to default
    set_diffusion_lm(model, False)
    return True


def main():
    args = parse_args()
    device = args.device or detect_device()
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print(f"attn_implementation: {args.attn_implementation}")
    print(f"seq_len: {args.seq_len}, batch_size: {args.batch_size}")

    model = load_model(args.model_path, args.attn_implementation, device)
    vocab_size = model.config.vocab_size
    input_ids = make_input(args.batch_size, args.seq_len, vocab_size, device)
    print(f"Model loaded. Num layers: {len(model.model.layers)}")

    results = {}

    try:
        results["1-A"] = experiment_1a(model, input_ids)
    except Exception as e:
        print(f"  1-A FAILED: {e}")
        results["1-A"] = False

    try:
        results["1-B"] = experiment_1b(model, input_ids)
    except Exception as e:
        print(f"  1-B FAILED: {e}")
        results["1-B"] = False

    try:
        results["1-D"] = experiment_1d(model, input_ids)
    except Exception as e:
        print(f"  1-D FAILED: {e}")
        results["1-D"] = False

    if not args.skip_1c:
        try:
            results["1-C"] = experiment_1c(
                args.model_path, input_ids, args.attn_implementation, device)
        except Exception as e:
            print(f"  1-C FAILED: {e}")
            results["1-C"] = False

    print("\n" + "=" * 50)
    print("Phase 1 Experiment Results")
    print("=" * 50)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  Experiment {name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll experiments PASSED. Phase 1 attention toggle verified.")
    else:
        print("\nSome experiments FAILED. See details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
