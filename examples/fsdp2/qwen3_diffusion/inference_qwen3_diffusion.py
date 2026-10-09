#!/usr/bin/env python3
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Single-prompt AR, block-diffusion, and linear-spec inference."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3Config

from mindspeed_llm.fsdp2.models.qwen3_diffusion.generation import (
    GenerationStats,
    block_diff_generate,
)
from mindspeed_llm.fsdp2.models.qwen3_diffusion.linear_spec_generate import (
    ar_generate,
    linear_spec_generate,
)
from mindspeed_llm.fsdp2.models.qwen3_diffusion.modeling_qwen3_diffusion import (
    Qwen3DiffusionForCausalLM,
)

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3 AR/diffusion inference")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mode", required=True, choices=["ar", "diffusion", "linear_spec"])
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=8)
    parser.add_argument("--steps-per-block", type=int, default=None)
    parser.add_argument("--seed-mode", choices=["none", "ar"], default="none")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=DTYPE_MAP, default="bfloat16")
    parser.add_argument("--config-diffusion", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--stats-json", default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def detect_device() -> str:
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return "npu:0"
    except ImportError:
        pass
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def synchronize(device: str | torch.device) -> None:
    device = torch.device(device)
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: str | torch.device) -> None:
    device = torch.device(device)
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.reset_peak_memory_stats(device)
    elif device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_bytes(device: str | torch.device) -> int:
    device = torch.device(device)
    if device.type == "npu" and hasattr(torch, "npu"):
        return int(torch.npu.max_memory_allocated(device))
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return 0


def load_model(model_path, device, dtype, config_diffusion_path=None):
    """Load the checkpoint config, applying only explicit or missing defaults."""
    config = Qwen3Config.from_pretrained(model_path)
    if config_diffusion_path is not None:
        with open(config_diffusion_path, encoding="utf-8") as stream:
            for key, value in json.load(stream).items():
                setattr(config, key, value)

    defaults = {
        "mask_token_id": 151669,
        "dlm_paradigm": "block_diff",
        "block_size": 8,
        "dlm_loss_weight": 0.3,
        "ar_loss_weight": 1.0,
        "dp_varying_mask_ratio": True,
        "enable_diffusion_lm": True,
        "diffusion_lm": True,
    }
    for key, value in defaults.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    config._attn_implementation = "eager"

    model = Qwen3DiffusionForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()
    return model


def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def generate_once(model, tokenizer, input_ids, args):
    if args.mode == "ar":
        output_ids, nfe = ar_generate(
            model=model,
            prompt_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            topk=args.top_k,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
            ignore_eos=args.ignore_eos,
        )
        generated = output_ids.shape[1] - input_ids.shape[1]
        stats = {
            "prefill_nfe": 1,
            "denoise_nfe": 0,
            "refresh_nfe": max(0, nfe - 1),
            "total_nfe": nfe,
            "decode_nfe": max(0, nfe - 1),
            "real_tpf": generated / max(1, nfe),
        }
        return output_ids, stats

    if args.mode == "diffusion":
        output_ids, generation_stats = block_diff_generate(
            model=model,
            prompt_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            block_length=args.block_length,
            steps_per_block=args.steps_per_block,
            threshold=args.threshold if args.threshold > 0 else None,
            causal_context=True,
            temperature=args.temperature,
            topk=args.top_k,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
            tokenizer=tokenizer,
            print_block_predictions=args.debug,
            seed_mode=args.seed_mode,
            ignore_eos=args.ignore_eos,
            return_stats=True,
            debug=args.debug,
        )
        if not isinstance(generation_stats, GenerationStats):
            raise TypeError("block_diff_generate did not return GenerationStats")
        return output_ids, generation_stats.to_dict()

    output_ids, nfe, average_accept = linear_spec_generate(
        model=model,
        prompt_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        block_length=args.block_length,
        temperature=args.temperature,
        eos_token_id=tokenizer.eos_token_id,
        threshold=args.threshold,
    )
    generated = output_ids.shape[1] - input_ids.shape[1]
    return output_ids, {
        "total_nfe": nfe,
        "decode_nfe": max(0, nfe - 1),
        "real_tpf": generated / max(1, nfe),
        "average_accept_length": average_accept,
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main():
    args = parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    device = args.device or detect_device()
    torch.manual_seed(args.seed)

    print(f"> Loading {args.model_path} on {device} ({args.dtype})")
    model = load_model(args.model_path, device, DTYPE_MAP[args.dtype], args.config_diffusion)
    tokenizer = load_tokenizer(args.model_path)
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)

    for _ in range(args.warmup):
        generate_once(model, tokenizer, input_ids, args)
    synchronize(device)
    reset_peak_memory(device)

    durations_ms = []
    last_output = None
    last_stats = None
    for _ in range(args.repeats):
        synchronize(device)
        started = time.perf_counter()
        last_output, last_stats = generate_once(model, tokenizer, input_ids, args)
        synchronize(device)
        durations_ms.append((time.perf_counter() - started) * 1000)

    generated_tokens = last_output.shape[1] - input_ids.shape[1]
    total_generated = generated_tokens * args.repeats
    total_seconds = sum(durations_ms) / 1000
    summary = {
        "model_path": args.model_path,
        "mode": args.mode,
        "block_length": args.block_length if args.mode != "ar" else 1,
        "steps_per_block": args.steps_per_block,
        "seed_mode": args.seed_mode if args.mode == "diffusion" else None,
        "prompt_tokens": input_ids.shape[1],
        "generated_tokens_per_repeat": generated_tokens,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "mean_ms": statistics.fmean(durations_ms),
        "p50_ms": percentile(durations_ms, 0.50),
        "p95_ms": percentile(durations_ms, 0.95),
        "decode_tokens_per_second": total_generated / total_seconds,
        "peak_memory_bytes": peak_memory_bytes(device),
        "generation_stats": last_stats,
    }

    generated_text = tokenizer.decode(
        last_output[0, input_ids.shape[1]:], skip_special_tokens=True
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nGenerated text:\n" + generated_text)

    if args.stats_json:
        output_path = Path(args.stats_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
