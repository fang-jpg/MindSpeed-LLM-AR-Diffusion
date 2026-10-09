#!/usr/bin/env python3
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Benchmark AR or block-diffusion decode speed on fixed GSM8K prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from gsm8k_experiment import iter_jsonl
from inference_qwen3_diffusion import (
    DTYPE_MAP,
    detect_device,
    generate_once,
    load_model,
    load_tokenizer,
    peak_memory_bytes,
    percentile,
    reset_peak_memory,
    synchronize,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure synchronized generation throughput. Tokenization and model load "
            "are excluded; prompt prefill and decoding are included."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--mode", required=True, choices=["ar", "diffusion"])
    parser.add_argument("--block-length", type=int, default=8)
    parser.add_argument("--steps-per-block", type=int, default=None)
    parser.add_argument("--seed-mode", choices=["none", "ar"], default="none")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=DTYPE_MAP, default="bfloat16")
    parser.add_argument("--config-diffusion", default=None)
    parser.add_argument(
        "--ar-baseline-json",
        default=None,
        help="AR benchmark JSON used to calculate speedup",
    )
    parser.add_argument(
        "--expected-prompts",
        type=int,
        default=100,
        help="fail if the frozen subset does not contain this many prompts",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def mean_metric(records: list[dict], key: str) -> float:
    values = [record[key] for record in records if record.get(key) is not None]
    return statistics.fmean(values) if values else 0.0


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_baseline(current: dict, baseline: dict) -> None:
    expected = {
        "model_path": current["model_path"],
        "prompts_sha256": current["prompts_sha256"],
        "temperature": current["temperature"],
        "max_new_tokens": current["max_new_tokens"],
        "prompt_count": current["prompt_count"],
        "warmup": current["warmup"],
        "repeats": current["repeats"],
        "ignore_eos": current["ignore_eos"],
        "seed": current["seed"],
        "dtype": current["dtype"],
    }
    mismatches = {
        key: (baseline.get(key), value)
        for key, value in expected.items()
        if baseline.get(key) != value
    }
    if mismatches:
        raise ValueError(f"AR baseline protocol does not match: {mismatches}")


def main():
    args = parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    rows = list(iter_jsonl(args.prompts_file))
    if len(rows) != args.expected_prompts:
        raise ValueError(
            f"Expected {args.expected_prompts} prompts, found {len(rows)} in "
            f"{args.prompts_file}"
        )
    if any("prompt" not in row for row in rows):
        raise ValueError("Every JSONL row must contain a 'prompt' field")

    device = args.device or detect_device()
    torch.manual_seed(args.seed)
    print(f"> Loading {args.model_path} on {device} ({args.dtype})")
    model = load_model(
        args.model_path, device, DTYPE_MAP[args.dtype], args.config_diffusion
    )
    tokenizer = load_tokenizer(args.model_path)

    # Tokenization and host-to-device copies are deliberately outside timing.
    prompts = [
        tokenizer.encode(row["prompt"], return_tensors="pt").to(device)
        for row in rows
    ]
    args.ignore_eos = True

    for warmup_index in range(args.warmup):
        generate_once(
            model,
            tokenizer,
            prompts[warmup_index % len(prompts)],
            args,
        )
    synchronize(device)
    reset_peak_memory(device)

    timings_ms: list[float] = []
    generation_stats: list[dict] = []
    generated_token_total = 0
    for repeat_index in range(args.repeats):
        for prompt_index, input_ids in enumerate(prompts):
            synchronize(device)
            started = time.perf_counter()
            output_ids, stats = generate_once(model, tokenizer, input_ids, args)
            synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000
            generated = output_ids.shape[1] - input_ids.shape[1]
            if generated != args.max_new_tokens:
                raise RuntimeError(
                    f"repeat={repeat_index} prompt={prompt_index}: expected "
                    f"{args.max_new_tokens} generated tokens, got {generated}"
                )
            timings_ms.append(elapsed_ms)
            generated_token_total += generated
            generation_stats.append(stats)

    total_seconds = sum(timings_ms) / 1000.0
    throughput = generated_token_total / total_seconds
    summary = {
        "schema_version": 1,
        "model_path": str(Path(args.model_path).resolve()),
        "prompts_file": str(Path(args.prompts_file).resolve()),
        "prompts_sha256": sha256_file(args.prompts_file),
        "mode": args.mode,
        "block_length": args.block_length if args.mode == "diffusion" else 1,
        "steps_per_block": (
            args.steps_per_block if args.mode == "diffusion" else 1
        ),
        "seed_mode": args.seed_mode if args.mode == "diffusion" else None,
        "temperature": args.temperature,
        "seed": args.seed,
        "device": str(device),
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": True,
        "prompt_count": len(prompts),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "measurement_count": len(timings_ms),
        "measurement_scope": (
            "synchronized prompt prefill plus generation; excludes model load, "
            "tokenization, and host-to-device prompt copy"
        ),
        "generated_tokens_total": generated_token_total,
        "measured_seconds_total": total_seconds,
        "decode_tokens_per_second": throughput,
        "mean_ms_per_prompt": statistics.fmean(timings_ms),
        "p50_ms_per_prompt": percentile(timings_ms, 0.50),
        "p95_ms_per_prompt": percentile(timings_ms, 0.95),
        "peak_memory_bytes": peak_memory_bytes(device),
        "peak_memory_gib": peak_memory_bytes(device) / (1024**3),
        "mean_total_nfe": mean_metric(generation_stats, "total_nfe"),
        "mean_decode_nfe": mean_metric(generation_stats, "decode_nfe"),
        "mean_real_tpf": mean_metric(generation_stats, "real_tpf"),
        "speedup_vs_ar": 1.0 if args.mode == "ar" else None,
    }

    if args.ar_baseline_json:
        baseline = json.loads(Path(args.ar_baseline_json).read_text(encoding="utf-8"))
        validate_baseline(summary, baseline)
        baseline_throughput = float(baseline["decode_tokens_per_second"])
        summary["ar_baseline_json"] = str(Path(args.ar_baseline_json))
        summary["speedup_vs_ar"] = throughput / baseline_throughput

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
