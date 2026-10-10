#!/usr/bin/env python3
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Benchmark AR or block-diffusion decode speed on fixed GSM8K prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
from distributed_experiment import (
    atomic_write_json,
    initialize_distributed,
    shard_sequence,
)
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
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
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
    parser.add_argument(
        "--rank-file-timeout-seconds",
        type=float,
        default=3600.0,
        help="maximum wait for slower ranks to publish benchmark JSON files",
    )
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=60.0,
        help=(
            "print per-rank measurement progress after this many seconds; "
            "set to 0 to disable"
        ),
    )
    return parser.parse_args()


def distributed_mean_metric(rank_payloads: list[dict], key: str) -> float:
    total = sum(payload["metric_sums"][key] for payload in rank_payloads)
    count = sum(payload["metric_counts"][key] for payload in rank_payloads)
    return total / count if count else 0.0


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
        "distributed_world_size": current["distributed_world_size"],
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
    context = initialize_distributed(args.device)
    output_path = Path(args.output_json)
    output_path_resolved = output_path.resolve()
    run_manifest_path = output_path.with_name(f"{output_path.stem}.run.json")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    if args.rank_file_timeout_seconds <= 0:
        raise ValueError("rank-file-timeout-seconds must be positive")
    if args.progress_interval_seconds < 0:
        raise ValueError("progress-interval-seconds must be non-negative")
    rows = list(iter_jsonl(args.prompts_file))
    if len(rows) != args.expected_prompts:
        raise ValueError(
            f"Expected {args.expected_prompts} prompts, found {len(rows)} in "
            f"{args.prompts_file}"
        )
    if any("prompt" not in row for row in rows):
        raise ValueError("Every JSONL row must contain a 'prompt' field")

    indexed_rows = list(enumerate(rows))
    local_rows = shard_sequence(indexed_rows, context.rank, context.world_size)
    if not local_rows:
        raise ValueError(
            f"Rank {context.rank} received no prompts; reduce distributed world size"
        )

    device = context.device or detect_device()
    torch.manual_seed(args.seed + context.rank)
    print(
        f"> [rank {context.rank}/{context.world_size}] Loading {args.model_path} "
        f"on {device} ({args.dtype}); prompts={len(local_rows)}",
        flush=True,
    )
    model = load_model(
        args.model_path, device, DTYPE_MAP[args.dtype], args.config_diffusion
    )
    tokenizer = load_tokenizer(args.model_path)

    # Tokenization and host-to-device copies are deliberately outside timing.
    prompts = [
        (
            prompt_index,
            tokenizer.encode(row["prompt"], return_tensors="pt").to(device),
        )
        for prompt_index, row in local_rows
    ]
    args.ignore_eos = True

    for warmup_index in range(args.warmup):
        generate_once(
            model,
            tokenizer,
            prompts[warmup_index % len(prompts)][1],
            args,
        )
    synchronize(device)
    reset_peak_memory(device)
    print(
        f"> [rank {context.rank}/{context.world_size}] Warmup complete; "
        "entering measurement start barrier",
        flush=True,
    )
    if context.is_main:
        run_id = uuid.uuid4().hex
        rank_payload_dir = (
            output_path.parent
            / ".rank_payloads"
            / output_path.stem
            / run_id
        ).resolve()
        atomic_write_json(
            run_manifest_path,
            {
                "schema_version": 1,
                "status": "running",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "world_size": context.world_size,
                "output_json": str(output_path_resolved),
                "rank_payload_dir": str(rank_payload_dir),
            },
        )
    context.barrier()
    print(
        f"> [rank {context.rank}/{context.world_size}] Start barrier complete",
        flush=True,
    )
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "world_size": context.world_size,
        "output_json": str(output_path_resolved),
    }
    manifest_mismatches = {
        key: (run_manifest.get(key), expected)
        for key, expected in expected_manifest.items()
        if run_manifest.get(key) != expected
    }
    if manifest_mismatches:
        raise ValueError(
            f"Benchmark run manifest does not match this process: "
            f"{manifest_mismatches}"
        )
    rank_payload_dir = Path(run_manifest["rank_payload_dir"])

    timings_ms: list[float] = []
    generation_stats: list[dict] = []
    generated_token_total = 0
    total_local_measurements = len(prompts) * args.repeats
    print(
        f"> [rank {context.rank}/{context.world_size}] Starting timed "
        f"measurement; samples={total_local_measurements}, "
        f"max_new_tokens={args.max_new_tokens}, "
        f"progress_interval={args.progress_interval_seconds:g}s",
        flush=True,
    )
    measurement_started = time.perf_counter()
    progress_output_seconds = 0.0
    next_progress_at = args.progress_interval_seconds
    for repeat_index in range(args.repeats):
        for prompt_index, input_ids in prompts:
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

            completed = len(timings_ms)
            now = time.perf_counter()
            active_elapsed = now - measurement_started - progress_output_seconds
            if (
                args.progress_interval_seconds > 0
                and active_elapsed >= next_progress_at
                and completed < total_local_measurements
            ):
                eta_seconds = (
                    active_elapsed / completed
                    * (total_local_measurements - completed)
                )
                progress_started = time.perf_counter()
                print(
                    f"> [rank {context.rank}/{context.world_size}] Speed "
                    f"progress {completed}/{total_local_measurements} "
                    f"({completed / total_local_measurements:.1%}); "
                    f"repeat={repeat_index + 1}/{args.repeats}, "
                    f"prompt_index={prompt_index}, "
                    f"elapsed={active_elapsed:.0f}s, eta={eta_seconds:.0f}s",
                    flush=True,
                )
                progress_output_seconds += (
                    time.perf_counter() - progress_started
                )
                next_progress_at = (
                    active_elapsed + args.progress_interval_seconds
                )

    synchronize(device)
    local_wall_seconds = (
        time.perf_counter() - measurement_started - progress_output_seconds
    )
    print(
        f"> [rank {context.rank}/{context.world_size}] Measurement complete; "
        f"samples={len(timings_ms)}, wall_seconds={local_wall_seconds:.2f}",
        flush=True,
    )
    local_payload = {
        "rank": context.rank,
        "prompt_count": len(prompts),
        "timings_ms": timings_ms,
        "metric_sums": {
            key: sum(
                float(record[key])
                for record in generation_stats
                if record.get(key) is not None
            )
            for key in ("total_nfe", "decode_nfe", "real_tpf")
        },
        "metric_counts": {
            key: sum(record.get(key) is not None for record in generation_stats)
            for key in ("total_nfe", "decode_nfe", "real_tpf")
        },
        "generated_tokens": generated_token_total,
        "parallel_wall_seconds": local_wall_seconds,
        "progress_output_seconds": progress_output_seconds,
        "peak_memory_bytes": peak_memory_bytes(device),
    }
    rank_payloads = context.all_gather_json_files(
        local_payload,
        rank_payload_dir,
        timeout_seconds=args.rank_file_timeout_seconds,
    )
    if not context.is_main:
        context.close()
        return 0

    timings_ms = [
        value
        for payload in rank_payloads
        for value in payload["timings_ms"]
    ]
    generated_token_total = sum(
        payload["generated_tokens"] for payload in rank_payloads
    )
    parallel_wall_seconds = max(
        payload["parallel_wall_seconds"] for payload in rank_payloads
    )
    peak_memory = max(payload["peak_memory_bytes"] for payload in rank_payloads)

    total_seconds = sum(timings_ms) / 1000.0
    throughput = generated_token_total / total_seconds
    aggregate_throughput = generated_token_total / parallel_wall_seconds
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
        "device": "data_parallel_npu" if context.enabled else str(device),
        "dtype": args.dtype,
        "distributed_world_size": context.world_size,
        "rank_result_exchange": "shared_json_files",
        "rank_run_manifest": str(run_manifest_path.resolve()),
        "rank_payload_dir": str(rank_payload_dir),
        "rank_payload_files": [
            f"rank{rank:02d}.json" for rank in range(context.world_size)
        ],
        "rank_file_timeout_seconds": args.rank_file_timeout_seconds,
        "progress_interval_seconds": args.progress_interval_seconds,
        "rank_progress_output_seconds": [
            payload["progress_output_seconds"] for payload in rank_payloads
        ],
        "local_prompt_counts": [
            payload["prompt_count"] for payload in rank_payloads
        ],
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": True,
        "prompt_count": len(rows),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "measurement_count": len(timings_ms),
        "measurement_scope": (
            "decode_tokens_per_second is per-request-equivalent throughput from "
            "summed synchronized sample times; aggregate_tokens_per_second uses "
            "multi-NPU wall time. Both include prompt prefill and generation, and "
            "exclude model load, tokenization, and host-to-device prompt copy"
        ),
        "generated_tokens_total": generated_token_total,
        "measured_seconds_total": total_seconds,
        "decode_tokens_per_second": throughput,
        "parallel_wall_seconds": parallel_wall_seconds,
        "aggregate_tokens_per_second": aggregate_throughput,
        "mean_ms_per_prompt": statistics.fmean(timings_ms),
        "p50_ms_per_prompt": percentile(timings_ms, 0.50),
        "p95_ms_per_prompt": percentile(timings_ms, 0.95),
        "peak_memory_bytes": peak_memory,
        "peak_memory_gib": peak_memory / (1024**3),
        "mean_total_nfe": distributed_mean_metric(rank_payloads, "total_nfe"),
        "mean_decode_nfe": distributed_mean_metric(rank_payloads, "decode_nfe"),
        "mean_real_tpf": distributed_mean_metric(rank_payloads, "real_tpf"),
        "speedup_vs_ar": 1.0 if args.mode == "ar" else None,
        "aggregate_speedup_vs_ar": 1.0 if args.mode == "ar" else None,
    }

    if args.ar_baseline_json:
        baseline = json.loads(Path(args.ar_baseline_json).read_text(encoding="utf-8"))
        validate_baseline(summary, baseline)
        baseline_throughput = float(baseline["decode_tokens_per_second"])
        baseline_aggregate = float(baseline["aggregate_tokens_per_second"])
        summary["ar_baseline_json"] = str(Path(args.ar_baseline_json))
        summary["speedup_vs_ar"] = throughput / baseline_throughput
        summary["aggregate_speedup_vs_ar"] = (
            aggregate_throughput / baseline_aggregate
        )

    atomic_write_json(output_path, summary)
    atomic_write_json(
        run_manifest_path,
        {
            **run_manifest,
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "summary_json": str(output_path_resolved),
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    context.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
