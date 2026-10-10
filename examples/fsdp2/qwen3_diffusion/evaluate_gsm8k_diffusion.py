#!/usr/bin/env python3
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Evaluate AR or block-diffusion generation on the full GSM8K test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# These flags must be set before torch/transformers imports. In the NPU image,
# torch_npu can import huggingface_hub early, which otherwise caches online mode
# before load_gsm8k() gets a chance to select the local dataset cache.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from distributed_experiment import initialize_distributed, shard_sequence
from gsm8k_experiment import (
    DATASET_CONFIG,
    DATASET_NAME,
    GSM8K_STOP_SEQUENCES,
    build_gsm8k_prompt,
    choose_fewshot_examples,
    dataset_fingerprint,
    extract_prediction_answer,
    extract_reference_answer,
    iter_jsonl,
    load_gsm8k,
    truncate_at_stop_sequence,
)
from inference_qwen3_diffusion import (
    DTYPE_MAP,
    detect_device,
    generate_once,
    load_model,
    load_tokenizer,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run reproducible 5-shot GSM8K accuracy evaluation"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--cache-dir", default="/home/f50058111/huggingface_cache"
    )
    parser.add_argument("--mode", required=True, choices=["ar", "diffusion"])
    parser.add_argument("--block-length", type=int, default=8)
    parser.add_argument("--steps-per-block", type=int, default=None)
    parser.add_argument("--seed-mode", choices=["none", "ar"], default="none")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-fewshot", type=int, default=5)
    parser.add_argument(
        "--fewshot-file",
        default=None,
        help="frozen fewshot_examples.json produced by prepare_gsm8k_data.py",
    )
    parser.add_argument(
        "--data-protocol",
        default=None,
        help="gsm8k_protocol.json used to validate dataset and few-shot identity",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="smoke test only; omit for the formal full 1319-example run",
    )
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue a validated partial samples.jsonl run",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    context = initialize_distributed(args.device)
    output_dir = Path(args.output_dir)
    combined_samples_path = output_dir / "samples.jsonl"
    samples_path = (
        combined_samples_path
        if not context.enabled
        else output_dir / f"samples.rank{context.rank:02d}.jsonl"
    )
    summary_path = output_dir / "summary.json"
    protocol_path = output_dir / "run_protocol.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Completed result already exists: {summary_path}; pass --overwrite to replace"
        )
    existing_sample_files = list(output_dir.glob("samples*.jsonl"))
    if existing_sample_files and not (args.overwrite or args.resume):
        raise FileExistsError(
            f"Partial samples exist in {output_dir}; pass --resume or --overwrite"
        )
    if args.max_eval_samples is not None and args.max_eval_samples <= 0:
        raise ValueError("max-eval-samples must be positive")

    test = load_gsm8k(args.cache_dir, "test")
    if args.fewshot_file:
        fewshot_payload = json.loads(
            Path(args.fewshot_file).read_text(encoding="utf-8")
        )
        fewshot_examples = fewshot_payload["examples"]
        fewshot_indices = fewshot_payload["indices"]
        if len(fewshot_examples) != args.num_fewshot:
            raise ValueError(
                f"Expected {args.num_fewshot} frozen few-shot examples, found "
                f"{len(fewshot_examples)}"
            )
    else:
        train = load_gsm8k(args.cache_dir, "train")
        fewshot_examples, fewshot_indices = choose_fewshot_examples(
            train, args.num_fewshot, args.seed
        )
    fewshot_json = json.dumps(
        fewshot_examples, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    fewshot_sha256 = hashlib.sha256(fewshot_json).hexdigest()
    data_protocol_sha256 = None
    if args.data_protocol:
        data_protocol_path = Path(args.data_protocol)
        data_protocol_bytes = data_protocol_path.read_bytes()
        data_protocol_sha256 = hashlib.sha256(data_protocol_bytes).hexdigest()
        data_protocol = json.loads(data_protocol_bytes)
        expected_protocol = {
            "dataset": DATASET_NAME,
            "dataset_config": DATASET_CONFIG,
            "test_fingerprint": dataset_fingerprint(test),
            "seed": args.seed,
            "num_fewshot": args.num_fewshot,
            "fewshot_indices": fewshot_indices,
            "fewshot_sha256": fewshot_sha256,
        }
        mismatches = {
            key: (data_protocol.get(key), expected)
            for key, expected in expected_protocol.items()
            if data_protocol.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"Frozen GSM8K data protocol does not match: {mismatches}")
    evaluation_count = (
        len(test)
        if args.max_eval_samples is None
        else min(args.max_eval_samples, len(test))
    )

    effective_steps = (
        (args.steps_per_block or args.block_length)
        if args.mode == "diffusion"
        else 1
    )
    protocol = {
        "dataset": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "split": "test",
        "test_fingerprint": dataset_fingerprint(test),
        "evaluation_count": evaluation_count,
        "num_fewshot": args.num_fewshot,
        "fewshot_seed": args.seed,
        "fewshot_indices": fewshot_indices,
        "fewshot_sha256": fewshot_sha256,
        "data_protocol_sha256": data_protocol_sha256,
        "model_path": str(Path(args.model_path).resolve()),
        "mode": args.mode,
        "block_length": args.block_length if args.mode == "diffusion" else 1,
        "steps_per_block": effective_steps,
        "seed_mode": args.seed_mode if args.mode == "diffusion" else None,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "sample_seed_rule": "base_seed_plus_dataset_index",
        "distributed_world_size": context.world_size,
        "prompt_format": "lm-eval gsm8k v3 Question/Answer",
        "generation_stop_sequences": list(GSM8K_STOP_SEQUENCES),
        "answer_filter": "strict-marker-then-flexible-last-number",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and existing_sample_files and not args.overwrite:
        if not protocol_path.is_file():
            raise FileNotFoundError(
                f"Cannot validate partial run because {protocol_path} is missing"
            )
        stored_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if stored_protocol != protocol:
            raise ValueError(
                "Partial result protocol does not match this command. Use a new "
                "output directory or --overwrite."
            )
    elif context.is_main:
        protocol_path.write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    context.barrier()

    stored_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if stored_protocol != protocol:
        raise ValueError(
            "Run protocol does not match this command. Use a new output directory "
            "or --overwrite."
        )

    assigned_indices = shard_sequence(
        list(range(evaluation_count)), context.rank, context.world_size
    )
    previous_rows = []
    if args.resume and samples_path.exists() and not args.overwrite:
        previous_rows = list(iter_jsonl(samples_path))
        expected_indices = assigned_indices[: len(previous_rows)]
        actual_indices = [row.get("dataset_index") for row in previous_rows]
        if actual_indices != expected_indices:
            raise ValueError(
                f"Partial samples for rank {context.rank} are not the expected "
                "ordered shard prefix"
            )
        if len(previous_rows) > len(assigned_indices):
            raise ValueError(
                f"Partial result for rank {context.rank} is longer than its shard"
            )
        print(
            f"> [rank {context.rank}/{context.world_size}] Resuming from "
            f"{len(previous_rows)}/{len(assigned_indices)} local samples",
            flush=True,
        )

    local_correct = sum(int(row["correct"]) for row in previous_rows)
    local_parsed = sum(
        int(row.get("predicted_answer") is not None) for row in previous_rows
    )
    remaining_indices = assigned_indices[len(previous_rows) :]
    if remaining_indices:
        device = context.device or detect_device()
        torch.manual_seed(args.seed + context.rank)
        print(
            f"> [rank {context.rank}/{context.world_size}] Loading "
            f"{args.model_path} on {device} ({args.dtype}); "
            f"remaining_samples={len(remaining_indices)}",
            flush=True,
        )
        model = load_model(
            args.model_path, device, DTYPE_MAP[args.dtype], args.config_diffusion
        )
        tokenizer = load_tokenizer(args.model_path)
        args.ignore_eos = False
        open_mode = "a" if previous_rows else "w"
        stream = samples_path.open(open_mode, encoding="utf-8")
        try:
            for local_offset, dataset_index in enumerate(
                remaining_indices, start=len(previous_rows) + 1
            ):
                sample = test[dataset_index]
                prompt = build_gsm8k_prompt(sample["question"], fewshot_examples)
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                torch.manual_seed(args.seed + dataset_index)
                output_ids, stats = generate_once(model, tokenizer, input_ids, args)
                generated_ids = output_ids[0, input_ids.shape[1] :]
                raw_prediction = tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                prediction = truncate_at_stop_sequence(raw_prediction)
                predicted_answer = extract_prediction_answer(prediction)
                reference_answer = extract_reference_answer(sample["answer"])
                is_correct = (
                    predicted_answer is not None
                    and reference_answer is not None
                    and predicted_answer == reference_answer
                )
                local_correct += int(is_correct)
                local_parsed += int(predicted_answer is not None)
                row = {
                    "dataset_index": dataset_index,
                    "question": sample["question"],
                    "reference_answer": reference_answer,
                    "prediction": prediction,
                    "raw_prediction": raw_prediction,
                    "predicted_answer": predicted_answer,
                    "correct": is_correct,
                    "generated_tokens": int(generated_ids.numel()),
                    "generation_stats": stats,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                if (
                    local_offset % 10 == 0
                    or local_offset == len(assigned_indices)
                ):
                    print(
                        f"[rank {context.rank}: {local_offset}/"
                        f"{len(assigned_indices)}] "
                        f"local_accuracy={local_correct / local_offset:.4f} "
                        f"local_parse_rate={local_parsed / local_offset:.4f}",
                        flush=True,
                    )
        finally:
            stream.close()
    else:
        print(
            f"> [rank {context.rank}/{context.world_size}] Local shard already "
            f"complete ({len(previous_rows)}/{len(assigned_indices)}); "
            "skipping model load",
            flush=True,
        )

    resumed_counts = context.all_gather_object(len(previous_rows))
    context.barrier()
    if not context.is_main:
        context.barrier()
        context.close()
        return 0

    rank_sample_paths = (
        [combined_samples_path]
        if not context.enabled
        else [
            output_dir / f"samples.rank{rank:02d}.jsonl"
            for rank in range(context.world_size)
        ]
    )
    all_rows = [
        row
        for rank_path in rank_sample_paths
        for row in iter_jsonl(rank_path)
    ]
    all_rows.sort(key=lambda row: row["dataset_index"])
    actual_indices = [row.get("dataset_index") for row in all_rows]
    if actual_indices != list(range(evaluation_count)):
        raise RuntimeError(
            "Distributed GSM8K shards are incomplete or duplicated; refusing to "
            "write a summary"
        )
    if context.enabled:
        with combined_samples_path.open("w", encoding="utf-8") as stream:
            for row in all_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    correct = sum(int(row["correct"]) for row in all_rows)
    parsed = sum(
        int(row.get("predicted_answer") is not None) for row in all_rows
    )
    generated_tokens = sum(int(row["generated_tokens"]) for row in all_rows)
    total_nfe = sum(
        float(row.get("generation_stats", {}).get("total_nfe", 0))
        for row in all_rows
    )
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "split": "test",
        "test_fingerprint": dataset_fingerprint(test),
        "test_split_size": len(test),
        "evaluated_samples": evaluation_count,
        "formal_full_test": evaluation_count == len(test),
        "num_fewshot": args.num_fewshot,
        "fewshot_seed": args.seed,
        "fewshot_indices": fewshot_indices,
        "fewshot_sha256": fewshot_sha256,
        "data_protocol_sha256": data_protocol_sha256,
        "model_path": str(Path(args.model_path).resolve()),
        "mode": args.mode,
        "block_length": args.block_length if args.mode == "diffusion" else 1,
        "steps_per_block": effective_steps,
        "seed_mode": args.seed_mode if args.mode == "diffusion" else None,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "prompt_format": protocol["prompt_format"],
        "generation_stop_sequences": protocol["generation_stop_sequences"],
        "answer_filter": protocol["answer_filter"],
        "distributed_world_size": context.world_size,
        "accuracy": correct / evaluation_count,
        "accuracy_percent": 100.0 * correct / evaluation_count,
        "correct": correct,
        "parse_rate": parsed / evaluation_count,
        "parse_rate_percent": 100.0 * parsed / evaluation_count,
        "parsed": parsed,
        "mean_generated_tokens": generated_tokens / evaluation_count,
        "mean_total_nfe": total_nfe / evaluation_count,
        "samples_file": combined_samples_path.name,
        "rank_sample_files": [path.name for path in rank_sample_paths],
        "run_protocol_file": protocol_path.name,
        "resumed_from_samples": sum(resumed_counts),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    context.barrier()
    context.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
