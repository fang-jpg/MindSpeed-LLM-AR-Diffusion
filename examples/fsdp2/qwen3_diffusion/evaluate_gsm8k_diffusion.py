#!/usr/bin/env python3
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Evaluate AR or block-diffusion generation on the full GSM8K test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from gsm8k_experiment import (
    DATASET_CONFIG,
    DATASET_NAME,
    build_gsm8k_prompt,
    choose_fewshot_examples,
    dataset_fingerprint,
    extract_prediction_answer,
    extract_reference_answer,
    iter_jsonl,
    load_gsm8k,
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
    output_dir = Path(args.output_dir)
    samples_path = output_dir / "samples.jsonl"
    summary_path = output_dir / "summary.json"
    protocol_path = output_dir / "run_protocol.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Completed result already exists: {summary_path}; pass --overwrite to replace"
        )
    if samples_path.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(
            f"Partial samples exist: {samples_path}; pass --resume or --overwrite"
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
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    previous_rows = []
    if args.resume and samples_path.exists() and not args.overwrite:
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
        previous_rows = list(iter_jsonl(samples_path))
        expected_indices = list(range(len(previous_rows)))
        actual_indices = [row.get("dataset_index") for row in previous_rows]
        if actual_indices != expected_indices:
            raise ValueError(
                "Partial samples must contain consecutive dataset_index values from 0"
            )
        if len(previous_rows) > evaluation_count:
            raise ValueError("Partial result is longer than this evaluation request")
        print(f"> Resuming {output_dir} from sample {len(previous_rows)}")
    else:
        protocol_path.write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    device = args.device or detect_device()
    torch.manual_seed(args.seed)
    print(f"> Loading {args.model_path} on {device} ({args.dtype})")
    model = load_model(
        args.model_path, device, DTYPE_MAP[args.dtype], args.config_diffusion
    )
    tokenizer = load_tokenizer(args.model_path)
    args.ignore_eos = False

    correct = sum(int(row["correct"]) for row in previous_rows)
    parsed = sum(int(row.get("predicted_answer") is not None) for row in previous_rows)
    generated_tokens = sum(int(row["generated_tokens"]) for row in previous_rows)
    total_nfe = sum(
        float(row.get("generation_stats", {}).get("total_nfe", 0))
        for row in previous_rows
    )
    open_mode = "a" if previous_rows else "w"
    with samples_path.open(open_mode, encoding="utf-8") as stream:
        for dataset_index in range(len(previous_rows), evaluation_count):
            sample = test[dataset_index]
            prompt = build_gsm8k_prompt(sample["question"], fewshot_examples)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            output_ids, stats = generate_once(model, tokenizer, input_ids, args)
            generated_ids = output_ids[0, input_ids.shape[1] :]
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
            predicted_answer = extract_prediction_answer(prediction)
            reference_answer = extract_reference_answer(sample["answer"])
            is_correct = (
                predicted_answer is not None
                and reference_answer is not None
                and predicted_answer == reference_answer
            )
            correct += int(is_correct)
            parsed += int(predicted_answer is not None)
            generated_tokens += int(generated_ids.numel())
            total_nfe += float(stats.get("total_nfe", 0))
            row = {
                "dataset_index": dataset_index,
                "question": sample["question"],
                "reference_answer": reference_answer,
                "prediction": prediction,
                "predicted_answer": predicted_answer,
                "correct": is_correct,
                "generated_tokens": int(generated_ids.numel()),
                "generation_stats": stats,
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if (dataset_index + 1) % 10 == 0 or dataset_index + 1 == evaluation_count:
                print(
                    f"[{dataset_index + 1}/{evaluation_count}] "
                    f"accuracy={correct / (dataset_index + 1):.4f} "
                    f"parse_rate={parsed / (dataset_index + 1):.4f}",
                    flush=True,
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
        "accuracy": correct / evaluation_count,
        "accuracy_percent": 100.0 * correct / evaluation_count,
        "correct": correct,
        "parse_rate": parsed / evaluation_count,
        "parse_rate_percent": 100.0 * parsed / evaluation_count,
        "parsed": parsed,
        "mean_generated_tokens": generated_tokens / evaluation_count,
        "mean_total_nfe": total_nfe / evaluation_count,
        "samples_file": samples_path.name,
        "run_protocol_file": protocol_path.name,
        "resumed_from_samples": len(previous_rows),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
