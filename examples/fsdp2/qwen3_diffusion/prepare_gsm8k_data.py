#!/usr/bin/env python3
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Freeze the 100-prompt GSM8K speed subset from an offline HF cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from gsm8k_experiment import (
    DATASET_CONFIG,
    DATASET_NAME,
    build_gsm8k_prompt,
    choose_fewshot_examples,
    choose_indices,
    dataset_fingerprint,
    load_gsm8k,
    resolve_datasets_cache,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a fixed GSM8K subset for decode-speed measurement"
    )
    parser.add_argument(
        "--cache-dir",
        default="/home/f50058111/huggingface_cache",
        help="HF_HOME root or HF_DATASETS_CACHE containing GSM8K",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-speed-prompts", type=int, default=100)
    parser.add_argument("--num-fewshot", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing export"
    )
    return parser.parse_args()


def stable_prompt_hash(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["prompt"].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def stable_json_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    prompts_path = output_dir / "speed_prompts_100.jsonl"
    manifest_path = output_dir / "gsm8k_protocol.json"
    fewshot_path = output_dir / "fewshot_examples.json"
    existing = [path for path in (prompts_path, manifest_path, fewshot_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Export already exists; pass --overwrite to replace: "
            + ", ".join(str(path) for path in existing)
        )

    train = load_gsm8k(args.cache_dir, "train")
    test = load_gsm8k(args.cache_dir, "test")
    fewshot_examples, fewshot_indices = choose_fewshot_examples(
        train, args.num_fewshot, args.seed
    )
    speed_indices = choose_indices(len(test), args.num_speed_prompts, args.seed)

    rows = []
    for order, dataset_index in enumerate(speed_indices):
        sample = test[dataset_index]
        rows.append(
            {
                "id": f"gsm8k-test-{dataset_index}",
                "order": order,
                "dataset_index": dataset_index,
                "question": sample["question"],
                "prompt": build_gsm8k_prompt(sample["question"], fewshot_examples),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    fewshot_payload = {
        "indices": fewshot_indices,
        "examples": [dict(example) for example in fewshot_examples],
    }
    fewshot_path.write_text(
        json.dumps(fewshot_payload, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "cache_root": str(Path(args.cache_dir)),
        "resolved_datasets_cache": str(resolve_datasets_cache(args.cache_dir)),
        "train_fingerprint": dataset_fingerprint(train),
        "test_fingerprint": dataset_fingerprint(test),
        "train_size": len(train),
        "test_size": len(test),
        "seed": args.seed,
        "num_fewshot": args.num_fewshot,
        "fewshot_indices": fewshot_indices,
        "fewshot_sha256": stable_json_hash(fewshot_payload["examples"]),
        "num_speed_prompts": len(rows),
        "speed_indices": speed_indices,
        "prompt_sha256": stable_prompt_hash(rows),
        "speed_prompts_file": prompts_path.name,
        "fewshot_examples_file": fewshot_path.name,
        "usage": "The 100 prompts are only for speed. Accuracy uses the full test split.",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
