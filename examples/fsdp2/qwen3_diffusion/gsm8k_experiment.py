# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Reproducible GSM8K data and answer helpers for AR/diffusion experiments."""

from __future__ import annotations

import json
import os
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path

DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"

_NUMBER_RE = re.compile(
    r"[-+]?(?:[$£€]\s*)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
)
_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|=|:)\s*"
    r"([-+]?(?:[$£€]\s*)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)",
    flags=re.IGNORECASE,
)


def resolve_datasets_cache(cache_root: str | Path) -> Path:
    """Accept either an HF_HOME root or its nested datasets directory."""
    root = Path(cache_root).expanduser()
    nested = root / "datasets"
    return nested if nested.is_dir() else root


def load_gsm8k(cache_root: str | Path, split: str):
    """Load GSM8K strictly from an existing Hugging Face datasets cache."""
    cache_dir = resolve_datasets_cache(cache_root)
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Hugging Face cache does not exist: {cache_dir}. "
            "Point --cache-dir at HF_HOME or HF_DATASETS_CACHE on the server."
        )

    # Set these before importing datasets so an offline experiment never
    # silently downloads a newer dataset revision.
    os.environ["HF_HOME"] = str(Path(cache_root).expanduser())
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    try:
        from datasets import DownloadMode, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Run with the Python environment "
            "from /home/f50058111/lm-eval-harness."
        ) from exc

    try:
        return load_dataset(
            DATASET_NAME,
            DATASET_CONFIG,
            split=split,
            cache_dir=str(cache_dir),
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {DATASET_NAME}/{DATASET_CONFIG}:{split} from "
            f"offline cache {cache_dir}. Confirm the server cache already contains "
            "GSM8K, or populate it once before enabling offline mode."
        ) from exc


def dataset_fingerprint(dataset) -> str | None:
    return getattr(dataset, "_fingerprint", None)


def choose_indices(population_size: int, sample_size: int, seed: int) -> list[int]:
    """Choose a deterministic subset while preserving sampled order."""
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if not 0 < sample_size <= population_size:
        raise ValueError(
            f"sample_size must be in [1, {population_size}], got {sample_size}"
        )
    return random.Random(seed).sample(range(population_size), sample_size)


def choose_fewshot_examples(train_dataset, num_fewshot: int, seed: int):
    if num_fewshot < 0:
        raise ValueError("num_fewshot must be non-negative")
    if num_fewshot == 0:
        return [], []
    indices = choose_indices(len(train_dataset), num_fewshot, seed)
    return [train_dataset[index] for index in indices], indices


def build_gsm8k_prompt(
    question: str,
    fewshot_examples: Sequence[Mapping[str, str]] = (),
) -> str:
    """Use the same Question/Answer form as the lm-eval GSM8K task."""
    sections = []
    for example in fewshot_examples:
        sections.append(
            f"Question: {example['question'].strip()}\n"
            f"Answer: {example['answer'].strip()}"
        )
    sections.append(f"Question: {question.strip()}\nAnswer:")
    return "\n\n".join(sections)


def normalize_numeric_answer(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().replace(",", "")
    cleaned = re.sub(r"^[$£€]\s*", "", cleaned)
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized in {"-0", "+0", ""} else normalized


def extract_reference_answer(answer: str) -> str | None:
    """Extract the official numeric target following GSM8K's #### marker."""
    tail = answer.rsplit("####", maxsplit=1)[-1]
    matches = _NUMBER_RE.findall(tail)
    return normalize_numeric_answer(matches[-1]) if matches else None


def extract_prediction_answer(prediction: str) -> str | None:
    """Extract a model's final numeric answer with a conservative fallback."""
    if "####" in prediction:
        marked = prediction.rsplit("####", maxsplit=1)[-1]
        matches = _NUMBER_RE.findall(marked)
        if matches:
            return normalize_numeric_answer(matches[0])

    explicit = list(_ANSWER_RE.finditer(prediction))
    if explicit:
        return normalize_numeric_answer(explicit[-1].group(1))

    matches = _NUMBER_RE.findall(prediction)
    return normalize_numeric_answer(matches[-1]) if matches else None


def iter_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc

