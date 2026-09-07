#!/usr/bin/env python3
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
# Phase 6: Evaluate script for Qwen3DiffusionForCausalLM.
#
# Supports three inference modes (AR / Diffusion / Linear-Spec) for
# test-set accuracy evaluation.  Currently supports MMLU‑style CSV
# datasets  (columns: question, A, B, C, D, answer).  The
# script loads each subject CSV from `--data-path`, runs inference
# with the selected mode, extracts the answer letter (A/B/C/D) from
# the generated text, and reports per‑subject and overall accuracy.
#
# Usage:
#   # AR mode (default)
#   python evaluate_qwen3_diffusion.py \
#     --model-path /path/to/Qwen3-1.7B-Base/ \
#     --data-path /path/to/testdata/mmlu/data/test/ \
#     --mode ar \
#     --task mmlu
#
#   # Diffusion mode
#   python evaluate_qwen3_diffusion.py \
#     --model-path /path/to/Qwen3-1.7B-Base/ \
#     --data-path /path/to/testdata/mmlu/data/test/ \
#     --mode diffusion \
#     --block-length 32
#
#   # Linear-Spec mode
#   python evaluate_qwen3_diffusion.py \
#     --model-path /path/to/Qwen3-1.7B-Base/ \
#     --data-path /path/to/testdata/mmlu/data/test/ \
#     --mode linear_spec \
#     --block-length 32

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen3Config

from mindspeed_llm.fsdp2.models.qwen3_diffusion.modeling_qwen3_diffusion import (
    Qwen3DiffusionForCausalLM,
)
from mindspeed_llm.fsdp2.models.qwen3_diffusion.generation import block_diff_generate
from mindspeed_llm.fsdp2.models.qwen3_diffusion.linear_spec_generate import (
    ar_generate,
    linear_spec_generate,
)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    """A single evaluation sample."""
    question: str
    choices: List[str]    # [A_text, B_text, C_text, D_text]
    answer: str           # "A", "B", "C", or "D"


@dataclass
class SubjectResult:
    """Accuracy result for one subject."""
    subject: str
    correct: int = 0
    total: int = 0
    details: List[dict] = field(default_factory=list)  # per-sample info

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Data loading (MMLU-style CSV)
# ──────────────────────────────────────────────────────────────────────────────

def load_subject_csv(file_path: str) -> List[EvalSample]:
    """Load a single MMLU-style CSV file.

    Expected format (no header):
        question, A, B, C, D, answer

    Returns:
        List of EvalSample
    """
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            question, choice_a, choice_b, choice_c, choice_d, answer = row[:6]
            answer = answer.strip().upper()
            if answer not in ("A", "B", "C", "D"):
                # Some files store the full answer text instead of letter
                # Try to map it back
                choices = [choice_a.strip(), choice_b.strip(), choice_c.strip(), choice_d.strip()]
                letter_map = {c: l for c, l in zip(choices, ["A", "B", "C", "D"])}
                answer = letter_map.get(answer.strip(), "A")
            samples.append(EvalSample(
                question=question.strip(),
                choices=[choice_a.strip(), choice_b.strip(), choice_c.strip(), choice_d.strip()],
                answer=answer,
            ))
    return samples


def build_prompt(sample: EvalSample, with_choices: bool = True) -> str:
    """Build an instruction prompt for a single MMLU sample."""
    if with_choices:
        prompt = (
            f"Question: {sample.question}\n"
            f"A. {sample.choices[0]}\n"
            f"B. {sample.choices[1]}\n"
            f"C. {sample.choices[2]}\n"
            f"D. {sample.choices[3]}\n"
            f"Answer:"
        )
    else:
        prompt = f"Question: {sample.question}\nAnswer:"
    return prompt


# ──────────────────────────────────────────────────────────────────────────────
# Answer extraction
# ──────────────────────────────────────────────────────────────────────────────

_ANSWER_PATTERNS = [
    re.compile(r"(?i)answer\s*is\s*[：:]\s*([A-D])"),
    re.compile(r"(?i)answer\s*[：:]\s*([A-D])"),
    re.compile(r"(?i)^\s*([A-D])\s*[.。]?\s*$"),
    re.compile(r"(?i)[\(（]\s*([A-D])\s*[\)）]"),
    re.compile(r"(?i)([A-D])\s*[.。]?\s*$"),
]


def extract_answer(text: str) -> Optional[str]:
    """Extract the answer letter (A/B/C/D) from generated text.

    Tries several regex patterns in order of specificity, returning
    the first match.  Returns None when no answer is found.
    """
    text = text.strip()
    for pattern in _ANSWER_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1).upper()
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Device / dtype helpers
# ──────────────────────────────────────────────────────────────────────────────

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


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_path, device, dtype, config_diffusion_path=None):
    """Load Qwen3DiffusionForCausalLM with diffusion config."""
    config = Qwen3Config.from_pretrained(model_path)

    if config_diffusion_path is not None:
        import json
        with open(config_diffusion_path, "r") as f:
            diff_cfg = json.load(f)
        for k, v in diff_cfg.items():
            setattr(config, k, v)
    else:
        config.mask_token_id = config.vocab_size
        config.dlm_paradigm = "block_diff"
        config.block_size = 8
        config.dlm_loss_weight = None
        config.ar_loss_weight = 1.0
        config.dp_varying_mask_ratio = False
        config.enable_diffusion_lm = True
        config.diffusion_lm = True

    config._attn_implementation = "eager"

    model = Qwen3DiffusionForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Inference dispatcher
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_answer(
    model,
    input_ids: torch.Tensor,
    mode: str,
    max_new_tokens: int,
    block_length: int,
    temperature: float,
    threshold: float,
    eos_token_id: int,
):
    """Run inference in the selected mode and return generated token IDs."""
    if mode == "ar":
        output_ids, nfe = ar_generate(
            model=model,
            prompt_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=eos_token_id,
        )
    elif mode == "diffusion":
        # Ensure max_new_tokens is divisible by block_length
        max_new = max_new_tokens
        if max_new % block_length != 0:
            max_new = max_new + (block_length - max_new % block_length)
        output_ids, nfe = block_diff_generate(
            model=model,
            prompt_ids=input_ids,
            max_new_tokens=max_new,
            block_length=block_length,
            threshold=threshold if threshold > 0 else None,
            causal_context=True,
            temperature=temperature,
            eos_token_id=eos_token_id,
        )
    elif mode == "linear_spec":
        output_ids, nfe, avg_accept = linear_spec_generate(
            model=model,
            prompt_ids=input_ids,
            max_new_tokens=max_new_tokens,
            block_length=block_length,
            temperature=temperature,
            eos_token_id=eos_token_id,
            threshold=threshold,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return output_ids, nfe


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_subject(
    model,
    tokenizer,
    subject: str,
    samples: List[EvalSample],
    mode: str,
    max_new_tokens: int,
    block_length: int,
    temperature: float,
    threshold: float,
    device: str,
    max_eval_samples: Optional[int] = None,
    verbose: bool = False,
    pbar: Optional[tqdm] = None,
) -> SubjectResult:
    """Evaluate a single subject and return accuracy result."""
    result = SubjectResult(subject=subject)

    if max_eval_samples is not None:
        samples = samples[:max_eval_samples]

    result.total = len(samples)

    # Per-subject progress bar
    sample_pbar = tqdm(
        samples,
        desc=f"  {subject[:30]:<30}",
        unit="sample",
        leave=False,
        position=1 if pbar else 0,
    )

    for idx, sample in enumerate(sample_pbar):
        # Build prompt
        prompt = build_prompt(sample, with_choices=True)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        prompt_len = input_ids.shape[1]

        # Truncate if too long (model has max_position_embeddings limit)
        max_pos = getattr(model.config, "max_position_embeddings", 32768)
        if prompt_len + max_new_tokens > max_pos:
            # Truncate from the left (preserve the last part with choices)
            max_prompt_len = max_pos - max_new_tokens
            input_ids = input_ids[:, -max_prompt_len:]

        # Generate
        output_ids, nfe = generate_answer(
            model=model,
            input_ids=input_ids,
            mode=mode,
            max_new_tokens=max_new_tokens,
            block_length=block_length,
            temperature=temperature,
            threshold=threshold,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode only the generated part (skip the prompt)
        generated_text = tokenizer.decode(
            output_ids[0, prompt_len:],
            skip_special_tokens=True,
        )

        # Extract answer
        predicted = extract_answer(generated_text)
        correct = predicted == sample.answer

        if correct:
            result.correct += 1

        detail = {
            "question": sample.question,
            "expected": sample.answer,
            "predicted": predicted,
            "correct": correct,
            "generated": generated_text[:100] if verbose else "",
            "nfe": nfe,
        }
        result.details.append(detail)

        # Update per-subject progress bar postfix
        running_acc = result.correct / (idx + 1)
        sample_pbar.set_postfix(acc=f"{running_acc:.0%}", nfe=nfe)

        if verbose:
            status = "✓" if correct else "✗"
            tqdm.write(f"  [{idx+1}/{result.total}] {status} "
                       f"expected={sample.answer}, predicted={predicted}")

        # Advance the global (file-level) progress bar by one sample
        if pbar is not None:
            pbar.update(1)

    sample_pbar.close()

    # Print per-subject summary line
    tqdm.write(f"  {subject:<30}  acc={result.accuracy:.2%}  ({result.correct}/{result.total})")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def print_report(results: List[SubjectResult], elapsed: float):
    """Print a summary report of all subjects."""
    total_correct = sum(r.correct for r in results)
    total_samples = sum(r.total for r in results)
    total_nfe = sum(
        d["nfe"] for r in results for d in r.details
    )

    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    print(f"{'Subject':<30} {'Correct':>8} {'Total':>6} {'Acc':>8}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x.subject):
        print(f"{r.subject:<30} {r.correct:>8} {r.total:>6} {r.accuracy:>7.2%}")
    print("-" * 70)
    print(f"{'TOTAL':<30} {total_correct:>8} {total_samples:>6} "
          f"{total_correct / total_samples:>7.2%}" if total_samples > 0 else "")
    print(f"\n  Total samples: {total_samples}")
    print(f"  Total NFEs:    {total_nfe}")
    print(f"  Wall time:     {elapsed:.2f}s")
    if total_nfe > 0:
        print(f"  Samples/NFE:   {total_samples / total_nfe:.4f}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Qwen3-1.7B-Diffusion on test datasets"
    )
    # Model
    p.add_argument(
        "--model-path", type=str, required=True,
        help="Path to Qwen3-1.7B-Base model directory"
    )
    p.add_argument(
        "--config-diffusion", type=str, default=None,
        help="Path to config_diffusion.json to inject diffusion fields"
    )
    # Data
    p.add_argument(
        "--data-path", type=str, required=True,
        help="Path to test dataset directory (e.g. /home/c00633840/testdata/mmlu/data/test/)"
    )
    p.add_argument(
        "--task", type=str, default="mmlu",
        help="Evaluation task name (currently only 'mmlu' for CSV format)"
    )
    p.add_argument(
        "--max-eval-samples", type=int, default=None,
        help="Max samples per subject (for quick debugging)"
    )
    # Inference mode
    p.add_argument(
        "--mode", type=str, default="ar",
        choices=["ar", "diffusion", "linear_spec"],
        help="Inference mode"
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=10,
        help="Max new tokens to generate per sample (MMLU only needs 1-2 tokens)"
    )
    p.add_argument(
        "--block-length", type=int, default=32,
        help="Block length for diffusion / linear_spec modes"
    )
    p.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (0 = greedy argmax)"
    )
    p.add_argument(
        "--threshold", type=float, default=0.0,
        help="Confidence threshold for diffusion denoising"
    )
    # Hardware
    p.add_argument(
        "--device", type=str, default=None,
        help="Device (auto-detected if not set)"
    )
    p.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype"
    )
    # Misc
    p.add_argument(
        "--verbose", action="store_true", default=False,
        help="Print per-sample inference details"
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or detect_device()
    dtype = DTYPE_MAP[args.dtype]

    print("=" * 70)
    print("Qwen3-1.7B Diffusion — Evaluation")
    print("=" * 70)
    print(f"  Model:         {args.model_path}")
    print(f"  Data path:     {args.data_path}")
    print(f"  Task:          {args.task}")
    print(f"  Mode:          {args.mode}")
    print(f"  Device:        {device}")
    print(f"  Dtype:         {args.dtype}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Block length:  {args.block_length}")
    print(f"  Temperature:   {args.temperature}")
    print(f"  Threshold:     {args.threshold}")
    if args.max_eval_samples:
        print(f"  Max samples:   {args.max_eval_samples}")
    print()

    # ── Validate data path ──
    if not os.path.isdir(args.data_path):
        print(f"[ERROR] Data path does not exist: {args.data_path}")
        print("  Please ensure the test dataset is available at the specified path.")
        print(f"  Expected format: {args.data_path}/<subject>_test.csv")
        return 1

    # ── Load model ──
    print("> Loading model...")
    t0 = time.time()
    model = load_model(args.model_path, device, dtype, args.config_diffusion)
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    print(f"  Num layers:   {len(model.model.layers)}")
    print(f"  Vocab size:   {model.config.vocab_size}")
    print(f"  Mask token:   {model.mask_token_id}")
    print()

    # ── Load tokenizer ──
    print("> Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print()

    # ── Discover CSV files ──
    csv_files = sorted([
        f for f in os.listdir(args.data_path)
        if f.endswith(".csv")
    ])
    if not csv_files:
        print(f"[ERROR] No CSV files found in {args.data_path}")
        return 1
    print(f"> Found {len(csv_files)} subject files:")
    for f in csv_files:
        print(f"    {f}")
    print()

    # ── Compute total samples per subject for the global progress bar ──
    print("> Pre-scanning dataset sizes...")
    subject_csv_pairs = []
    total_eval_samples = 0
    for csv_file in csv_files:
        file_path = os.path.join(args.data_path, csv_file)
        subject = re.sub(r"(?:_test|_val|_dev)?\.csv$", "", csv_file, flags=re.IGNORECASE)
        subject = subject.replace("_", " ")
        samples = load_subject_csv(file_path)
        if args.max_eval_samples is not None:
            samples = samples[:args.max_eval_samples]
        subject_csv_pairs.append((subject, file_path, samples))
        total_eval_samples += len(samples)
    print(f"  Total samples across {len(csv_files)} files: {total_eval_samples}")
    print()

    # ── Evaluate each subject ──
    all_results: List[SubjectResult] = []
    t_start = time.time()

    # Global progress bar: tracks samples across ALL subjects
    global_pbar = tqdm(
        total=total_eval_samples,
        desc="Overall   ",
        unit="sample",
        position=0,
        smoothing=0.1,
    )

    for file_idx, (subject, file_path, samples) in enumerate(subject_csv_pairs):
        current_pos = global_pbar.n  # capture current position before starting

        tqdm.write(f"[{file_idx + 1}/{len(csv_files)}] Evaluating: {subject} "
                   f"({len(samples)} samples, starting at #{current_pos + 1}/{total_eval_samples})")

        result = evaluate_subject(
            model=model,
            tokenizer=tokenizer,
            subject=subject,
            samples=samples,
            mode=args.mode,
            max_new_tokens=args.max_new_tokens,
            block_length=args.block_length,
            temperature=args.temperature,
            threshold=args.threshold,
            device=device,
            max_eval_samples=None,  # already trimmed
            verbose=args.verbose,
            pbar=global_pbar,
        )

        all_results.append(result)
        tqdm.write("")  # blank line after each subject

    global_pbar.close()
    elapsed = time.time() - t_start

    # ── Print summary ──
    print_report(all_results, elapsed)

    return 0


if __name__ == "__main__":
    sys.exit(main())
