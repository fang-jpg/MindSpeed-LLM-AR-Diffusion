#!/usr/bin/env python3
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
# Phase 5: Three-mode inference script for Qwen3DiffusionForCausalLM.
#
# Modes:
#   ar          — Standard causal autoregressive generation
#   diffusion   — Block-wise diffusion denoising generation
#   linear_spec — Linear speculative decoding (diffusion draft + AR verify)
#
# Usage:
#   # AR mode
#   python inference_qwen3_diffusion.py \
#     --model-path /path/to/Qwen3-1.7B-Base/ \
#     --mode ar \
#     --prompt "Explain quantum computing in simple terms." \
#     --max-new-tokens 256
#
#   # Diffusion mode
#   python inference_qwen3_diffusion.py \
#     --model-path /path/to/Qwen3-1.7B-Base/ \
#     --mode diffusion \
#     --prompt "Explain quantum computing in simple terms." \
#     --max-new-tokens 256 \
#     --block-length 32
#
#   # Linear-Spec mode
#   python inference_qwen3_diffusion.py \
#     --model-path /path/to/Qwen3-1.7B-Base/ \
#     --mode linear_spec \
#     --prompt "Explain quantum computing in simple terms." \
#     --max-new-tokens 256 \
#     --block-length 32 \
#     --threshold 0.0

import argparse
import sys
import time

import torch
from transformers import AutoTokenizer, Qwen3Config

from mindspeed_llm.fsdp2.models.qwen3_diffusion.modeling_qwen3_diffusion import (
    Qwen3DiffusionForCausalLM,
)
from mindspeed_llm.fsdp2.models.qwen3_diffusion.generation import block_diff_generate
from mindspeed_llm.fsdp2.models.qwen3_diffusion.linear_spec_generate import (
    ar_generate,
    linear_spec_generate,
)
def parse_args():
    p = argparse.ArgumentParser(
        description="Three-mode inference for Qwen3-1.7B-Diffusion"
    )
    p.add_argument(
        "--model-path", type=str, required=True,
        help="Path to Qwen3-1.7B-Base model directory (must contain config.json with diffusion fields, or use --config-diffusion)"
    )
    p.add_argument(
        "--mode", type=str, required=True, choices=["ar", "diffusion", "linear_spec"],
        help="Inference mode: ar (causal), diffusion (block-wise denoising), linear_spec (draft + verify)"
    )
    p.add_argument(
        "--prompt", type=str, default=None,
        help="Input prompt text. If not provided, uses a default example."
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=128,
        help="Maximum number of new tokens to generate"
    )
    p.add_argument(
        "--block-length", type=int, default=32,
        help="Block length for diffusion / linear_spec modes (must divide max-new-tokens for diffusion mode)"
    )
    p.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (0 = greedy argmax)"
    )
    p.add_argument(
        "--threshold", type=float, default=0.0,
        help="Confidence threshold for diffusion denoising (0 = one-shot, >0 = progressive)"
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="Device (auto-detected if not set). Examples: cpu, npu:0, cuda:0"
    )
    p.add_argument(
        "--dtype", type=str, default="bfloat16",
        help="Model dtype: bfloat16, float16, float32"
    )
    p.add_argument(
        "--config-diffusion", type=str, default=None,
        help="Path to config_diffusion.json to inject diffusion fields into model config. "
             "If None, diffusion fields are set programmatically with defaults."
    )
    p.add_argument(
        "--top-k", type=str, default=0,
        help="Path to config_diffusion.json to inject diffusion fields into model config. "
             "If None, diffusion fields are set programmatically with defaults."
    )
    p.add_argument(
        "--top-p", type=str, default=0.0,
        help="Path to config_diffusion.json to inject diffusion fields into model config. "
             "If None, diffusion fields are set programmatically with defaults."
    )
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


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_model(model_path, device, dtype, config_diffusion_path=None):
    """Load Qwen3DiffusionForCausalLM with diffusion config."""
    config = Qwen3Config.from_pretrained(model_path)
    config.tie_word_embeddings = False
    # Inject diffusion config fields
    if config_diffusion_path is not None:
        import json
        with open(config_diffusion_path, "r") as f:
            diff_cfg = json.load(f)
        print(f"change before:{config}")
        for k, v in diff_cfg.items():
            setattr(config, k, v)
        print(f"change after:{config}")
    else:
        # Default diffusion config
        config.mask_token_id = 151669  # one past current vocab
        config.dlm_paradigm = "block_diff"
        config.block_size = 8
        config.dlm_loss_weight = 0.3
        config.ar_loss_weight = 1.0
        config.dp_varying_mask_ratio = True
        config.enable_diffusion_lm = True
        config.diffusion_lm = True

    # Use eager attention for compatibility (flash-attn may not support dynamic masks)
    config._attn_implementation = "eager"

    model = Qwen3DiffusionForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        device_map=device,
    )

    # print(f"test model struct:{model}")
    # print(model.diffusion_head.state_dict())

    model.eval()
    return model


def main():
    args = parse_args()
    device = args.device or detect_device()
    dtype = DTYPE_MAP[args.dtype]

    print("=" * 60)
    print("Qwen3-1.7B Diffusion — Three-Mode Inference")
    print("=" * 60)
    print(f"  Model:       {args.model_path}")
    print(f"  Mode:        {args.mode}")
    print(f"  Device:      {device}")
    print(f"  Dtype:       {args.dtype}")
    print(f"  Max tokens:  {args.max_new_tokens}")
    print(f"  Block size:  {args.block_length}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Threshold:   {args.threshold}")
    print(f"  top_k: {args.top_k}")
    print(f"  top_p:   {args.top_p}")

    # Load model
    print("\n> Loading model...")
    t0 = time.time()
    model = load_model(args.model_path, device, dtype, args.config_diffusion)
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    print(f"  Num layers:  {len(model.model.layers)}")
    print(f"  Vocab size:  {model.config.vocab_size}")
    print(f"  Mask token:  {model.mask_token_id}")

    # Load tokenizer
    print("\n> Loading tokenizer...")
    print(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    # tokenizer2 = AutoTokenizer.from_pretrained(
    #     "/home/c00633840/models/Qwen3-1.7B-Base", trust_remote_code=True
    # )
    print("pad_token:", tokenizer.pad_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("eos_token:", tokenizer.eos_token)
    print("eos_token_id:", tokenizer.eos_token_id)
    # Prepare prompt
    prompt = args.prompt
    if prompt is None:
        prompt = "The meaning of life is"
    print(f"\n> Prompt: {prompt!r}")

    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    print(f"  Input length: {input_ids.shape[1]} tokens")
    print(f"  Input input_ids: {input_ids}")

    # Generate
    print(f"\n> Running {args.mode} generation...")
    t_start = time.time()

    if args.mode == "ar":
        output_ids, nfe = ar_generate(
            model=model,
            prompt_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            topk = args.top_k,
            eos_token_id=tokenizer.eos_token_id,
        )
        extra_info = ""
    elif args.mode == "diffusion":
        # max_new_tokens must be divisible by block_length for diffusion mode
        max_new = args.max_new_tokens
        if max_new % args.block_length != 0:
            max_new = max_new + (args.block_length - max_new % args.block_length)
            print(f"  [INFO] Adjusted max_new_tokens from {args.max_new_tokens} to {max_new} (divisible by block_length={args.block_length})")
        output_ids, nfe = block_diff_generate(
            model=model,
            prompt_ids=input_ids,
            max_new_tokens=max_new,
            block_length=args.block_length,
            threshold=args.threshold if args.threshold > 0 else None,
            causal_context=True,
            temperature=args.temperature,
            topk = args.top_k,
            top_p = args.top_p,
            eos_token_id=tokenizer.eos_token_id,
            tokenizer=tokenizer,
            print_block_predictions=True
        )
        extra_info = ""
    elif args.mode == "linear_spec":
        output_ids, nfe, avg_accept = linear_spec_generate(
            model=model,
            prompt_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            block_length=args.block_length,
            temperature=args.temperature,
            eos_token_id=tokenizer.eos_token_id,
            threshold=args.threshold,
        )
        extra_info = f"  Avg accept length: {avg_accept:.2f}\n"
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    elapsed = time.time() - t_start

    # Decode
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    print(len(output_text))

    new_tokens = output_ids.shape[1] - input_ids.shape[1]
    tpf = new_tokens / nfe if nfe > 0 else 0  # tokens per forward

    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Mode:           {args.mode}")
    print(f"  Prompt tokens:  {input_ids.shape[1]}")
    print(f"  Generated:      {new_tokens} tokens")
    print(f"  NFE:            {nfe}")
    print(f"  TPF:            {tpf:.2f} tokens/forward")
    print(f"  Wall time:      {elapsed:.2f}s")
    print(f"  Throughput:     {new_tokens / elapsed:.1f} tokens/s")
    if extra_info:
        print(extra_info)
    print(f"\n  Generated text:")
    print(f"  {'─' * 50}")
    # Print indented, wrapped output
    text = output_text[len(prompt):] if output_text.startswith(prompt) else output_text
    for line in text.split("\n"):
        print(f"  {line}")
    print(f"  {'─' * 50}")

    return 0


if __name__ == "__main__":
    sys.exit(main())