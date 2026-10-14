"""Shared model/device loading helpers for sampler data collection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def detect_device(local_rank: int = 0) -> str:
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return f"npu:{local_rank}"
    except ImportError:
        pass
    if torch.cuda.is_available():
        return f"cuda:{local_rank}"
    return "cpu"


def load_diffusion_model(
    model_path: str | Path,
    device: str | torch.device,
    dtype: torch.dtype,
    config_diffusion_path: str | Path | None = None,
) -> Any:
    """Load the same HF-format checkpoint used by the existing example scripts."""
    from transformers import Qwen3Config

    from ..modeling_qwen3_diffusion import Qwen3DiffusionForCausalLM

    config = Qwen3Config.from_pretrained(model_path)
    if config_diffusion_path is not None:
        with Path(config_diffusion_path).open(encoding="utf-8") as stream:
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
        device_map=str(device),
    )
    model.requires_grad_(False)
    model.eval()
    return model


def load_tokenizer(model_path: str | Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
