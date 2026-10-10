"""Ascend/FSDP2 reproduction utilities for future-aware on-policy distillation."""

from .config import DOPDArguments
from .core import (
    apply_future_reweighting,
    build_dopd_attention_mask,
    cosine_response_length,
    sample_intermediate_state,
    topk_forward_kl,
)

__all__ = [
    "DOPDArguments",
    "apply_future_reweighting",
    "build_dopd_attention_mask",
    "cosine_response_length",
    "sample_intermediate_state",
    "topk_forward_kl",
]

