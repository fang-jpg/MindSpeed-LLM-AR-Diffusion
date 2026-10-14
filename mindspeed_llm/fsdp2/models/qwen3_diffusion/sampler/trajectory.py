"""Serial denoising trajectories and eventual-token labels for sampler training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache

from ..generation import _crop_dynamic_cache
from ...qwen3.utils import set_diffusion_lm
from .data import SamplerTrainingRecord
from .features import SamplerFeatureExtractor


@dataclass
class _PendingState:
    features: torch.Tensor
    top1_ids: torch.LongTensor
    active_mask: torch.BoolTensor


def _head_logits(model, hidden_states: torch.Tensor) -> torch.Tensor:
    output = model.lm_head(hidden_states)
    return output[0] if isinstance(output, tuple) else output


@torch.no_grad()
def collect_block_trajectory(
    model,
    prefix_ids: torch.LongTensor,
    block_length: int,
    feature_extractor: SamplerFeatureExtractor,
    policy: str,
    sample_index: int,
    block_index: int,
    target_ids: torch.LongTensor | None = None,
) -> tuple[list[SamplerTrainingRecord], torch.LongTensor]:
    """Collect every intermediate state from one serially denoised block.

    ``confidence`` commits exactly one globally most-confident masked position
    per step.  ``hybrid`` first commits every current top-1 that agrees with the
    ground-truth block; when none agree it falls back to the single most-
    confident position.  Labels are assigned after completion by comparing each
    saved current top-1 with the final token committed by the same policy.
    """
    if policy not in {"confidence", "hybrid"}:
        raise ValueError("policy must be 'confidence' or 'hybrid'")
    if prefix_ids.ndim != 2 or prefix_ids.shape[0] != 1:
        raise ValueError("prefix_ids must have shape [1, sequence]")
    if prefix_ids.shape[1] == 0:
        raise ValueError("prefix_ids must contain at least one token")
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    if policy == "hybrid":
        if target_ids is None or target_ids.shape != (1, block_length):
            raise ValueError("hybrid policy requires target_ids [1, block_length]")

    device = prefix_ids.device
    mask_token_id = int(model.mask_token_id)
    prefix_length = prefix_ids.shape[1]
    past_key_values = DynamicCache()
    positions = torch.arange(prefix_length, device=device)

    set_diffusion_lm(model, False)
    prefill = model.model(
        input_ids=prefix_ids,
        position_ids=positions.unsqueeze(0),
        past_key_values=past_key_values,
        use_cache=True,
        cache_position=positions,
    )
    past_key_values = prefill.past_key_values
    set_diffusion_lm(model, True)

    block = torch.full(
        (1, block_length),
        mask_token_id,
        dtype=prefix_ids.dtype,
        device=device,
    )
    pending: list[_PendingState] = []
    try:
        for _ in range(block_length):
            active_mask = block.eq(mask_token_id)
            if not active_mask.any():
                break
            cache_length = past_key_values.get_seq_length()
            block_positions = torch.arange(
                prefix_length,
                prefix_length + block_length,
                device=device,
            )
            denoise = model.model(
                input_ids=block,
                past_key_values=past_key_values,
                use_cache=True,
                position_ids=block_positions.unsqueeze(0),
                cache_position=torch.arange(
                    cache_length,
                    cache_length + block_length,
                    device=device,
                ),
            )
            past_key_values = denoise.past_key_values
            _crop_dynamic_cache(past_key_values, cache_length)
            logits = _head_logits(model, denoise.last_hidden_state)
            feature_batch = feature_extractor(logits)
            pending.append(
                _PendingState(
                    features=feature_batch.features[0].cpu().to(torch.float16),
                    top1_ids=feature_batch.top1_ids[0].cpu(),
                    active_mask=active_mask[0].cpu(),
                )
            )

            proposed = feature_batch.top1_ids
            commit = torch.zeros_like(active_mask)
            if policy == "hybrid":
                commit = active_mask & proposed.eq(target_ids)
            if not commit.any():
                confidence = feature_batch.top1_probability.masked_fill(
                    ~active_mask, -torch.inf
                )
                best_position = confidence[0].argmax()
                commit[0, best_position] = True
            block[commit] = proposed[commit]
    finally:
        set_diffusion_lm(model, False)

    if block.eq(mask_token_id).any():
        raise RuntimeError("serial trajectory failed to complete the block")
    final_ids = block[0].detach().cpu()
    records = []
    for step_index, state in enumerate(pending):
        labels = state.top1_ids.eq(final_ids) & state.active_mask
        records.append(
            SamplerTrainingRecord(
                features=state.features,
                labels=labels,
                active_mask=state.active_mask,
                policy=policy,
                sample_index=sample_index,
                block_index=block_index,
                step_index=step_index,
            )
        )
    return records, final_ids
