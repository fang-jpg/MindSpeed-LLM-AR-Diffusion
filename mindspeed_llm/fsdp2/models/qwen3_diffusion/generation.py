# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Block-wise diffusion generation for Qwen3DiffusionForCausalLM."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from ..qwen3.utils import set_diffusion_lm


@dataclass
class GenerationStats:
    """Forward-count statistics for one block-diffusion generation call."""

    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_nfe: int = 0
    denoise_nfe: int = 0
    refresh_nfe: int = 0
    committed_per_step: list[int] = field(default_factory=list)

    @property
    def total_nfe(self) -> int:
        return self.prefill_nfe + self.denoise_nfe + self.refresh_nfe

    @property
    def decode_nfe(self) -> int:
        return self.denoise_nfe + self.refresh_nfe

    @property
    def real_tpf(self) -> float:
        if self.total_nfe == 0:
            return 0.0
        return self.generated_tokens / self.total_nfe

    def to_dict(self) -> dict:
        result = asdict(self)
        result.update(
            total_nfe=self.total_nfe,
            decode_nfe=self.decode_nfe,
            real_tpf=self.real_tpf,
        )
        return result


def _crop_dynamic_cache(past_key_values: DynamicCache, max_length: int) -> None:
    """Restore a cache after a temporary denoising forward."""
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(max_length)
        return
    for layer_index in range(len(past_key_values)):
        past_key_values.key_cache[layer_index] = past_key_values.key_cache[
            layer_index
        ][:, :, :max_length]
        past_key_values.value_cache[layer_index] = past_key_values.value_cache[
            layer_index
        ][:, :, :max_length]
    past_key_values._seen_tokens = max_length


def _get_num_transfer_tokens(mask_index: torch.BoolTensor, steps: int) -> torch.LongTensor:
    """Evenly split masked positions across a fixed denoising budget."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.long
    ) + base
    for batch_index in range(mask_num.size(0)):
        num_transfer_tokens[batch_index, : int(remainder[batch_index].item())] += 1
    return num_transfer_tokens


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply the Gumbel-max transformation used by the reference sampler."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    mask_index: torch.BoolTensor,
    x: torch.LongTensor,
    num_transfer_tokens: torch.LongTensor,
    threshold: Optional[float] = None,
    invalid_token_id: Optional[int] = None,
):
    """Select the highest-confidence masked positions to commit."""
    logits = logits.clone()
    if invalid_token_id is not None and 0 <= invalid_token_id < logits.shape[-1]:
        logits[..., invalid_token_id] = -torch.inf

    logits_with_noise = _add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    probabilities = F.softmax(logits.float(), dim=-1)
    x0_probability = torch.gather(probabilities, -1, x0.unsqueeze(-1)).squeeze(-1)

    x0 = torch.where(mask_index, x0, x)
    confidence = x0_probability.masked_fill(~mask_index, -torch.inf)
    transfer_index = torch.zeros_like(x0, dtype=torch.bool)

    for batch_index in range(confidence.shape[0]):
        requested = int(num_transfer_tokens[batch_index].item())
        available = int(mask_index[batch_index].sum().item())
        count = min(requested, available)
        if count <= 0:
            continue
        selected = torch.topk(confidence[batch_index], k=count).indices
        if threshold is not None:
            above_threshold = confidence[batch_index, selected] >= threshold
            # Fixed-budget experiments do not use threshold. For compatibility,
            # threshold mode still guarantees progress by keeping the best token.
            above_threshold[0] = True
            selected = selected[above_threshold]
        transfer_index[batch_index, selected] = True
    return x0, transfer_index


def _filter_logits(logits: torch.Tensor, topk: int, top_p: float) -> torch.Tensor:
    if topk > 0:
        k = min(topk, logits.shape[-1])
        cutoff = torch.topk(logits, k=k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < cutoff, -torch.inf)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        logits = torch.full_like(logits, -torch.inf).scatter(-1, sorted_indices, sorted_logits)
    return logits


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    topk: int,
    top_p: float,
) -> torch.LongTensor:
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    filtered = _filter_logits(logits.float() / temperature, topk=topk, top_p=top_p)
    return torch.multinomial(torch.softmax(filtered, dim=-1), num_samples=1)


def _finish_result(
    output_ids: torch.LongTensor,
    prompt_length: int,
    stats: GenerationStats,
    return_stats: bool,
):
    stats.generated_tokens = output_ids.shape[1] - prompt_length
    return output_ids, stats if return_stats else stats.total_nfe


@torch.no_grad()
def block_diff_generate(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    block_length: int,
    steps_per_block: Optional[int] = None,
    threshold: Optional[float] = None,
    causal_context: bool = True,
    temperature: float = 0.0,
    topk: int = 0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    tokenizer=None,
    print_block_predictions: bool = False,
    seed_mode: str = "ar",
    ignore_eos: bool = False,
    return_stats: bool = False,
    debug: bool = False,
):
    """Generate left-to-right blocks with bidirectional denoising inside a block.

    ``block_length`` and ``steps_per_block`` are independent. The final block
    may be shorter than ``block_length`` and the returned sequence never exceeds
    ``max_new_tokens``. ``seed_mode='none'`` creates an all-mask block, while
    ``seed_mode='ar'`` preserves the repository's historical hybrid AR seed.
    """
    if prompt_ids.ndim != 2:
        raise ValueError("prompt_ids must have shape [batch, sequence]")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    if steps_per_block is None:
        steps_per_block = block_length
    if steps_per_block <= 0:
        raise ValueError("steps_per_block must be positive")
    if seed_mode not in {"none", "ar"}:
        raise ValueError("seed_mode must be either 'none' or 'ar'")
    if seed_mode == "ar" and not causal_context:
        raise ValueError("seed_mode='ar' requires causal_context=True")

    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)
    mask_token_id = int(model.mask_token_id)
    output_ids = prompt_ids.clone()
    batch_size, prompt_length = prompt_ids.shape
    device = prompt_ids.device
    stats = GenerationStats(prompt_tokens=prompt_length)

    if causal_context:
        set_diffusion_lm(model, False)

    past_key_values = DynamicCache()
    cache_position = torch.arange(prompt_length, device=device)
    position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
    prefill = model.model(
        input_ids=prompt_ids,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        cache_position=cache_position,
    )
    past_key_values = prefill.past_key_values
    stats.prefill_nfe = 1

    next_token = None
    if seed_mode == "ar":
        next_logits, _ = model.lm_head(prefill.last_hidden_state[:, -1:, :])
        next_token = _sample_next_token(
            next_logits.squeeze(1), temperature=temperature, topk=topk, top_p=top_p
        )
    if causal_context:
        set_diffusion_lm(model, True)

    generated = 0
    num_blocks = math.ceil(max_new_tokens / block_length)
    for block_index in range(num_blocks):
        current_length = min(block_length, max_new_tokens - generated)
        block_start = prompt_length + generated
        block = torch.full(
            (batch_size, current_length),
            mask_token_id,
            dtype=prompt_ids.dtype,
            device=device,
        )
        if seed_mode == "ar" and next_token is not None:
            block[:, 0] = next_token[:, 0]
        block_seed = block[:, :1].clone()
        output_ids = torch.cat([output_ids, block], dim=1)
        block_slice = slice(block_start, block_start + current_length)

        initial_mask = output_ids[:, block_slice].eq(mask_token_id)
        max_masks = int(initial_mask.sum(dim=1).max().item())
        block_steps = min(steps_per_block, max_masks) if max_masks > 0 else 0
        transfer_schedule = (
            _get_num_transfer_tokens(initial_mask, block_steps)
            if block_steps > 0
            else None
        )

        for step_index in range(block_steps):
            current_mask = output_ids[:, block_slice].eq(mask_token_id)
            if not current_mask.any():
                break
            prefix_cache_length = past_key_values.get_seq_length()
            denoise_position_ids = torch.arange(
                block_start, block_start + current_length, device=device
            ).unsqueeze(0).expand(batch_size, -1)
            denoise = model.model(
                input_ids=output_ids[:, block_slice],
                past_key_values=past_key_values,
                use_cache=True,
                position_ids=denoise_position_ids,
                cache_position=torch.arange(
                    prefix_cache_length,
                    prefix_cache_length + current_length,
                    device=device,
                ),
            )
            past_key_values = denoise.past_key_values
            _crop_dynamic_cache(past_key_values, prefix_cache_length)
            stats.denoise_nfe += 1
            block_logits, _ = model.lm_head(denoise.last_hidden_state)
            proposed, transfer = _get_transfer_index(
                block_logits,
                temperature,
                current_mask,
                output_ids[:, block_slice],
                num_transfer_tokens=transfer_schedule[:, step_index],
                threshold=threshold,
                invalid_token_id=mask_token_id,
            )
            committed = int(transfer.sum().item())
            stats.committed_per_step.append(committed)
            current_block = output_ids[:, block_slice].clone()
            current_block[transfer] = proposed[transfer]
            output_ids[:, block_slice] = current_block
            if debug:
                print(
                    f"block={block_index} step={step_index} "
                    f"committed={committed} remaining={int(current_mask.sum().item()) - committed}"
                )

        unresolved = output_ids[:, block_slice].eq(mask_token_id)
        if unresolved.any():
            raise RuntimeError(
                f"Block {block_index} still contains {int(unresolved.sum().item())} "
                f"mask tokens after {block_steps} denoising steps."
            )

        generated += current_length
        if print_block_predictions:
            for batch_index in range(batch_size):
                completed_ids = output_ids[batch_index, block_slice].tolist()
                print(f"\n[Diffusion block {block_index} | batch {batch_index}]")
                print(f"  positions: [{block_start}, {block_start + current_length})")
                print(f"  seed ids: {block_seed[batch_index].tolist()}")
                print(f"  block ids: {completed_ids}")
                if tokenizer is not None:
                    print(
                        "  completed block:",
                        repr(tokenizer.decode(completed_ids, skip_special_tokens=False)),
                    )

        if not ignore_eos and eos_token_id is not None:
            generated_region = output_ids[:, prompt_length:]
            eos_mask = generated_region.eq(eos_token_id)
            if eos_mask.any(dim=1).all():
                first_eos = eos_mask.to(torch.int64).argmax(dim=1)
                common_end = int(first_eos.max().item()) + 1
                return _finish_result(
                    output_ids[:, : prompt_length + common_end],
                    prompt_length,
                    stats,
                    return_stats,
                )

        # The final block does not need to be inserted into KV cache because
        # no later block consumes it. Avoiding this unused forward is part of
        # the measured decoder implementation, not a change to model output.
        if block_index + 1 == num_blocks:
            break

        if causal_context:
            set_diffusion_lm(model, False)
        cache_length = past_key_values.get_seq_length()
        refresh_cache_position = torch.arange(
            cache_length, cache_length + current_length, device=device
        )
        refresh_position_ids = refresh_cache_position.unsqueeze(0).expand(batch_size, -1)
        refresh = model.model(
            input_ids=output_ids[:, block_slice],
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=refresh_cache_position,
            position_ids=refresh_position_ids,
        )
        past_key_values = refresh.past_key_values
        stats.refresh_nfe += 1

        if seed_mode == "ar" and block_index + 1 < num_blocks:
            next_logits, _ = model.lm_head(refresh.last_hidden_state[:, -1:, :])
            next_token = _sample_next_token(
                next_logits.squeeze(1),
                temperature=temperature,
                topk=topk,
                top_p=top_p,
            )
        if causal_context:
            set_diffusion_lm(model, True)

    return _finish_result(output_ids, prompt_length, stats, return_stats)
