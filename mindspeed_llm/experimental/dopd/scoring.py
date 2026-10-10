"""Teacher scoring and token-level future-aware distillation loss."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mindspeed_llm.fsdp2.models.qwen3.utils import set_diffusion_lm

from .core import (
    apply_future_reweighting,
    build_dopd_attention_mask,
    topk_forward_kl,
)


@dataclass
class LossStats:
    loss_tokens: int = 0
    eligible_positions: int = 0
    corrected_positions: int = 0
    scored_branches: int = 0
    skipped_no_visible_future: int = 0
    skipped_nonfinite: int = 0


def _project(head, hidden_states: torch.Tensor) -> torch.Tensor:
    output = head(hidden_states)
    return output[0] if isinstance(output, tuple) else output


def _cache_tensor_pairs(cache):
    if hasattr(cache, "layers"):
        pairs = []
        for layer in cache.layers:
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if key is None or value is None:
                continue
            pairs.append((key, value))
        if pairs:
            return pairs
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    try:
        return list(cache)
    except TypeError as exc:
        raise TypeError(f"unsupported teacher cache type: {type(cache).__name__}") from exc


def _branch_cache(shared_cache, prefix_length: int, batch_size: int, teacher_config):
    """Create a branch cache using expanded prefix views when supported."""

    from transformers.cache_utils import DynamicCache

    pairs = []
    for key, value in _cache_tensor_pairs(shared_cache):
        key_prefix = key[:, :, :prefix_length, :].expand(batch_size, -1, -1, -1)
        value_prefix = value[:, :, :prefix_length, :].expand(batch_size, -1, -1, -1)
        pairs.append((key_prefix, value_prefix))
    if hasattr(DynamicCache, "from_legacy_cache"):
        try:
            return DynamicCache.from_legacy_cache(tuple(pairs))
        except (TypeError, AttributeError):
            pass
    constructors = (
        lambda: DynamicCache(pairs),
        lambda: DynamicCache(ddp_cache_data=pairs, config=teacher_config),
        lambda: DynamicCache(ddp_cache_data=pairs),
    )
    for constructor in constructors:
        try:
            return constructor()
        except TypeError:
            continue

    # Compatibility fallback for cache implementations without a data ctor.
    cache = copy.deepcopy(shared_cache)
    if hasattr(cache, "crop"):
        cache.crop(prefix_length)
    if hasattr(cache, "batch_repeat_interleave"):
        cache.batch_repeat_interleave(batch_size)
        return cache
    raise TypeError("DynamicCache cannot be sliced/repeated by this Transformers version")


def _cache_bytes_per_branch(shared_cache, prefix_length: int) -> int:
    total = 0
    for key, value in _cache_tensor_pairs(shared_cache):
        total += key[:, :, :prefix_length, :].numel() * key.element_size()
        total += value[:, :, :prefix_length, :].numel() * value.element_size()
    return max(1, total)


@torch.no_grad()
def score_future_branches(
    teacher,
    shared_cache,
    clean_sequence: torch.LongTensor,
    *,
    position: int,
    future_end: int,
    branch_token_ids: torch.LongTensor,
    max_chunk_size: int,
    cache_max_bytes: int,
) -> torch.Tensor:
    """Score p_T(clean future | causal prefix, branch token) for each branch."""

    future = clean_sequence[position + 1 : future_end]
    if future.numel() == 0:
        return torch.zeros(branch_token_ids.numel(), device=clean_sequence.device)
    suffix_tail = future[:-1]
    suffix_length = 1 + suffix_tail.numel()
    per_branch = _cache_bytes_per_branch(shared_cache, position)
    memory_limited = max(1, int(cache_max_bytes) // per_branch)
    chunk_size = max(1, min(int(max_chunk_size), memory_limited))
    scores = []
    for start in range(0, branch_token_ids.numel(), chunk_size):
        branch_tokens = branch_token_ids[start : start + chunk_size]
        branch_count = branch_tokens.numel()
        branch_cache = _branch_cache(
            shared_cache, prefix_length=position, batch_size=branch_count,
            teacher_config=teacher.config,
        )
        suffix = torch.empty(
            (branch_count, suffix_length),
            dtype=clean_sequence.dtype,
            device=clean_sequence.device,
        )
        suffix[:, 0] = branch_tokens
        if suffix_tail.numel():
            suffix[:, 1:] = suffix_tail.unsqueeze(0)
        cache_positions = torch.arange(
            position, position + suffix_length, device=clean_sequence.device
        )
        outputs = teacher.model(
            input_ids=suffix,
            attention_mask=torch.ones(
                (branch_count, position + suffix_length),
                dtype=torch.long,
                device=clean_sequence.device,
            ),
            position_ids=cache_positions.unsqueeze(0).expand(branch_count, -1),
            cache_position=cache_positions,
            past_key_values=branch_cache,
            use_cache=True,
        )
        logits = _project(teacher.lm_head, outputs.last_hidden_state).float()
        log_probs = F.log_softmax(logits, dim=-1)
        targets = future.unsqueeze(0).expand(branch_count, -1)
        scores.append(log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum(-1))
        del branch_cache, outputs, logits, log_probs
    return torch.cat(scores, dim=0)


def compute_dopd_loss(
    student,
    teacher,
    prompt_ids: torch.LongTensor,
    response_ids: torch.LongTensor,
    noised_response: torch.LongTensor,
    loss_mask: torch.BoolTensor,
    *,
    block_size: int,
    candidate_top_k: int,
    loss_top_k: int,
    correction_strength: float,
    require_visible_future: bool,
    correction_allowed: bool,
    score_chunk_size: int,
    cache_max_bytes: int,
    future_valid_length: int | None = None,
) -> tuple[torch.Tensor, LossStats]:
    """Compute forward KL for one rank-local trajectory (micro-batch size 1)."""

    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("compute_dopd_loss currently requires one prompt per rank")
    if response_ids.shape != noised_response.shape or response_ids.shape != loss_mask.shape:
        raise ValueError("response, noised response, and loss mask shapes must match")
    if response_ids.ndim != 2 or response_ids.shape[0] != 1:
        raise ValueError("compute_dopd_loss currently requires one response per rank")

    stats = LossStats(loss_tokens=int(loss_mask.sum().item()))
    prompt_length = prompt_ids.shape[1]
    response_length = response_ids.shape[1]
    if future_valid_length is None:
        future_valid_length = response_length
    future_valid_length = max(0, min(int(future_valid_length), response_length))
    student_input = torch.cat([prompt_ids, response_ids, noised_response], dim=1)
    model_dtype = next(student.parameters()).dtype
    attention_mask = build_dopd_attention_mask(
        prompt_length,
        response_length,
        block_size,
        dtype=model_dtype,
        device=student_input.device,
    )
    clean_positions = torch.arange(
        prompt_length + response_length, device=student_input.device
    )
    response_positions = torch.arange(
        prompt_length, prompt_length + response_length, device=student_input.device
    )
    position_ids = torch.cat([clean_positions, response_positions]).unsqueeze(0)

    # The existing Qwen3 diffusion attention has a special equal-halves training
    # branch for Nemotron CPT. d-OPD uses [prompt, clean response, sampled state]
    # with a different mask, so eval mode is deliberate; gradients remain
    # enabled and Qwen3 has no training-only dropout.
    student.eval()
    set_diffusion_lm(student, True)
    selected_positions = loss_mask[0].nonzero(as_tuple=True)[0]
    noised_start = prompt_length + response_length
    projection_positions = noised_start + selected_positions
    if projection_positions.numel() == 0:
        projection_positions = torch.tensor([noised_start], device=student_input.device)

    # Use the Qwen3Diffusion root so composable FSDP2 root hooks run, but project
    # only KL positions through the 152k-token LM head. This preserves the full
    # transformer computation and removes multi-gigabyte unused logits.
    def select_hidden(_module, inputs):
        hidden_states = inputs[0].index_select(1, projection_positions)
        return (hidden_states, *inputs[1:])

    handle = student.lm_head.register_forward_pre_hook(select_hidden)
    try:
        student_outputs = student(
            input_ids=student_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
    finally:
        handle.remove()
    if selected_positions.numel() == 0:
        dummy = student_outputs.logits
        return dummy.float().sum() * 0, stats
    student_logits = student_outputs.logits[0]
    del student_outputs
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)

    clean_sequence = torch.cat([prompt_ids[0], response_ids[0]], dim=0)
    with torch.no_grad():
        teacher_positions = torch.arange(clean_sequence.numel(), device=clean_sequence.device)
        teacher_outputs = teacher.model(
            input_ids=clean_sequence.unsqueeze(0),
            attention_mask=torch.ones_like(clean_sequence).unsqueeze(0),
            position_ids=teacher_positions.unsqueeze(0),
            cache_position=teacher_positions,
            use_cache=True,
        )
        shared_cache = teacher_outputs.past_key_values
        predictor_positions = prompt_length + selected_positions - 1
        teacher_hidden = teacher_outputs.last_hidden_state[0, predictor_positions]
        teacher_logits = _project(teacher.lm_head, teacher_hidden)
        base_teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
        del teacher_outputs, teacher_hidden, teacher_logits

    losses = []
    for selected_index, relative_position_tensor in enumerate(selected_positions):
        relative_position = int(relative_position_tensor.item())
        absolute_position = prompt_length + relative_position
        teacher_target = base_teacher_log_probs[selected_index]
        block_start = (relative_position // block_size) * block_size
        block_end = min(block_start + block_size, future_valid_length)
        visible_future = noised_response[0, relative_position + 1 : block_end].ne(
            int(student.mask_token_id)
        )
        if correction_allowed and correction_strength > 0:
            if require_visible_future and not visible_future.any().item():
                stats.skipped_no_visible_future += 1
            elif relative_position + 1 < block_end:
                stats.eligible_positions += 1
                k = min(candidate_top_k, teacher_target.numel())
                teacher_candidates = torch.topk(teacher_target, k=k).indices
                student_candidates = torch.topk(
                    student_log_probs[selected_index].detach(), k=k
                ).indices
                candidates = torch.unique(torch.cat([teacher_candidates, student_candidates]))
                clean_token = clean_sequence[absolute_position]
                candidates = candidates[candidates.ne(clean_token)]
                branches = torch.cat([clean_token.reshape(1), candidates])
                scores = score_future_branches(
                    teacher,
                    shared_cache,
                    clean_sequence,
                    position=absolute_position,
                    future_end=prompt_length + block_end,
                    branch_token_ids=branches,
                    max_chunk_size=score_chunk_size,
                    cache_max_bytes=cache_max_bytes,
                )
                stats.scored_branches += int(branches.numel())
                deltas = scores[1:] - scores[0]
                if torch.isfinite(deltas).any().item():
                    corrected = apply_future_reweighting(
                        teacher_target,
                        candidates,
                        deltas,
                        correction_strength,
                    )
                    if not torch.equal(corrected, teacher_target):
                        teacher_target = corrected
                        stats.corrected_positions += 1
                else:
                    stats.skipped_nonfinite += 1
        losses.append(
            topk_forward_kl(
                teacher_target.unsqueeze(0),
                student_log_probs[selected_index].unsqueeze(0),
                loss_top_k,
            ).squeeze(0)
        )

    del shared_cache, base_teacher_log_probs
    return torch.stack(losses).mean(), stats

