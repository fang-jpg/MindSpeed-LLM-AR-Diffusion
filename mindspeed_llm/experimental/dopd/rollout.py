"""On-policy block-diffusion rollout with first-reveal trajectory capture."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from mindspeed_llm.fsdp2.models.qwen3.utils import set_diffusion_lm


def _root_forward(model, *, lm_positions: torch.LongTensor | None = None, **kwargs):
    """Run the FSDP root while projecting only requested hidden positions."""

    if lm_positions is None:
        return model(**kwargs)

    def select_hidden(_module, inputs):
        hidden_states = inputs[0].index_select(1, lm_positions)
        return (hidden_states, *inputs[1:])

    handle = model.lm_head.register_forward_pre_hook(select_hidden)
    try:
        return model(**kwargs)
    finally:
        handle.remove()


def _crop_cache(cache, length: int) -> None:
    if hasattr(cache, "crop"):
        cache.crop(length)
        return
    for layer_index in range(len(cache.key_cache)):
        cache.key_cache[layer_index] = cache.key_cache[layer_index][:, :, :length]
        cache.value_cache[layer_index] = cache.value_cache[layer_index][:, :, :length]
    cache._seen_tokens = length


def _filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k > 0:
        k = min(top_k, logits.shape[-1])
        threshold = torch.topk(logits, k=k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, -torch.inf)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        logits = torch.full_like(logits, -torch.inf).scatter(-1, sorted_indices, sorted_logits)
    return logits


def _sample(logits, temperature, top_k, top_p, mask_token_id, generator):
    logits = torch.nan_to_num(logits.float(), nan=-1e4, posinf=1e4, neginf=-1e4)
    if 0 <= mask_token_id < logits.shape[-1]:
        logits[..., mask_token_id] = -torch.inf
    raw_probabilities = F.softmax(logits, dim=-1)
    if temperature == 0 or top_k == 1:
        proposed = logits.argmax(dim=-1)
    else:
        sample_logits = _filter_logits(logits / temperature, top_k=top_k, top_p=top_p)
        probabilities = F.softmax(sample_logits, dim=-1)
        proposed = torch.multinomial(
            probabilities.reshape(-1, probabilities.shape[-1]),
            num_samples=1,
            generator=generator,
        ).reshape(probabilities.shape[:-1])
    confidence = raw_probabilities.gather(-1, proposed.unsqueeze(-1)).squeeze(-1)
    return proposed, confidence


def _transfer_schedule(block_length: int, steps: int) -> list[int]:
    base, remainder = divmod(block_length, steps)
    return [base + (1 if index < remainder else 0) for index in range(steps)]


@torch.no_grad()
def block_diffusion_rollout(
    model,
    prompt_ids: torch.LongTensor,
    *,
    response_tokens: int,
    block_size: int,
    denoise_steps: int,
    mask_token_id: int,
    temperature: float,
    top_k: int,
    top_p: float,
    remasking_strategy: str,
    dynamic_threshold: float,
    seed: int,
    eos_token_id: int | None = None,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Generate responses and record each token's first reveal step.

    Ranks stop only when every trajectory across the FSDP group has emitted EOS,
    so all ranks execute the same number of student forwards. The configured
    response length remains a hard curriculum cap.
    """

    from transformers.cache_utils import DynamicCache

    if prompt_ids.ndim != 2 or prompt_ids.shape[0] < 1:
        raise ValueError("prompt_ids must have shape [local_rollout_batch, prompt_length]")
    if response_tokens <= 0:
        raise ValueError("response_tokens must be positive")
    device = prompt_ids.device
    generator = torch.Generator(device=device).manual_seed(int(seed))
    batch_size, prompt_length = prompt_ids.shape
    was_training = model.training
    model.eval()

    set_diffusion_lm(model, False)
    cache = DynamicCache()
    prompt_positions = torch.arange(prompt_length, device=device)
    # Call the Qwen3Diffusion root rather than its backbone directly.  The root
    # owns the composable FSDP2 hooks for parameters not covered by child plans.
    prefill = _root_forward(
        model,
        lm_positions=torch.tensor([prompt_length - 1], device=device),
        input_ids=prompt_ids,
        position_ids=prompt_positions.unsqueeze(0),
        cache_position=prompt_positions,
        past_key_values=cache,
        use_cache=True,
    )
    cache = prefill.past_key_values
    del prefill
    set_diffusion_lm(model, True)

    response_blocks = []
    reveal_blocks = []
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    generated = 0
    global_denoise_step = 0
    num_blocks = math.ceil(response_tokens / block_size)
    for block_index in range(num_blocks):
        current_length = min(block_size, response_tokens - generated)
        block = torch.full(
            (batch_size, current_length),
            mask_token_id,
            dtype=prompt_ids.dtype,
            device=device,
        )
        first_reveal = torch.zeros_like(block)
        steps = min(denoise_steps, current_length)
        schedule = _transfer_schedule(current_length, steps)
        block_start = prompt_length + generated

        for step_index in range(steps):
            masked = block.eq(mask_token_id)
            if not masked.any().item():
                break
            prefix_cache_length = cache.get_seq_length()
            positions = torch.arange(
                block_start, block_start + current_length, device=device
            )
            denoise = model(
                input_ids=block,
                position_ids=positions.unsqueeze(0),
                cache_position=torch.arange(
                    prefix_cache_length,
                    prefix_cache_length + current_length,
                    device=device,
                ),
                past_key_values=cache,
                use_cache=True,
            )
            cache = denoise.past_key_values
            _crop_cache(cache, prefix_cache_length)
            logits = denoise.logits
            del denoise
            proposed, confidence = _sample(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                mask_token_id=mask_token_id,
                generator=generator,
            )
            update = torch.zeros_like(masked)
            masked_confidence = confidence.masked_fill(~masked, -torch.inf)
            if remasking_strategy == "low_confidence_dynamic":
                update = masked & (confidence > dynamic_threshold)
            # Enforce the paper's per-sequence transfer schedule.  Rollouts are
            # batched locally so 8 ranks generate 128 trajectories without
            # serializing 16 independent decode loops on every rank.
            for row in range(batch_size):
                minimum = min(schedule[step_index], int(masked[row].sum().item()))
                selected_count = int(update[row].sum().item())
                if selected_count < minimum:
                    needed = minimum - selected_count
                    remaining = masked_confidence[row].masked_fill(update[row], -torch.inf)
                    selected = torch.topk(remaining, k=needed).indices
                    update[row].scatter_(0, selected, True)
            if step_index + 1 == steps:
                update |= masked
            global_denoise_step += 1
            first_reveal[update & first_reveal.eq(0)] = global_denoise_step
            block[update] = proposed[update]

        if block.eq(mask_token_id).any().item() or first_reveal.eq(0).any().item():
            raise RuntimeError(f"rollout block {block_index} did not fully denoise")
        response_blocks.append(block)
        reveal_blocks.append(first_reveal)
        generated += current_length

        globally_finished = False
        if eos_token_id is not None:
            finished |= block.eq(int(eos_token_id)).any(dim=1)
            all_finished = finished.all().to(dtype=torch.int32)
            if dist.is_initialized():
                dist.all_reduce(all_finished, op=dist.ReduceOp.MIN)
            globally_finished = bool(all_finished.item())

        if block_index + 1 < num_blocks and not globally_finished:
            set_diffusion_lm(model, False)
            cache_length = cache.get_seq_length()
            positions = torch.arange(block_start, block_start + current_length, device=device)
            refresh = _root_forward(
                model,
                lm_positions=torch.tensor([current_length - 1], device=device),
                input_ids=block,
                position_ids=positions.unsqueeze(0),
                cache_position=torch.arange(
                    cache_length, cache_length + current_length, device=device
                ),
                past_key_values=cache,
                use_cache=True,
            )
            cache = refresh.past_key_values
            del refresh
            set_diffusion_lm(model, True)
        if globally_finished:
            break

    if was_training:
        model.train()
    return torch.cat(response_blocks, dim=1), torch.cat(reveal_blocks, dim=1)

