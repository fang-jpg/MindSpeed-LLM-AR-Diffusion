# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
# Phase 3 + Phase 6: Block-wise diffusion generation for Qwen3DiffusionForCausalLM.
#
# Algorithm source: Nemotron modeling_nemotron_labs_diffusion.py:379-529
#
# Phase 6 change: Causal forward steps (prefill, post-block seed sampling) now
# use model.lm_head() instead of model.lm_head(), because lm_head is
# tied with embed_tokens (pretrained weights) while lm_head is randomly
# initialized. This ensures AR predictions are correct even before CPT training.
# Denoising steps continue using model.lm_head() for bidirectional context.

from typing import Optional

import torch
import torch.nn.functional as F

from transformers.cache_utils import DynamicCache

from ..qwen3.utils import set_diffusion_lm
import copy

def _get_num_transfer_tokens(mask_index: torch.BoolTensor, steps: int) -> torch.LongTensor:
    """Even split of masked positions across denoising steps, remainder front-loaded."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.long
    ) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : int(remainder[i])] += 1
    return num_transfer_tokens


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Gumbel-max sampling (float64 for precision)."""
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
):
    """Pick which masked positions to commit this denoising step.

    Returns (x0, transfer_index): x0 is argmax tokens; transfer_index is a
    bool mask over positions to finalize, chosen by top-k confidence.
    """
    import numpy as np

    logits_with_noise = _add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    p = F.softmax(logits, dim=-1)
    x0_p = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(-1)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device, dtype=x0_p.dtype))

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index


@torch.no_grad()
def block_diff_generate(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    block_length: int,
    threshold: Optional[float] = None,
    causal_context: bool = True,
    temperature: float = 0.0,
    topk: int = 0,
    top_p: float = 0.0,
    eos_token_id: Optional[int] = None,
    tokenizer=None,
    print_block_predictions: bool=False,
):
    """Block-wise diffusion decoding with prefix-cached KV (LLaDA-style).

    Each block: append `block_length` mask tokens, then iteratively unmask
    by confidence top-k (with optional threshold). When `causal_context`,
    the KV cache and the next-block seed are produced via a causal forward
    between blocks (flipping diffusion_lm), matching the AR objective at
    block boundaries.

    Head selection (Phase 6):
      - Causal context (prefill, post-block seed): uses lm_head (tied with
        embed_tokens, pretrained weights)
      - Bidirectional context (denoising): uses lm_head (trained for
        masked token prediction after CPT)

    Args:
        model: Qwen3DiffusionForCausalLM instance
        prompt_ids: (batch, prompt_len) input token IDs
        max_new_tokens: total number of tokens to generate
        block_length: tokens per block
        threshold: confidence threshold for denoising (None = one-shot)
        causal_context: if True, use causal forward between blocks for KV cache
        temperature: sampling temperature (0 = argmax)
        eos_token_id: optional EOS token ID for early stopping

    Returns:
        (output_ids, nfe): output_ids includes prompt, nfe = number of forward evals
    """
    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)
    mask_id = model.mask_token_id

    x_accum = prompt_ids.clone()
    B = prompt_ids.shape[0]
    prompt_len = prompt_ids.shape[1]
    device = prompt_ids.device

    assert max_new_tokens % block_length == 0, (
        f"max_new_tokens ({max_new_tokens}) must be divisible by block_length ({block_length})"
    )
    num_blocks = max_new_tokens // block_length
    steps_per_block = block_length

    nfe = 0

    # ── Initial causal prefill: produces KV cache and next-block seed ──
    if causal_context:
        set_diffusion_lm(model, False)

    past_key_values = DynamicCache()
    cache_position = torch.arange(prompt_len, device=device)
    position_ids = cache_position.unsqueeze(0).expand(B, -1)
    print("position_ids:", position_ids)
    print("cache_position:", cache_position)
    print("--------start prefill----------------")
    enc_out = model.model(
        input_ids=prompt_ids,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        cache_position=cache_position,
    )
    past_key_values = enc_out.past_key_values
    nfe += 1

    next_token = None
    print("causal_context:",causal_context)
    print("mask_id:",mask_id)
    if causal_context:
        set_diffusion_lm(model, True)
        # Causal 上下文用 lm_head 采样 seed token（与 embed_tokens 权重绑定）
        last_logit, _ = model.lm_head(enc_out.last_hidden_state[:, -1:, :])
        last_logit = last_logit.squeeze(1)
        if temperature > 0:
            next_token = torch.multinomial(
                torch.softmax(last_logit / temperature, dim=-1), num_samples=1
            )
        else:
            next_token = torch.argmax(last_logit, dim=-1, keepdim=True)

    for num_block in range(num_blocks):
        # Create a block of mask tokens, seed first position with next_token
        print("the idx of block:", num_block)
        mask_block = torch.full(
            (B, block_length), mask_id, dtype=prompt_ids.dtype, device=prompt_ids.device
        )
        if causal_context and next_token is not None:
            mask_block[:, 0] = next_token[:, 0]
        block_seed = mask_block[:, :1].clone()
        x_accum = torch.cat([x_accum, mask_block], dim=1)
        block_start = prompt_ids.size(1) + num_block * block_length
        block_slice = slice(block_start, block_start + block_length)

        mask_block_idx0 = x_accum[:, block_slice] == mask_id
        num_transfer_tokens = _get_num_transfer_tokens(mask_block_idx0, steps_per_block)

        # ── Denoise the current block by repeated confidence-based unmasking ──
        # 双向 attention 去噪用 lm_head
        for i in range(steps_per_block):
            # print("the idx of step:", i)
            mask_block_idx = x_accum[:, block_slice] == mask_id
            if mask_block_idx.sum() == 0:
                break

            nfe += 1
            denoise_position_ids = torch.arange(
                block_start, block_start + block_length, device=device
            ).unsqueeze(0).expand(B, -1)
            # print("denoise_position_ids:", denoise_position_ids)
            # print(
            #     f"denoise step={i}, "
            #     f"cache before={past_key_values.get_seq_length()}"
            # )
            denoise_cache = copy.deepcopy(past_key_values)
            # enc_out = model.model(
            #     x_accum[:, block_slice],
            #     past_key_values=past_key_values,
            #     use_cache=False,
            #     position_ids=denoise_position_ids,
            # )

            enc_out = model.model(
                x_accum[:, block_slice],
                past_key_values=denoise_cache,
                use_cache=True,
                position_ids=denoise_position_ids,
                cache_position=torch.arange(
                    past_key_values.get_seq_length(),
                    past_key_values.get_seq_length() + block_length,
                    device=device,
                ),
            )
            # print(
            #     f"denoise step={i}, "
            #     f"cache after={past_key_values.get_seq_length()}"
            # )
            logits_block, _ = model.lm_head(enc_out.last_hidden_state)
            print("mask_block_idx:", mask_block_idx)
            print("x_accum[:, block_slice]:", x_accum[:, block_slice])

            x0, transfer_idx = _get_transfer_index(
                logits_block, temperature, mask_block_idx,
                x_accum[:, block_slice],
                num_transfer_tokens=num_transfer_tokens[:, i],
                threshold=threshold,
            )
            print("x0:", x0)
            print("transfer_idx:", transfer_idx)
            cur = x_accum[:, block_slice].clone()
            cur[transfer_idx] = x0[transfer_idx]
            x_accum[:, block_slice] = cur

            if eos_token_id is not None:
                block_tokens = x_accum[:, block_slice]
                eos_mask = block_tokens == eos_token_id
                if eos_mask.any(dim=1).any():
                    after_eos = eos_mask.cumsum(dim=1).bool()
                    mask_before = (block_tokens == mask_id) & ~after_eos
                    if (eos_mask.any(dim=1) & ~mask_before.any(dim=1)).any():
                        break

        if print_block_predictions:
            completed_block = x_accum[:, block_slice].detach()

            for batch_idx in range(B):
                seed_ids = block_seed[batch_idx].tolist()
                diffusion_ids = completed_block[batch_idx, 1:].tolist()
                block_ids = completed_block[batch_idx].tolist()
                accumulated_ids = x_accum[
                    batch_idx,
                    prompt_len:block_start + block_length,
                ].tolist()

                print(f"\n[Diffusion block {num_block} | batch {batch_idx}]")
                print(f"  positions: [{block_start}, {block_start + block_length})")
                print(f"  seed ids: {seed_ids}")
                print(f"  diffusion ids: {diffusion_ids}")
                print(f"  block ids: {block_ids}")

                if tokenizer is not None:
                    print(
                        "  AR seed:",
                        repr(
                            tokenizer.decode(
                                seed_ids,
                                skip_special_tokens=False,
                            )
                        ),
                    )
                    print(
                        "  diffusion remainder:",
                        repr(
                            tokenizer.decode(
                                diffusion_ids,
                                skip_special_tokens=False,
                            )
                        ),
                    )
                    print(
                        "  completed block:",
                        repr(
                            tokenizer.decode(
                                block_ids,
                                skip_special_tokens=False,
                            )
                        ),
                    )
                    print(
                        "  accumulated generation:",
                        repr(
                            tokenizer.decode(
                                accumulated_ids,
                                skip_special_tokens=False,
                            )
                        ),
                    )
        # ── Post-block: causal forward to update KV cache and sample next seed ──
        if causal_context:
            set_diffusion_lm(model, False)

        cache_len = past_key_values.get_seq_length()
        post_cache_position = torch.arange(
            cache_len, cache_len + block_length, device=device
        )
        post_position_ids = post_cache_position.unsqueeze(0).expand(B, -1)
        
        enc_out = model.model(
            x_accum[:, block_slice],
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=post_cache_position,
            position_ids=post_position_ids,
        )
        past_key_values = enc_out.past_key_values
        nfe += 1

        if causal_context:
            set_diffusion_lm(model, True)
            # Causal 上下文用 lm_head 采样下一个 seed token
            last_logit, _ = model.lm_head(enc_out.last_hidden_state[:, -1:, :])
            last_logit = last_logit.squeeze(1)
            if temperature > 0:
                next_token = torch.multinomial(
                    torch.softmax(last_logit / temperature, dim=-1), num_samples=1
                )
            else:
                next_token = torch.argmax(last_logit, dim=-1, keepdim=True)

        # Early stop on EOS
        if eos_token_id is not None:
            gen_so_far = x_accum[:, prompt_ids.size(1):]
            is_eos = gen_so_far == eos_token_id
            if is_eos.any(dim=1).all():
                first_eos = is_eos.to(torch.int64).argmax(dim=1)
                max_eos = first_eos.max().item()
                return x_accum[:, : prompt_ids.size(1) + max_eos + 1], nfe

    return x_accum, nfe

