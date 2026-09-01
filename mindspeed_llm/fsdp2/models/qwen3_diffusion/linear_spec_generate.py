# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
# Phase 5: AR generate + Linear speculative decoding for Qwen3DiffusionForCausalLM.
#
# Algorithm source: NVIDIA Nemotron-Labs-Diffusion-8B
#   - AR generate: modeling_nemotron_labs_diffusion.py:533-623
#   - Linear spec generate: modeling_nemotron_labs_diffusion.py:626-798
#   - _crop_dynamic_cache: modeling_nemotron_labs_diffusion.py:803-811
#   - _toggle_adapters: modeling_nemotron_labs_diffusion.py:667-669

from typing import Optional

import torch
from transformers.cache_utils import DynamicCache

from ..qwen3.utils import set_diffusion_lm


# ──────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ──────────────────────────────────────────────────────────────────────────────

def _crop_dynamic_cache(past_key_values: DynamicCache, max_length: int):
    """Crop a DynamicCache to max_length, compatible with both old and new transformers.

    After linear-spec verify, we accepted only `accepted` tokens but the cache
    was filled for the entire block. This function trims the cache so that
    subsequent forwards only see the accepted prefix.
    """
    if hasattr(past_key_values, 'crop'):
        past_key_values.crop(max_length)
    else:
        for layer_idx in range(len(past_key_values)):
            past_key_values.key_cache[layer_idx] = \
                past_key_values.key_cache[layer_idx][:, :, :max_length]
            past_key_values.value_cache[layer_idx] = \
                past_key_values.value_cache[layer_idx][:, :, :max_length]
        past_key_values._seen_tokens = max_length


def _toggle_adapters(model, enable: bool):
    """Toggle PEFT/LoRA adapters on/off.

    When a PEFT adapter is attached (e.g. after Phase 6 LoRA training),
    this flips the internal _disable_adapters flag. Without adapters,
    this is a no-op — module._disable_adapters simply doesn't exist.

    - enable=True  → adapters ACTIVE  (draft phase, bidirectional)
    - enable=False → adapters DISABLED (prefill / verify phase, causal)
    """
    for module in model.modules():
        if hasattr(module, "_disable_adapters"):
            module._disable_adapters = not enable


# ──────────────────────────────────────────────────────────────────────────────
# AR generate — standard causal autoregressive decoding
# Source: Nemotron modeling_nemotron_labs_diffusion.py:533-623
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def ar_generate(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    topk: int = 0,
    top_p: float = 0.0,
    eos_token_id: Optional[int] = None,
):
    """Autoregressive generation with KV cache.

    Calls the Qwen3Model encoder directly with causal attention,
    bypassing Qwen3DiffusionForCausalLM.forward() to avoid
    diffusion-specific code paths (forward_process, block_diff mask, etc.).

    Args:
        model: Qwen3DiffusionForCausalLM instance
        prompt_ids: (batch, prompt_len) input token IDs
        max_new_tokens: number of tokens to generate
        temperature: sampling temperature (0 = greedy argmax)
        eos_token_id: optional EOS token ID for early stopping

    Returns:
        (output_ids, nfe): output_ids includes prompt, nfe = number of forward evals
    """
    # Switch all layers to causal attention
    set_diffusion_lm(model, False)

    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)

    device = prompt_ids.device
    batch_size, prompt_len = prompt_ids.shape

    # Prefill: encode the entire prompt with KV cache
    past_key_values = DynamicCache()
    cache_position = torch.arange(prompt_len, device=device)
    position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

    enc_out = model.model(
        input_ids=prompt_ids,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        cache_position=cache_position,
    )
    print(model)
    past_key_values = enc_out.past_key_values
    # AR 生成使用 lm_head（与 embed_tokens 权重绑定，继承预训练权重），
    # 而非 lm_head（随机初始化，仅 CPT 训练后才有意义）。
    next_logit, _ = model.lm_head(enc_out.last_hidden_state[:, -1:, :])
    next_logit = next_logit.squeeze(1)
    print(next_logit.shape)
    

    generated_tokens = []
    nfe = 1  # prefill counts as 1 forward

    for step in range(max_new_tokens):
        # Sample or argmax the next token
        if temperature > 0:
            next_logit = next_logit / temperature
            if int(topk) > 0:
                k = min(int(topk), next_logit.size(-1))
                top_values, _ = torch.topk(next_logit, k)
                threshold = top_values[..., -1, None]
                next_logit = torch.where(next_logit < threshold, torch.full_like(next_logit, float('-inf')), next_logit)
            # probs = torch.softmax(next_logit, dim=-1)
            # next_token = torch.multinomial(probs, num_samples=1)
            if top_p > 0.0 and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_logit,
                    descending=True
                )

                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p

                # 保留第一个超过 top_p 的 token
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logit[indices_to_remove] = filter_value
            probs = torch.softmax(next_logit, dim=-1)
            # print(probs.shape)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_logit, dim=-1, keepdim=True)
        # print("next_token:", next_token)
        generated_tokens.append(next_token)
        # print(f"eos_token_id:{eos_token_id}")
        # Early stop on EOS
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

        # Generate next logit using KV cache (single token forward)
        if step < max_new_tokens - 1:
            cur_pos = prompt_len + step
            step_cache_pos = torch.tensor([cur_pos], device=device)
            step_pos_ids = step_cache_pos.unsqueeze(0).expand(batch_size, -1)

            enc_out = model.model(
                input_ids=next_token,
                position_ids=step_pos_ids,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=step_cache_pos,
            )
            past_key_values = enc_out.past_key_values
            next_logit, _ = model.lm_head(enc_out.last_hidden_state[:, -1:, :])
            next_logit = next_logit.squeeze(1)
            nfe += 1

    all_generated = torch.cat(generated_tokens, dim=1)
    output_ids = torch.cat([prompt_ids, all_generated], dim=1)
    return output_ids, nfe


# ──────────────────────────────────────────────────────────────────────────────
# Linear speculative decoding — diffusion draft + AR verify
# Source: Nemotron modeling_nemotron_labs_diffusion.py:626-798
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def linear_spec_generate(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    mask_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    threshold: float = 0.0,
):
    """Linear speculative decoding: diffusion draft + AR verify.

    Each iteration:
      1. Draft: diffusion model (bidirectional attention) generates a block
         of candidate tokens by iteratively denoising mask tokens.
      2. Verify: AR model (causal attention) runs a single forward on the
         drafted block and produces AR predictions.
      3. Accept: the longest consecutive prefix where draft matches AR,
         plus one bonus token from AR at the first mismatch position.

    LoRA-aware: when a PEFT adapter is attached, it is toggled ON for the
    bidirectional draft phase and OFF for the causal prefille/verify phases.
    Without an adapter the calls are no-ops.

    Requires batch_size == 1 (acceptance logic is per-sample).

    Args:
        model: Qwen3DiffusionForCausalLM instance
        prompt_ids: (1, prompt_len) input token IDs (batch must be 1)
        max_new_tokens: total number of tokens to generate
        block_length: number of draft tokens per iteration
        temperature: sampling temperature (0 = greedy argmax)
        mask_token_id: override for mask token ID (default: model.config)
        eos_token_id: optional EOS token ID for early stopping
        threshold: confidence threshold for draft denoising
                   (0.0 = one-shot denoise all mask tokens at once)

    Returns:
        (output_ids, nfe, avg_accept_len):
            output_ids includes prompt
            nfe = number of forward evaluations
            avg_accept_len = average acceptance length across all iterations
    """
    if prompt_ids.shape[0] != 1:
        raise ValueError("Linear speculative decoding requires batch_size == 1")

    token_mask_id = mask_token_id if mask_token_id is not None else model.mask_token_id
    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)

    device = prompt_ids.device

    # ── Prefill (causal, LoRA OFF) ──
    set_diffusion_lm(model, False)
    _toggle_adapters(model, False)

    enc_out = model.model(
        input_ids=prompt_ids,
        past_key_values=DynamicCache(),
        use_cache=True,
    )
    past_key_values = enc_out.past_key_values
    # Prefill 是因果 AR 预测，使用 lm_head
    last_logit, _ = model.lm_head(enc_out.last_hidden_state[:, -1:, :])
    last_logit = last_logit.squeeze(1)
    nfe = 1

    # Sample the first token from AR logits
    if temperature > 0:
        next_token = torch.multinomial(
            torch.softmax(last_logit / temperature, dim=-1), num_samples=1
        )
    else:
        next_token = torch.argmax(last_logit, dim=-1, keepdim=True)

    # Edge case: first token is already EOS
    if eos_token_id is not None and next_token.item() == eos_token_id:
        return torch.cat([prompt_ids, next_token], dim=1), nfe, 1.0

    generated = [next_token]
    total_gen = 1
    accept_lengths = []  # track acceptance for reporting

    # ── Main draft-verify loop ──
    while total_gen < max_new_tokens:
        cache_len = past_key_values.get_seq_length()

        # Create a block of mask tokens, seed first position with next_token
        block = torch.full(
            (1, block_length), token_mask_id, dtype=torch.long, device=device
        )
        block[0, 0] = next_token.item()

        # ── Draft phase (bidirectional, LoRA ON) ──
        # Iteratively denoise the block. When threshold > 0, unmask only
        # high-confidence positions per step (progressive refinement).
        # When threshold == 0, unmask all in one shot.
        set_diffusion_lm(model, True)
        _toggle_adapters(model, True)

        while True:
            is_mask = block == token_mask_id
            if not is_mask.any():
                break

            # Bidirectional forward (no KV cache — draft doesn't extend prefix cache)
            enc_out = model.model(
                input_ids=block,
                past_key_values=past_key_values,
                use_cache=False,
            )
            nfe += 1

            draft_logits, _ = model.lm_head(enc_out.last_hidden_state)

            if temperature > 0:
                draft_probs = torch.softmax(draft_logits / temperature, dim=-1)
                draft_tokens = torch.multinomial(
                    draft_probs.view(-1, draft_logits.shape[-1]), num_samples=1
                ).view(1, block_length)
            else:
                draft_tokens = draft_logits.argmax(dim=-1)
                draft_probs = torch.softmax(draft_logits, dim=-1)

            if threshold > 0:
                # Progressive denoising: only unmask positions above confidence threshold
                draft_conf = torch.gather(
                    draft_probs, -1, draft_tokens.unsqueeze(-1)
                ).squeeze(-1)
                draft_conf = torch.where(is_mask, draft_conf, -torch.inf)
                unmask = draft_conf >= threshold
                # Force at least 1 position to unmask per step (guarantee progress)
                if not unmask.any():
                    best_idx = draft_conf.view(-1).argmax()
                    unmask = torch.zeros_like(is_mask, dtype=torch.bool)
                    unmask.view(-1)[best_idx] = True
                block[unmask] = draft_tokens[unmask]
            else:
                # One-shot: unmask all remaining mask positions at once
                block[is_mask] = draft_tokens[is_mask]
                break

        # ── Verify phase (causal, LoRA OFF) ──
        set_diffusion_lm(model, False)
        _toggle_adapters(model, False)

        enc_out = model.model(
            input_ids=block,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = enc_out.past_key_values
        nfe += 1

        # Verify 是因果 AR 预测，使用 lm_head（而非 lm_head）
        verify_logits, _ = model.lm_head(enc_out.last_hidden_state)

        if temperature > 0:
            ar_tokens = torch.multinomial(
                torch.softmax(verify_logits / temperature, dim=-1).view(
                    -1, verify_logits.shape[-1]
                ),
                num_samples=1,
            ).view(1, block_length)
        else:
            ar_tokens = verify_logits.argmax(dim=-1)

        # ── Accept: consecutive matches + 1 bonus token ──
        # AR token at position i predicts the next token (position i+1).
        # We verify by comparing ar_tokens[i] (AR prediction) with
        # draft_tokens[i+1] (the draft's token at the next position).
        accepted = 0
        for i in range(block_length - 1):
            # ar_tokens[0, i] = AR's prediction of what comes after position i
            # block[0, i+1]   = draft's token at position i+1
            if ar_tokens[0, i].item() == block[0, i + 1].item():
                accepted += 1
            else:
                break
        accepted += 1  # bonus token: the first AR token past the match prefix

        accept_lengths.append(accepted)
        accepted_toks = ar_tokens[:, :accepted]
        generated.append(accepted_toks)
        total_gen += accepted

        # Crop KV cache to accepted prefix length
        _crop_dynamic_cache(past_key_values, cache_len + accepted)

        # Next seed token = last accepted AR token
        next_token = ar_tokens[:, accepted - 1 : accepted]

        # Early stop on EOS
        if eos_token_id is not None:
            eos_pos = (accepted_toks[0] == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                first_eos = eos_pos[0].item()
                generated[-1] = accepted_toks[:, : first_eos + 1]
                total_gen = total_gen - accepted + first_eos + 1
                break

        if total_gen >= max_new_tokens:
            break

    all_generated = torch.cat(generated, dim=1)
    output_ids = torch.cat([prompt_ids, all_generated], dim=1)
    avg_accept = sum(accept_lengths) / len(accept_lengths) if accept_lengths else 1.0
    return output_ids, nfe, avg_accept
