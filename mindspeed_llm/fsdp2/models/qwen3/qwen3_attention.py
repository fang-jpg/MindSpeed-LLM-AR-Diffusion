

from typing import Optional

import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3RMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)


class Qwen3AttentionWithDiffusionToggle(Qwen3Attention):


    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.diffusion_lm = getattr(config, "diffusion_lm", True)
        # print("self.diffusion_lm:", self.diffusion_lm)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # q/k/v projections + q_norm/k_norm (Qwen3-specific) — identical to HF original
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings


        is_block_diff = (
            self.diffusion_lm
            and attention_mask is not None and attention_mask.dim() == 4
            and self.training
        )

        if is_block_diff:
            # print("is_block_diff")
            q1, q2 = query_states.chunk(2, dim=2)
            k1, k2 = key_states.chunk(2, dim=2)
            q1, k1 = apply_rotary_pos_emb(q1, k1, cos, sin)
            q2, k2 = apply_rotary_pos_emb(q2, k2, cos, sin)
            query_states = torch.cat([q1, q2], dim=2)
            key_states = torch.cat([k1, k2], dim=2)
        else:
            # print("is_not_block_diff")
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.diffusion_lm:
            # print("diffusion_lm")
            # Bidirectional / block_diff: when an external 4D mask is provided
            # (block_diff paradigm), pass it through; otherwise use no mask.
            use_custom_mask = attention_mask is not None and attention_mask.dim() == 4
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=attention_mask if use_custom_mask else None,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                is_causal=False,
                **kwargs,
            )
        else:
            # Causal: identical to original Qwen3Attention
            # print("no_diffusion_lm")
            # print("attention_mask:",attention_mask)
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def _swap_attention(layer: nn.Module, attn_class: type) -> None:
    """Replace layer.self_attn with attn_class, copying all existing weights.

    Creates a new attention module of the given class, loads the state dict
    from the current module, and swaps it in-place. This preserves any
    pretrained weights that were loaded before the swap.
    """
    old_attn = layer.self_attn
    new_attn = attn_class(config=old_attn.config, layer_idx=old_attn.layer_idx)
    new_attn.load_state_dict(old_attn.state_dict(), strict=True)
    # Preserve device & dtype
    new_attn = new_attn.to(next(old_attn.parameters()).device, dtype=next(old_attn.parameters()).dtype)
    layer.self_attn = new_attn
