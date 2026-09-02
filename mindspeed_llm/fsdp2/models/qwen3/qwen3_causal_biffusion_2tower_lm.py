# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Qwen3 two-tower model for causal biffusion."""

from typing import Optional

import torch
import torch.nn.functional as F
import transformers
from transformers.modeling_outputs import CausalLMOutputWithPast

from mindspeed_llm.fsdp2.models.qwen3.qwen3 import Qwen3ForCausalLM


class Qwen3CausalBiffusion2TowerLm(transformers.Qwen3PreTrainedModel):
    """A pair of independent, complete Qwen3 causal language models.

    The encoder and decoder both contain a Qwen3 backbone and an LM head. The
    encoder is frozen at construction time, while every decoder parameter
    remains trainable. Decoder attention is bidirectional inside each block
    and causal between blocks.
    """

    def __init__(self, config):
        super().__init__(config)
        self.block_size = getattr(config, "block_size", None)

        original_vocab_size = config.vocab_size
        configured_mask_token_id = getattr(config, "mask_token_id", None)
        self.mask_token_id = (
            151669
            if configured_mask_token_id is None
            else configured_mask_token_id
        )
        if self.mask_token_id < 0 or self.mask_token_id > original_vocab_size:
            raise ValueError(
                "mask_token_id must reference the existing vocabulary or the "
                f"new row at index {original_vocab_size}, but got "
                f"{self.mask_token_id}."
            )

        # When mask_token_id equals the old vocabulary length, reserve one new
        # row in both the token embedding and LM head before FSDP2 wrapping.
        expanded_vocab_size = max(original_vocab_size, self.mask_token_id + 1)
        config.vocab_size = expanded_vocab_size
        config.mask_token_id = self.mask_token_id

        self.encoder = Qwen3ForCausalLM(config)
        self.decoder = Qwen3ForCausalLM(config)
        self.freeze_encoder()

    def freeze_encoder(self):
        """Freeze all encoder parameters and return this model."""
        self.encoder.requires_grad_(False)
        return self

    def _validate_diffusion_config(self):
        if self.block_size is None or self.block_size <= 0:
            raise ValueError("block_size must be a positive integer.")
        if not 0 <= self.mask_token_id < self.config.vocab_size:
            raise ValueError(
                f"mask_token_id must be in [0, {self.config.vocab_size}), "
                f"but got {self.mask_token_id}."
            )

    def _sample_block_mask(
        self,
        input_ids: torch.LongTensor,
        eps: float = 1e-3,
    ):
        """Mask tokens using an independently sampled ratio for every block."""
        if not 0 < eps <= 1:
            raise ValueError(f"eps must be in (0, 1], but got {eps}.")

        batch_size, sequence_length = input_ids.shape
        masked_blocks = []
        mask_blocks = []
        probability_blocks = []

        for block_start in range(0, sequence_length, self.block_size):
            block = input_ids[:, block_start:block_start + self.block_size]
            block_length = block.shape[1]

            t = torch.rand(batch_size, device=input_ids.device)
            block_p_mask = ((1 - eps) * t + eps).unsqueeze(1)
            block_p_mask = block_p_mask.expand(-1, block_length)
            block_mask = torch.rand(
                (batch_size, block_length),
                device=input_ids.device,
            ) < block_p_mask
            masked_block = torch.where(block_mask, self.mask_token_id, block)

            masked_blocks.append(masked_block)
            mask_blocks.append(block_mask)
            probability_blocks.append(block_p_mask)

        return (
            torch.cat(masked_blocks, dim=1),
            torch.cat(mask_blocks, dim=1),
            torch.cat(probability_blocks, dim=1),
        )

    def _build_block_causal_attention_mask(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Build bidirectional intra-block and causal inter-block attention."""
        batch_size, sequence_length = input_ids.shape
        token_positions = torch.arange(sequence_length, device=input_ids.device)
        block_indices = token_positions.div(self.block_size, rounding_mode="floor")

        # Query tokens can see every key in their own block and earlier blocks.
        allowed = block_indices.unsqueeze(0) <= block_indices.unsqueeze(1)
        allowed = allowed.unsqueeze(0).unsqueeze(0).expand(
            batch_size,
            1,
            sequence_length,
            sequence_length,
        )

        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape != input_ids.shape:
                raise ValueError(
                    "attention_mask must have shape (batch_size, sequence_length)."
                )
            valid_keys = attention_mask.to(device=input_ids.device, dtype=torch.bool)
            allowed = allowed & valid_keys[:, None, None, :]

        model_dtype = self.decoder.get_input_embeddings().weight.dtype
        block_causal_mask = torch.zeros(
            allowed.shape,
            dtype=model_dtype,
            device=input_ids.device,
        )
        return block_causal_mask.masked_fill(
            ~allowed,
            torch.finfo(model_dtype).min,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        eps: float = 1e-3,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Compute frozen AR-teacher and trainable block-diffusion losses."""
        if labels is None:
            raise ValueError("labels must be provided when training Qwen3CausalBiffusion2TowerLm.")
        if input_ids.ndim != 2 or labels.shape != input_ids.shape:
            raise ValueError(
                "input_ids and labels must both have shape (batch_size, sequence_length)."
            )
        self._validate_diffusion_config()

        with torch.no_grad():
            encoder_outputs = self.encoder.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )
            ar_logits, _ = self.encoder.lm_head(encoder_outputs.last_hidden_state)

        masked_input_ids, masked_indices, p_mask = self._sample_block_mask(
            input_ids,
            eps=eps,
        )
        block_causal_attention_mask = self._build_block_causal_attention_mask(
            masked_input_ids,
            attention_mask,
        )
        decoder_outputs = self.decoder.model(
            input_ids=masked_input_ids,
            attention_mask=block_causal_attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        diffusion_logits, _ = self.decoder.lm_head(decoder_outputs.last_hidden_state)

        loss1 = F.kl_div(
            F.log_softmax(diffusion_logits.float(), dim=-1),
            F.softmax(ar_logits.float(), dim=-1),
            reduction="batchmean",
        )

        valid_masked_indices = masked_indices & labels.ne(-100)
        if valid_masked_indices.any():
            token_loss = F.cross_entropy(
                diffusion_logits[valid_masked_indices].float(),
                labels[valid_masked_indices],
                reduction="none",
            ) / p_mask[valid_masked_indices]
            loss2 = token_loss.sum()
        else:
            loss2 = diffusion_logits.sum() * 0.0

        return CausalLMOutputWithPast(
            loss=loss1 + loss2,
            logits=diffusion_logits,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )

    @staticmethod
    def register_patches(config):
        """Register the same optimized kernels used by standard Qwen3."""
        Qwen3ForCausalLM.register_patches(config)
