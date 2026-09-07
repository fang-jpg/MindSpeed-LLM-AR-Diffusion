# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
# Phase 2 + Phase 3: Qwen3-based Diffusion LM with bidirectional and block_diff paradigms.
#
# Algorithm source: NVIDIA Nemotron-Labs-Diffusion-8B
#   - forward_process: modeling_nemotron_labs_diffusion.py:201-232
#   - Diffusion loss (bidirectional): modeling_nemotron_labs_diffusion.py:333-346
#   - Block_diff forward: modeling_nemotron_labs_diffusion.py:259-367
#   - Joint loss: modeling_nemotron_labs_diffusion.py:349-365
#   - Config fields: configuration_nemotron_labs_diffusion.py:131-137

import copy
from typing import Optional, Union, Any
from transformers.generation import GenerationMixin
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, Qwen3Config
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import can_return_tuple, ModelOutput

from mindspeed.patch_utils import MindSpeedPatchesManager as pm
from mindspeed_llm.fsdp2.models.common.fusions import (
    apply_rotary_pos_emb,
    fused_rmsnorm_forward,
)
from mindspeed_llm.fsdp2.models.common.modules import LMHead
from dataclasses import dataclass, field


@dataclass
class Qwen3DiffusionOutput(ModelOutput):
    loss: Optional[Any] = None
    logits: Optional[torch.FloatTensor] = None
    causal_logits: Optional[torch.FloatTensor] = None
    masked_token_count: Optional[torch.Tensor] = None
    ar_token_count: Optional[torch.Tensor] = None
    diffusion_loss_sum: Optional[torch.Tensor] = None
    ar_loss_sum: Optional[torch.Tensor] = None
    token_loss_details: Optional[dict] = None


@dataclass
class Qwen3TokenLossOutput(CausalLMOutputWithPast):
    token_loss_details: Optional[dict] = None


class Qwen3DiffusionForCausalLM(transformers.Qwen3PreTrainedModel, GenerationMixin):
    """Qwen3-based Diffusion Language Model supporting bidirectional and block_diff paradigms.

    Single shared Qwen3Model backbone with:
      - diffusion_head: predicts masked tokens (diffusion training objective)
      - lm_head: standard AR head (for block_diff AR loss)
    Attention is swapped to Qwen3AttentionWithDiffusionToggle after init,
    defaulting to diffusion_lm=True (bidirectional) for diffusion training.

    Paradigms:
      - 'bidirectional': Phase 2. Random masking + bidirectional attention + diffusion loss.
      - 'block_diff': Phase 3. Concatenated [noisy, original] + block diagonal mask
        + joint AR + diffusion loss.
    """

    _tied_weights_keys = []
    _tp_plan = {}
    _pp_plan = {}
    supports_token_loss_logging = True

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.mask_token_id = getattr(config, "mask_token_id", -1)
        self.bd_mask = None
        self.causal_mask = None
        self.block_diff_position_ids = None

        diffusion_config = copy.deepcopy(config)
        #训练时为了方便强制指定，推理的时候需要注释掉，用config读取
        # diffusion_config.diffusion_lm = True
        self.config.diffusion_lm = False

        self.model = transformers.Qwen3Model(diffusion_config)

        self.vocab_size = config.vocab_size

        self.lm_head = LMHead(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

        # Phase 1: Replace Qwen3Attention with diffusion-toggleable variant.
        # Must happen after post_init() so pretrained weights are already loaded.
        from ..qwen3.qwen3_attention import Qwen3AttentionWithDiffusionToggle, _swap_attention
        for layer in self.model.layers:
            _swap_attention(layer, Qwen3AttentionWithDiffusionToggle)

        # Phase 4: track loss components for training-step logging
        self._last_loss_components = {"diff_loss": 0.0, "ar_loss": 0.0}

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Phase 4: Auto-resize embeddings to accommodate mask_token_id.

        Qwen3-1.7B-Base has no <mask> token in its vocab. After loading,
        if mask_token_id >= vocab_size, resize embed_tokens + lm_head
        (handled by resize_token_embeddings) and rebuild diffusion_head
        (not handled by resize_token_embeddings, since it's not the
        get_output_embeddings output).
        """
        print("---------------pretrain start---------------------")
        model, loading_info  = super().from_pretrained(*args,output_loading_info=True,**kwargs)
        ##训练时为了方便强制指定，推理的时候需要注释/删除权重拷贝
        # with torch.no_grad():
        #     print("copied")
        #     model.lm_head.weight.copy_(
        #         model.model.embed_tokens.weight
        #     )

        return model

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward_process(self, input_ids, eps=1e-3, loss_mask=None):
        """Random masking procedure (LLaDA-style).

        Samples a per-batch mask ratio p_mask ~ Uniform(eps, 1), then
        independently masks each token with probability p_mask. Masked
        positions are replaced with self.mask_token_id.

        Args:
            input_ids: (batch, seq_len) token IDs
            eps: lower bound for mask ratio sampling (prevents p=0)
            loss_mask: optional (batch, seq_len) binary mask — positions
                       with 0 are excluded from masking

        Returns:
            noisy_batch: (batch, seq_len) with masked positions replaced
            masked_indices: (batch, seq_len) bool — True where masked
            p_mask: (batch, seq_len) float — per-position mask probability
        """
        b, l = input_ids.shape
        device = input_ids.device

        if getattr(self.config, "dp_varying_mask_ratio", False):
            import torch.distributed as dist
            dp_rank = dist.get_rank() if dist.is_initialized() else 0
            generator = torch.Generator(device=device)
            generator.manual_seed(torch.seed() + dp_rank)
        else:
            generator = None

        t = torch.rand(b, device=device, generator=generator)
        p_mask = (1 - eps) * t + eps          # (b,)
        p_mask = p_mask[:, None].expand(-1, l) # (b, l)

        masked_indices = torch.rand((b, l), device=device) < p_mask

        if loss_mask is not None:
            masked_indices[loss_mask == 0] = 0

        noisy_batch = torch.where(masked_indices, self.mask_token_id, input_ids)
        return noisy_batch, masked_indices, p_mask

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        eps: float = 1e-3,
        loss_ctx: Optional[callable] = None,
        loss_mask: Optional[torch.Tensor] = None,
        log_per_token_loss: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Forward pass supporting bidirectional and block_diff paradigms.

        Bidirectional (Phase 2):
          1. forward_process() randomly masks tokens
          2. Encoder forward with bidirectional attention
          3. diffusion_head predicts masked tokens
          4. CE loss on masked positions, debiased by p_mask

        Block_diff (Phase 3):
          1. forward_process() randomly masks tokens
          2. Concatenate [noisy_inputs, input_ids] → (B, 2L)
          3. Construct block diagonal attention mask
          4. Construct position_ids with RoPE break at L
          5. Encoder forward with block_diff mask
          6. Split logits → diffusion loss on noisy half + AR loss on original half
          7. Joint loss = dlm_loss_weight * L_diff + ar_loss_weight * L_ar
        """
        # # print(paradigm)
        # paradigm = "autoregressive"
        
        paradigm = getattr(self.config, "dlm_paradigm", "bidirectional")
        if log_per_token_loss and loss_ctx is not None:
            raise ValueError("Per-token loss logging requires chunk_loss_size to be disabled.")
        token_loss_details = {} if log_per_token_loss and labels is not None else None
        # ── 1. Apply forward_process during training ──
        if labels is not None and paradigm != "autoregressive":
            noisy_inputs, masked_indices, p_mask = self.forward_process(
                input_ids, eps=eps, loss_mask = loss_mask
            )
        else:
            noisy_inputs = input_ids
            masked_indices = None
            p_mask = None
            # if self.causal_mask is None:
            #     self.causal_mask = self.make_causal_train_mask(
            #         seq_len=noisy_inputs.shape[1],
            #         batch_size=noisy_inputs.shape[0],
            #         dtype=next(self.model.parameters()).dtype,
            #         device=input_ids.device,
            #     )
            # attention_mask = self.causal_mask

        # ── 2. Block_diff: concatenate + mask + position_ids ──
        causal_logits = None
        input_ids_len = noisy_inputs.shape[1]

        if labels is not None and paradigm == "block_diff":
            # Concatenate [noisy, original]
            noisy_inputs = torch.cat([noisy_inputs, input_ids], dim=1)  # (B, 2L)
            # Block diagonal attention mask (cached after first call)
            if self.bd_mask is None:
                block_size = getattr(self.config, "block_size", 8)
                model_dtype = next(self.model.parameters()).dtype
                self.bd_mask = self.make_block_diff_mask(
                    seq_len=input_ids_len,
                    block_size=block_size,
                    batch_size=input_ids.shape[0],
                    dtype=model_dtype,
                    device=input_ids.device,
                )
            attention_mask = self.bd_mask

            # Construct position_ids with RoPE break at boundary
            # First half: [0, 1, ..., L-1], second half: [0, 1, ..., L-1]
            if position_ids is None:
                if self.block_diff_position_ids is None or input_ids_len != self.block_diff_position_ids.shape[-1] // 2:
                    half_pos = torch.arange(input_ids_len, device=input_ids.device)
                    self.block_diff_position_ids = torch.cat([half_pos, half_pos]).unsqueeze(0)
                position_ids = self.block_diff_position_ids.expand(
                    input_ids.shape[0], -1
                )
                # print("position_ids:", position_ids)

        # ── 3. Encoder forward ──
        outputs = self.model(
            input_ids=noisy_inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        # ── 4. Head logits (single shared head for both halves) ──
        # Single forward through diffusion_head produces logits for both halves.
        # Nemotron-style: the same weight handles both diffusion and AR prediction.
        if loss_ctx:
            logits, loss = self.lm_head(hidden_states, loss_ctx=loss_ctx)
        else:
            logits, _ = self.lm_head(hidden_states)

        # Block_diff: split the shared head output for each half.
        causal_logits = None
        logits_orig = logits  # keep full output before slicing
        if labels is not None and paradigm == "block_diff":
            # Noisy half → diffusion loss
            logits = logits_orig[:, :input_ids_len]
            # Original half → AR loss (same head, different slice)
            causal_logits = logits_orig[:, input_ids_len:]

        # ── 5. Loss computation ──
        loss = None

        if labels is not None and loss_ctx is None:
            if paradigm == "autoregressive":
                causal_logits = logits.contiguous().float()
                shift_labels = labels
                shift_labels = shift_labels.reshape(-1)
                causal_logits = causal_logits.view(-1, logits.shape[-1])
                if log_per_token_loss:
                    ar_token_loss = F.cross_entropy(
                        causal_logits, shift_labels, reduction="none", ignore_index=-100
                    )
                    # Labels are already shifted by the data loader. Preserve the
                    # original mean reduction (including its all-ignored behavior).
                    loss = ar_token_loss.sum() / shift_labels.ne(-100).sum()
                    token_loss_details["ar"] = {
                        "losses": ar_token_loss.detach().reshape_as(labels),
                        "target_ids": labels.detach(),
                        "valid_mask": labels.ne(-100),
                    }
                else:
                    loss = F.cross_entropy(causal_logits, shift_labels, reduction="mean", ignore_index=-100)
                ar_loss_value = loss.item()
                # if self.training and torch.is_grad_enabled():
                #     # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                #     print(f"ar_loss:{ar_loss_value}")
            else:
                diff_labels = input_ids.clone()
                # Diffusion loss: LLaDA-style on masked positions
                raw_diff_token_loss = F.cross_entropy(
                    logits[masked_indices],
                    diff_labels[masked_indices],
                    reduction="none",
                    ignore_index=-100
                )
                token_loss = raw_diff_token_loss / p_mask[masked_indices].clamp_min(1e-3)

                num_mask_tokens = masked_indices.sum()
                diff_loss = token_loss.sum()
                # if num_mask_tokens > 0:
                #     diff_loss = token_loss.sum() / num_mask_tokens
                # else:
                #     diff_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)

                # Apply dlm_loss_weight if specified
                dlm_weight = getattr(self.config, "dlm_loss_weight", 0.5)
                # dlm_weight = 0.1
                if dlm_weight is not None:
                    diff_loss = dlm_weight * diff_loss
                loss = diff_loss
                if log_per_token_loss:
                    # Only masked positions have an observed diffusion loss.
                    # Keep original [batch, sequence] positions after selection.
                    raw_losses = torch.zeros_like(p_mask, dtype=raw_diff_token_loss.dtype)
                    weighted_losses = torch.zeros_like(p_mask, dtype=token_loss.dtype)
                    raw_losses[masked_indices] = raw_diff_token_loss.detach()
                    weighted_losses[masked_indices] = token_loss.detach() * (
                        dlm_weight if dlm_weight is not None else 1.0
                    )
                    token_loss_details["diffusion"] = {
                        "losses": raw_losses,
                        "target_ids": diff_labels.detach(),
                        "valid_mask": masked_indices & diff_labels.ne(-100),
                        "weighted_losses": weighted_losses,
                        "p_mask": p_mask.detach(),
                    }

                # Block_diff: add AR loss from original half (shared diffusion_head)
                ar_loss_value = 0.0

                if paradigm == "block_diff" and causal_logits is not None:
                    ar_weight = getattr(self.config, "ar_loss_weight", 1.0)

                    # labels 已经在数据侧完成 shift，因此模型内部不再 shift。
                    # causal_logits: [batch_size, seq_len, vocab_size]
                    # labels:        [batch_size, seq_len]
                    flat_ar_logits = causal_logits.contiguous().float().reshape(
                        -1, causal_logits.size(-1)
                    )
                    flat_ar_labels = labels.contiguous().reshape(-1)

                    # 当前 rank 整个 batch 内所有有效 AR token 的 loss 之和。
                    ar_loss = F.cross_entropy(
                        flat_ar_logits,
                        flat_ar_labels,
                        reduction="none" if log_per_token_loss else "sum",
                        ignore_index=-100,
                    )
                    if log_per_token_loss:
                        token_loss_details["ar"] = {
                            "losses": ar_loss.detach().reshape_as(labels),
                            "target_ids": labels.detach(),
                            "valid_mask": labels.ne(-100),
                            "weighted_losses": ar_loss.detach().reshape_as(labels) * ar_weight,
                        }
                        ar_loss = ar_loss.sum()

                    # 统计整个 batch 的有效 AR token 数，而不是只使用 seq_len。
                    ar_token_count = flat_ar_labels.ne(-100).sum()

                    # 联合 loss 的分子：
                    # dlm_weight * diffusion_loss_sum + ar_weight * ar_loss_sum
                    weighted_loss_sum = loss + ar_weight * ar_loss

                    # 联合 loss 的分母。
                    # num_mask_tokens 已经包含当前 rank 整个 batch 的 mask token。
                    weighted_token_count = (
                        num_mask_tokens.to(dtype=torch.float32)
                        + ar_weight * ar_token_count.to(dtype=torch.float32)
                    )

                    ar_loss_value = ar_loss.detach().item()
                    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                        print(f"diff_loss:{diff_loss.item()}")
                        print(f"ar_loss:{ar_loss_value}")

                    return Qwen3DiffusionOutput(
                        loss=(weighted_loss_sum, weighted_token_count),
                        logits=logits,

                        # 保留原始三维形状，不返回已经 flatten 的 logits。
                        causal_logits=causal_logits,

                        masked_token_count=num_mask_tokens,
                        ar_token_count=ar_token_count,

                        # 注意：你当前的 diff_loss 已经乘过 dlm_weight。
                        diffusion_loss_sum=diff_loss,
                        ar_loss_sum=ar_loss,
                        token_loss_details=token_loss_details,
                    )
                # Phase 4: record per-component losses for the trainer to read.
                # The trainer accesses these via _get_raw_model()._last_loss_components
                # after each micro-batch forward pass.
                # if self.training and torch.is_grad_enabled():
                #     self._last_loss_components = {
                #         "diff_loss": diff_loss.item(),
                #         "ar_loss": ar_loss_value,
                #     }
                    # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    #     print(f"diff_loss:{diff_loss.item()}")
                    #     print(f"ar_loss:{ar_loss_value}")

        output_cls = Qwen3TokenLossOutput if token_loss_details is not None else CausalLMOutputWithPast
        diagnostic_kwargs = {"token_loss_details": token_loss_details} if token_loss_details is not None else {}
        return output_cls(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            **diagnostic_kwargs,
        )

    @staticmethod
    def make_block_diff_mask(
        seq_len: int,
        block_size: int,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Construct the block_diff 4D attention mask.

        For a concatenated input [noisy(0..L), original(L..2L)]:
          - M_BD: noisy→noisy, same block → bidirectional (attend)
          - M_OBC: noisy→original, earlier blocks → attend (conditioning)
          - M_BC: original→original → standard causal

        Args:
            seq_len: length of one half (L). Total sequence is 2L.
            block_size: block size for block-diagonal partitioning.
            batch_size: batch dimension (expanded after construction).
            dtype: mask dtype (should match model dtype).
            device: device for the mask tensor.

        Returns:
            mask: (batch_size, 1, 2L, 2L) float tensor.
                  0.0 = attend, -inf = blocked
        """
        L = seq_len
        n = L
        total_len = 2 * L
        # print("block_size:",block_size)

        q_idx = torch.arange(total_len, device=device).unsqueeze(1)
        kv_idx = torch.arange(total_len, device=device).unsqueeze(0)

        x0_flag_q = q_idx >= n
        x0_flag_kv = kv_idx >= n

        block_q = torch.where(x0_flag_q, (q_idx - n) // block_size, q_idx // block_size)
        block_kv = torch.where(x0_flag_kv, (kv_idx - n) // block_size, kv_idx // block_size)

        m_bd = (block_q == block_kv) & (~x0_flag_kv) & (~x0_flag_q)
        m_obc = (block_q > block_kv) & x0_flag_kv & (~x0_flag_q)
        m_bc = (q_idx >= kv_idx) & x0_flag_kv & x0_flag_q

        attend = m_bd | m_obc | m_bc

        mask = torch.full((1, 1, total_len, total_len), float("-inf"), dtype=dtype, device=device)
        mask[0, 0][attend] = 0.0

        if batch_size > 1:
            mask = mask.expand(batch_size, -1, -1, -1)

        return mask

    @staticmethod
    def make_causal_train_mask(
        seq_len: int,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Construct the standard causal 4D attention mask for AR training.

        For a sequence input_ids[0..L):
        - token i can attend to token j only if j <= i
        - future tokens j > i are masked

        Args:
            seq_len: sequence length L.
            batch_size: batch dimension.
            dtype: mask dtype, should match model dtype.
            device: device for the mask tensor.

        Returns:
            mask: (batch_size, 1, L, L) float tensor.
                0.0 = attend
                -inf = blocked
        """
        L = seq_len

        q_idx = torch.arange(L, device=device).unsqueeze(1)   # (L, 1)
        kv_idx = torch.arange(L, device=device).unsqueeze(0)  # (1, L)

        # causal attend rule:
        # query position q can attend key/value position kv if kv <= q
        attend = kv_idx <= q_idx

        mask = torch.full(
            (1, 1, L, L),
            float("-inf"),
            dtype=dtype,
            device=device,
        )
        mask[0, 0][attend] = 0.0

        if batch_size > 1:
            mask = mask.expand(batch_size, -1, -1, -1)

        return mask

    @staticmethod
    def register_patches(config):
        """Patch transformers modules for fused operations."""
        if getattr(config, "use_fused_rmsnorm", False):
            pm.register_patch(
                "transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm.forward",
                fused_rmsnorm_forward,
            )
        if getattr(config, "use_fused_rotary_pos_emb", False):
            pm.register_patch(
                "transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb",
                apply_rotary_pos_emb,
            )
        pm.apply_patches()


__all__ = ["Qwen3DiffusionForCausalLM", "Qwen3DiffusionOutput"]
