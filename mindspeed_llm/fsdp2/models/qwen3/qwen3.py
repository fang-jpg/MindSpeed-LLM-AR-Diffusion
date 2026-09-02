# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
from typing import Optional, Union

import torch
try:
    import torch_npu
except ImportError:
    pass
import transformers
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import can_return_tuple
from transformers.generation import GenerationMixin
from mindspeed.patch_utils import MindSpeedPatchesManager as pm
from mindspeed_llm.fsdp2.models.common.fusions import apply_rotary_pos_emb, \
    fused_rmsnorm_forward
from mindspeed_llm.fsdp2.models.common.modules import LMHead


class Qwen3ForCausalLM(transformers.Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = transformers.Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = LMHead(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Phase 1: Replace Qwen3Attention with diffusion-toggleable variant.
        # Must happen after post_init() so pretrained weights are already loaded.
        # if getattr(config, "enable_diffusion_lm", False):
        #     from .qwen3_attention import Qwen3AttentionWithDiffusionToggle, _swap_attention
        #     for layer in self.model.layers:
        #         _swap_attention(layer, Qwen3AttentionWithDiffusionToggle)

    @can_return_tuple
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            loss_ctx: Optional[callable] = None,
            **kwargs,
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        if loss_ctx:
            logits, loss = self.lm_head(hidden_states[:, slice_indices, :], loss_ctx=loss_ctx)
        else:
            logits, loss = self.lm_head(hidden_states[:, slice_indices, :])

            loss = None
            if labels is not None:
                shift_labels = labels
                shift_labels = shift_labels.reshape(-1)
                logits = logits.view(-1, logits.shape[-1])

                loss = F.cross_entropy(logits, shift_labels, ignore_index=-100)
                print(f"_compute_language_model_pretrain_loss:{loss}")
                print(f"kwargs:{kwargs}")
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
                print(f"self.loss_function:{self.loss_function}")
                print(f"self.loss_function:{self.loss_function}")
                print(f"type(self.loss_function):{type(self.loss_function)}")
                print(f"has reduction:{hasattr(self.loss_function, 'reduction')}")

                logits_for_loss = logits[..., :self.config.vocab_size].float()

                labels = F.pad(labels, (0, 1), value=-100)
                shift_labels = labels[..., 1:].contiguous()
                logits_for_loss = logits_for_loss.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                shift_labels = shift_labels.to(logits_for_loss.device)
                manual_loss = F.cross_entropy(logits_for_loss, shift_labels, ignore_index=-100, reduction="mean")

                # lens = (tokens['input_ids'] != tokenizer.tokenizer.pad_token_id).sum(-1)
                # ce_loss = loss.float().sum(-1).cpu().detach().numpy() / lens.cpu().numpy()
                print(f"loss, manual_loss, ratio:{loss}, {manual_loss}, {manual_loss/loss}")
                num_items = kwargs.get("num_items_in_batch", None)

                valid_tokens_local = shift_labels.ne(-100).sum()

                manual_sum = F.cross_entropy(
                    logits_for_loss,
                    shift_labels,
                    ignore_index=-100,
                    reduction="sum"
                )

                manual_mean = F.cross_entropy(
                    logits_for_loss,
                    shift_labels,
                    ignore_index=-100,
                    reduction="mean"
                )

                manual_global_norm = manual_sum / num_items

                print(f"valid_tokens_local:{valid_tokens_local}")
                print(f"num_items_in_batch:{num_items}")
                print(f"num_items/local:{num_items / valid_tokens_local.item()}")
                print(f"loss:{loss}")
                print(f"manual_mean:{manual_mean}")
                print(f"manual_sum/num_items:{manual_global_norm}")
                print(f"manual_mean/loss:{manual_mean / loss}")
                print(f"manual_global_norm/loss:{manual_global_norm / loss}")
                print("-------++++++++++++++++-------------")


        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @staticmethod
    def register_patches(config):
        """patching the transformers model."""
        if getattr(config, "use_fused_rmsnorm", False):
            pm.register_patch("transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm.forward",
                              fused_rmsnorm_forward)
        if getattr(config, "use_fused_rotary_pos_emb", False):
            pm.register_patch("transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb",
                              apply_rotary_pos_emb)

        pm.apply_patches()
