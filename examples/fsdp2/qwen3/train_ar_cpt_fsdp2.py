"""Isolated FSDP2 entrypoint for a native Qwen3 autoregressive CPT baseline.

The Megatron-style pretraining dataset used by this repository already returns
next-token labels (``input_ids = text[:-1]`` and ``labels = text[1:]``).  The
generic Trainer therefore has to compute CE directly from the returned logits;
passing those labels into a HuggingFace causal-LM loss would shift them twice.

Qwen3-1.7B-Base stores its tied input/output embedding matrix only once.  This
entrypoint supplies an AR-only model class that restores ``lm_head.weight``
from ``model.embed_tokens.weight`` before FSDP wrapping, even when a local
config asks to keep the two parameters independent.

This entrypoint deliberately changes only the newly-created training process.
It does not modify the source model config, the diffusion model, or the shared
``train_fsdp2.py`` entrypoint.
"""

import torch

from train_fsdp2 import AutoTrainer
from mindspeed_llm.fsdp2.models.model_registry import ModelRegistry
from mindspeed_llm.fsdp2.models.qwen3.qwen3 import (
    Qwen3ForCausalLM as MindSpeedQwen3ForCausalLM,
)
from mindspeed_llm.fsdp2.utils.logging import get_logger


logger = get_logger(__name__)


class Qwen3ForCausalLM(MindSpeedQwen3ForCausalLM):
    """Qwen3 AR model with a deterministic pretrained output-head restore."""

    # Transformers 5.x requires an explicit target -> source mapping.  A dict
    # remains compatible with the 4.x missing-key filtering path as well.
    _tied_weights_keys = {
        "lm_head.weight": "model.embed_tokens.weight",
    }

    # Keep compatibility with Transformers releases predating the generic
    # embedding accessors.
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        config = kwargs.get("config")
        if config is None:
            raise ValueError(
                "The isolated AR loader requires the resolved HuggingFace config."
            )

        # The Base checkpoint may omit lm_head.weight because it is serialized
        # through model.embed_tokens.weight.  Temporarily enable tying so that
        # from_pretrained resolves the omitted key instead of randomly
        # initializing an output head.
        keep_tied = bool(getattr(config, "tie_word_embeddings", True))
        config.tie_word_embeddings = True
        try:
            loaded = super().from_pretrained(*args, **kwargs)
        finally:
            config.tie_word_embeddings = keep_tied

        if isinstance(loaded, tuple):
            model, *loading_info = loaded
        else:
            model = loaded
            loading_info = None

        model.config.tie_word_embeddings = keep_tied
        input_weight = model.model.embed_tokens.weight

        if keep_tied:
            # Re-apply after loading in case the loader replaced a Parameter.
            model.tie_weights()
            restore_mode = "tied"
        else:
            # Preserve the local untied/FSDP layout, but initialize the output
            # head from the pretrained embedding rather than random weights.
            model.lm_head.weight = torch.nn.Parameter(
                input_weight.detach().clone(),
                requires_grad=input_weight.requires_grad,
            )
            restore_mode = "copied"

        if not torch.equal(model.lm_head.weight, model.model.embed_tokens.weight):
            raise RuntimeError(
                "AR lm_head restore failed: output and pretrained embedding "
                "weights differ before FSDP wrapping."
            )

        model._ar_lm_head_restored_from_pretrained = True
        logger.info_rank0(
            "> AR lm_head restored from pretrained model.embed_tokens.weight "
            f"(mode={restore_mode}, exact_match=True)."
        )

        if loading_info is not None:
            return (model, *loading_info)
        return model


def _register_ar_baseline_model() -> None:
    """Override qwen3 only inside this standalone AR training process."""
    ModelRegistry._REGISTRY["qwen3"] = Qwen3ForCausalLM


def _force_native_ar_pretraining(auto_trainer: AutoTrainer) -> None:
    backend = auto_trainer.trainer

    if backend.model_args.model_id != "qwen3":
        raise ValueError(
            "The AR CPT baseline must use `model_id: qwen3`; "
            f"got {backend.model_args.model_id!r}."
        )
    if backend.training_args.stage != "pt":
        raise ValueError(
            "The AR CPT baseline requires `training.stage: pt` so the "
            "Megatron pretraining labels are consumed without another shift."
        )
    if backend.data_args.data_manager_type != "mg":
        raise ValueError(
            "The AR CPT baseline expects `data.data_manager_type: mg`."
        )
    if backend.training_args.calculate_per_token_loss:
        raise ValueError(
            "Keep `calculate_per_token_loss: false` for the native AR path in "
            "the current Trainer implementation. The indexed CPT samples are "
            "fixed-length, so local-mean CE has the same token weighting."
        )

    # Trainer._compute_loss checks the parallel-engine config, not the source
    # HF config.  False makes it remove labels before model.forward() and then
    # compare every causal logit with the already-shifted Megatron label.
    backend.model.config.enable_diffusion_lm = False

    # Save an unambiguous native-AR config with exported HF checkpoints.  This
    # does not affect the architecture: model_id=qwen3 was already constructed.
    raw_model = backend.model.model
    raw_model.config.enable_diffusion_lm = False
    raw_model.config.diffusion_lm = False

    if not getattr(raw_model, "_ar_lm_head_restored_from_pretrained", False):
        raise RuntimeError(
            "AR lm_head was not restored from the pretrained embedding; "
            "refusing to train from a random output head."
        )

    if any("diffusion_head" in name for name, _ in raw_model.named_modules()):
        raise RuntimeError("A diffusion head was found in the requested AR baseline model.")

    logger.info_rank0(
        "> Native AR CPT guard enabled: pretrained lm_head, qwen3 causal "
        "model, pre-shifted labels, no diffusion branch."
    )


def main() -> None:
    _register_ar_baseline_model()
    auto_trainer = AutoTrainer()
    _force_native_ar_pretraining(auto_trainer)
    auto_trainer.train()


if __name__ == "__main__":
    main()
