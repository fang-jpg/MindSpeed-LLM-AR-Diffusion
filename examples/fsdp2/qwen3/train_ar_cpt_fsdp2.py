"""Isolated FSDP2 entrypoint for a native Qwen3 autoregressive CPT baseline.

The Megatron-style pretraining dataset used by this repository already returns
next-token labels (``input_ids = text[:-1]`` and ``labels = text[1:]``).  The
generic Trainer therefore has to compute CE directly from the returned logits;
passing those labels into a HuggingFace causal-LM loss would shift them twice.

This entrypoint deliberately changes only the newly-created training process.
It does not modify the source model config, the diffusion model, or the shared
``train_fsdp2.py`` entrypoint.
"""

from train_fsdp2 import AutoTrainer
from mindspeed_llm.fsdp2.utils.logging import get_logger


logger = get_logger(__name__)


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

    if any("diffusion_head" in name for name, _ in raw_model.named_modules()):
        raise RuntimeError("A diffusion head was found in the requested AR baseline model.")

    logger.info_rank0(
        "> Native AR CPT guard enabled: qwen3 causal model, pre-shifted labels, no diffusion branch."
    )


def main() -> None:
    auto_trainer = AutoTrainer()
    _force_native_ar_pretraining(auto_trainer)
    auto_trainer.train()


if __name__ == "__main__":
    main()
