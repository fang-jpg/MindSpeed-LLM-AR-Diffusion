# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Phase 1: Runtime toggle for diffusion_lm across all attention layers.
#
# Aligns with Nemotron _set_diffusion_lm (modeling_nemotron_labs_diffusion.py L416-419):
#   iterate over all decoder layers and flip self_attn.diffusion_lm.


def set_diffusion_lm(model, value: bool):
    """Set diffusion_lm flag on every attention layer in the model.

    When True: all layers use bidirectional attention (draft/diffusion mode).
    When False: all layers use causal attention (AR/verify mode).

    This is a no-op on layers whose self_attn does not have the
    diffusion_lm attribute, so it is safe to call on unmodified models.
    """
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "diffusion_lm"):
            layer.self_attn.diffusion_lm = value
