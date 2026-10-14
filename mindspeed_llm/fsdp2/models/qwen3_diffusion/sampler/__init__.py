"""Learned acceptance sampler for block-wise diffusion decoding.

The sampler is intentionally isolated from the existing generation path.  It
can be trained and evaluated without changing the diffusion backbone.
"""

from .features import EmbeddingPCA, SamplerFeatureExtractor
from .model import DiffusionSampler, SamplerConfig, load_sampler_checkpoint

__all__ = [
    "DiffusionSampler",
    "EmbeddingPCA",
    "SamplerConfig",
    "SamplerFeatureExtractor",
    "load_sampler_checkpoint",
]
