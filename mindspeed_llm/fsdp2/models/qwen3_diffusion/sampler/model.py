"""The lightweight Transformer acceptance sampler described in the report."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class SamplerConfig:
    """Architecture defaults chosen to match the reported 4.8M sampler.

    The report specifies four layers, width 384, a 144-dimensional input and
    approximately 4.8M parameters.  It does not disclose the head count or FFN
    width.  Eight heads and a 2x FFN reproduce the reported parameter count and
    remain configurable for ablations.
    """

    feature_dim: int = 144
    hidden_size: int = 384
    num_layers: int = 4
    num_attention_heads: int = 8
    intermediate_size: int = 768
    max_block_length: int = 32
    dropout: float = 0.1
    activation: str = "gelu"
    norm_first: bool = True

    def validate(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_layers <= 0 or self.intermediate_size <= 0:
            raise ValueError("num_layers and intermediate_size must be positive")
        if self.max_block_length <= 0:
            raise ValueError("max_block_length must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class DiffusionSampler(nn.Module):
    """Predict whether each masked position's current top-1 is final.

    ``active_mask`` is true only at positions that are still masked in the
    diffusion block.  Inactive positions neither provide attention keys nor
    contribute to the BCE loss.
    """

    def __init__(self, config: SamplerConfig | None = None) -> None:
        super().__init__()
        self.config = config or SamplerConfig()
        self.config.validate()

        self.input_norm = nn.LayerNorm(self.config.feature_dim)
        self.input_projection = nn.Linear(
            self.config.feature_dim, self.config.hidden_size
        )
        self.position_embedding = nn.Embedding(
            self.config.max_block_length, self.config.hidden_size
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_size,
            nhead=self.config.num_attention_heads,
            dim_feedforward=self.config.intermediate_size,
            dropout=self.config.dropout,
            activation=self.config.activation,
            batch_first=True,
            norm_first=self.config.norm_first,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_layers,
            norm=nn.LayerNorm(self.config.hidden_size),
            enable_nested_tensor=False,
        )
        self.output_head = nn.Linear(self.config.hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.input_norm.reset_parameters()
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        # TransformerEncoder deep-copies one prototype layer, which otherwise
        # leaves every layer with identical initial values.
        for layer in self.encoder.layers:
            layer.self_attn._reset_parameters()
            layer.linear1.reset_parameters()
            layer.linear2.reset_parameters()
            layer.norm1.reset_parameters()
            layer.norm2.reset_parameters()
        if self.encoder.norm is not None:
            self.encoder.norm.reset_parameters()
        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(
        self,
        features: torch.Tensor,
        active_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Return one unnormalized correctness logit per block position."""
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, block, feature]")
        if active_mask.shape != features.shape[:2]:
            raise ValueError("active_mask must match features[:2]")
        if features.shape[-1] != self.config.feature_dim:
            raise ValueError(
                f"expected feature_dim={self.config.feature_dim}, "
                f"got {features.shape[-1]}"
            )
        block_length = features.shape[1]
        if block_length > self.config.max_block_length:
            raise ValueError(
                f"block length {block_length} exceeds configured maximum "
                f"{self.config.max_block_length}"
            )
        if not active_mask.any(dim=1).all():
            raise ValueError("every sample must contain at least one active position")

        positions = torch.arange(block_length, device=features.device)
        hidden = self.input_projection(self.input_norm(features))
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        hidden = hidden.masked_fill(~active_mask.unsqueeze(-1), 0.0)
        hidden = self.encoder(hidden, src_key_padding_mask=~active_mask)
        return self.output_head(hidden).squeeze(-1)

    @torch.no_grad()
    def predict_proba(
        self,
        features: torch.Tensor,
        active_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        probabilities = torch.sigmoid(self(features, active_mask))
        return probabilities.masked_fill(~active_mask, 0.0)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _torch_load(path: str | Path, map_location: Any = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_sampler_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[DiffusionSampler, dict[str, Any]]:
    """Load a sampler checkpoint produced by ``train_sampler.py``."""
    payload = _torch_load(checkpoint_path, map_location="cpu")
    if payload.get("format") != "mindspeed-diffusion-sampler-v1":
        raise ValueError("unsupported sampler checkpoint format")
    config = SamplerConfig(**payload["sampler_config"])
    model = DiffusionSampler(config)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"model_state", "optimizer_state"}
    }
    return model, metadata


def sampler_config_dict(config: SamplerConfig) -> dict[str, Any]:
    return asdict(config)
