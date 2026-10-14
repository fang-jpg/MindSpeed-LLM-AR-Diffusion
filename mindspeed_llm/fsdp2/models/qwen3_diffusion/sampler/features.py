"""PCA semantic features and output-distribution statistics for the sampler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


PCA_FORMAT = "mindspeed-diffusion-sampler-pca-v1"
NUM_TOP_PREDICTIONS = 3
NUM_OUTPUT_STATISTICS = 6


def _torch_load(path: str | Path, map_location: Any = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


@dataclass
class EmbeddingPCA:
    """A fitted PCA projection for token embedding vectors."""

    mean: torch.Tensor
    components: torch.Tensor
    explained_variance: torch.Tensor
    source_rows: int
    seed: int
    metadata: dict[str, Any] | None = None

    @property
    def input_dim(self) -> int:
        return int(self.components.shape[1])

    @property
    def output_dim(self) -> int:
        return int(self.components.shape[0])

    def validate(self) -> None:
        if self.mean.ndim != 1:
            raise ValueError("PCA mean must be one-dimensional")
        if self.components.ndim != 2:
            raise ValueError("PCA components must be two-dimensional")
        if self.components.shape[1] != self.mean.shape[0]:
            raise ValueError("PCA mean and component dimensions do not match")
        if self.explained_variance.shape != (self.output_dim,):
            raise ValueError("invalid explained_variance shape")

    def project(self, embeddings: torch.Tensor) -> torch.Tensor:
        self.validate()
        mean = self.mean.to(device=embeddings.device, dtype=embeddings.dtype)
        components = self.components.to(
            device=embeddings.device, dtype=embeddings.dtype
        )
        return torch.matmul(embeddings - mean, components.transpose(0, 1))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": PCA_FORMAT,
                "mean": self.mean.detach().cpu(),
                "components": self.components.detach().cpu(),
                "explained_variance": self.explained_variance.detach().cpu(),
                "source_rows": self.source_rows,
                "seed": self.seed,
                "metadata": self.metadata or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingPCA":
        payload = _torch_load(path, map_location="cpu")
        if payload.get("format") != PCA_FORMAT:
            raise ValueError("unsupported PCA artifact format")
        result = cls(
            mean=payload["mean"].float(),
            components=payload["components"].float(),
            explained_variance=payload["explained_variance"].float(),
            source_rows=int(payload["source_rows"]),
            seed=int(payload["seed"]),
            metadata=dict(payload.get("metadata", {})),
        )
        result.validate()
        return result


def fit_embedding_pca(
    embedding_weight: torch.Tensor,
    output_dim: int = 46,
    sample_size: int = 32768,
    seed: int = 42,
) -> EmbeddingPCA:
    """Fit PCA to a deterministic sample of the frozen token embedding table."""
    if embedding_weight.ndim != 2:
        raise ValueError("embedding_weight must have shape [vocab, hidden]")
    rows, hidden_size = embedding_weight.shape
    if not 0 < output_dim <= hidden_size:
        raise ValueError("output_dim must be in (0, embedding hidden size]")
    sample_size = min(int(sample_size), rows)
    if sample_size <= output_dim:
        raise ValueError("sample_size must exceed output_dim")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(rows, generator=generator)[:sample_size]
    source_indices = indices.to(embedding_weight.device)
    sample = (
        embedding_weight.detach()
        .index_select(0, source_indices)
        .cpu()
        .float()
    )
    mean = sample.mean(dim=0)
    centered = sample - mean

    # A small oversampling margin stabilizes the randomized decomposition.
    q = min(output_dim + 8, min(centered.shape))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        _, singular_values, right_vectors = torch.pca_lowrank(
            centered,
            q=q,
            center=False,
            niter=4,
        )
    components = right_vectors[:, :output_dim].transpose(0, 1).contiguous()
    explained_variance = singular_values[:output_dim].square() / (sample_size - 1)
    return EmbeddingPCA(
        mean=mean,
        components=components,
        explained_variance=explained_variance,
        source_rows=sample_size,
        seed=seed,
        metadata={},
    )


@dataclass
class SamplerFeatureBatch:
    features: torch.Tensor
    top1_ids: torch.LongTensor
    top1_probability: torch.Tensor


class SamplerFeatureExtractor:
    """Build the report's semantic and distributional per-position features.

    The report fixes the total width at 144 but does not publish the individual
    PCA/statistic dimensions.  This implementation uses 46 PCA dimensions for
    each of the top-3 token embeddings and six explicit output statistics:
    top-1/2/3 probabilities, top-1 minus top-2 margin, top-3 mass, and entropy.
    This gives ``3 * 46 + 6 = 144`` exactly.
    """

    def __init__(
        self,
        pca: EmbeddingPCA,
        embedding_weight: torch.Tensor,
        invalid_token_id: int | None = None,
        table_dtype: torch.dtype = torch.float16,
        projection_batch_size: int = 8192,
    ) -> None:
        pca.validate()
        if pca.output_dim * NUM_TOP_PREDICTIONS + NUM_OUTPUT_STATISTICS != 144:
            raise ValueError(
                "the paper-aligned feature layout requires PCA output_dim=46"
            )
        if embedding_weight.shape[1] != pca.input_dim:
            raise ValueError("embedding width does not match PCA input dimension")
        if projection_batch_size <= 0:
            raise ValueError("projection_batch_size must be positive")
        projected_chunks = []
        for start in range(0, embedding_weight.shape[0], projection_batch_size):
            chunk = embedding_weight[start : start + projection_batch_size]
            projected_chunks.append(
                pca.project(chunk.detach().float()).to(dtype=table_dtype)
            )
        self.projected_embeddings = torch.cat(projected_chunks, dim=0)
        self.invalid_token_id = invalid_token_id

    @property
    def feature_dim(self) -> int:
        return (
            NUM_TOP_PREDICTIONS * self.projected_embeddings.shape[1]
            + NUM_OUTPUT_STATISTICS
        )

    def to(self, device: str | torch.device) -> "SamplerFeatureExtractor":
        self.projected_embeddings = self.projected_embeddings.to(device)
        return self

    @torch.no_grad()
    def __call__(self, logits: torch.Tensor) -> SamplerFeatureBatch:
        if logits.ndim != 3:
            raise ValueError("logits must have shape [batch, block, vocab]")
        if logits.shape[-1] != self.projected_embeddings.shape[0]:
            raise ValueError("logit vocabulary and projected embedding table differ")

        scores = logits.float()
        if (
            self.invalid_token_id is not None
            and 0 <= self.invalid_token_id < scores.shape[-1]
        ):
            scores = scores.clone()
            scores[..., self.invalid_token_id] = -torch.inf

        top_values, top_ids = torch.topk(scores, k=NUM_TOP_PREDICTIONS, dim=-1)
        log_normalizer = torch.logsumexp(scores, dim=-1, keepdim=True)
        top_probabilities = torch.exp(top_values - log_normalizer)

        probabilities = torch.softmax(scores, dim=-1)
        finite_scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
        expected_logit = (probabilities * finite_scores).sum(dim=-1)
        entropy = log_normalizer.squeeze(-1) - expected_logit

        margin = top_probabilities[..., 0] - top_probabilities[..., 1]
        top3_mass = top_probabilities.sum(dim=-1)
        statistics = torch.cat(
            [
                top_probabilities,
                margin.unsqueeze(-1),
                top3_mass.unsqueeze(-1),
                entropy.unsqueeze(-1),
            ],
            dim=-1,
        )

        table = self.projected_embeddings
        semantic = table[top_ids].float().flatten(start_dim=-2)
        features = torch.cat([semantic, statistics], dim=-1)
        if features.shape[-1] != 144:
            raise RuntimeError(f"expected 144 features, got {features.shape[-1]}")
        return SamplerFeatureBatch(
            features=features,
            top1_ids=top_ids[..., 0],
            top1_probability=top_probabilities[..., 0],
        )
