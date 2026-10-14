"""Distributed-friendly validation metrics for the sampler."""

from __future__ import annotations

import torch


class BinaryAUCMeter:
    """Streaming histogram approximation of binary ROC AUC."""

    def __init__(self, bins: int, device: torch.device) -> None:
        if bins < 2:
            raise ValueError("bins must be at least two")
        self.bins = int(bins)
        # Float32 is used because Ascend NPU collectives do not universally
        # support float64.  Its integer precision is ample for per-run counts.
        self.positive = torch.zeros(bins, dtype=torch.float32, device=device)
        self.negative = torch.zeros(bins, dtype=torch.float32, device=device)

    @torch.no_grad()
    def update(self, probabilities: torch.Tensor, labels: torch.Tensor) -> None:
        probabilities = probabilities.detach().float().clamp_(0.0, 1.0)
        labels = labels.detach().bool()
        indices = torch.clamp(
            (probabilities * (self.bins - 1)).long(), 0, self.bins - 1
        )
        ones = torch.ones_like(probabilities, dtype=torch.float32)
        self.positive.scatter_add_(0, indices[labels], ones[labels])
        self.negative.scatter_add_(0, indices[~labels], ones[~labels])

    def synchronize(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.positive)
            torch.distributed.all_reduce(self.negative)

    def compute(self) -> float:
        positives = self.positive.sum()
        negatives = self.negative.sum()
        if positives.item() == 0 or negatives.item() == 0:
            return float("nan")
        cumulative_negatives = torch.cumsum(self.negative, dim=0)
        negatives_below = cumulative_negatives - self.negative
        favorable_pairs = (
            self.positive * (negatives_below + 0.5 * self.negative)
        ).sum()
        return float((favorable_pairs / (positives * negatives)).item())
