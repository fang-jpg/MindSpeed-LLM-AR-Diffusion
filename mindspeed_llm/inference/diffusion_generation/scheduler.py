"""Scheduling and token selection for masked discrete diffusion."""

import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch


@dataclass
class SchedulerStep:
    """State passed to a diffusion model and used by the update rule."""

    index: int
    total_steps: int
    timestep: float


@dataclass
class DiffusionBlock:
    """A half-open generation span and its allocated denoising steps."""

    index: int
    start: int
    end: int
    num_steps: int


class MaskDiffusionScheduler:
    """Monotonically commits masked positions over a fixed number of steps.

    At every iteration the model predicts every still-masked position.  The
    scheduler commits the most confident predictions and leaves the remaining
    positions masked.  This is equivalent to low-confidence remasking without
    temporarily materializing predictions that are immediately masked again.
    """

    def __init__(
        self,
        num_steps: int,
        schedule: Literal["linear", "cosine"] = "linear",
        selection: Literal["low_confidence", "random"] = "low_confidence",
    ):
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if schedule not in ("linear", "cosine"):
            raise ValueError(f"Unsupported schedule: {schedule}")
        if selection not in ("low_confidence", "random"):
            raise ValueError(f"Unsupported selection strategy: {selection}")
        self.num_steps = num_steps
        self.schedule = schedule
        self.selection = selection

    def steps(self):
        for index in range(self.num_steps):
            # Models conventionally receive a noise level in [1, 0].
            timestep = 1.0 - index / max(self.num_steps - 1, 1)
            yield SchedulerStep(index=index, total_steps=self.num_steps, timestep=timestep)

    def _committed_fraction(self, completed_steps: int) -> float:
        progress = min(max(completed_steps / self.num_steps, 0.0), 1.0)
        if self.schedule == "cosine":
            return 1.0 - math.cos(progress * math.pi / 2.0)
        return progress

    def target_commit_count(self, initial_count: int, step_index: int) -> int:
        """Return cumulative committed tokens after ``step_index`` is run."""
        if initial_count <= 0:
            return 0
        fraction = self._committed_fraction(step_index + 1)
        target = math.ceil(initial_count * fraction)
        return min(max(target, 1), initial_count)

    def select_updates(
        self,
        confidence: torch.Tensor,
        mutable_mask: torch.Tensor,
        initial_mutable_counts: torch.Tensor,
        step_index: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Select positions to commit for each sequence in the batch."""
        if confidence.shape != mutable_mask.shape:
            raise ValueError("confidence and mutable_mask must have identical shapes")

        update_mask = torch.zeros_like(mutable_mask, dtype=torch.bool)
        for batch_index in range(mutable_mask.size(0)):
            candidates = torch.nonzero(mutable_mask[batch_index], as_tuple=False).flatten()
            if candidates.numel() == 0:
                continue

            initial_count = int(initial_mutable_counts[batch_index].item())
            already_committed = initial_count - int(candidates.numel())
            target = self.target_commit_count(initial_count, step_index)
            num_updates = min(max(target - already_committed, 1), int(candidates.numel()))

            if self.selection == "random":
                scores = torch.rand(candidates.numel(), device=confidence.device, generator=generator)
            else:
                scores = confidence[batch_index, candidates]
            selected = candidates[torch.topk(scores, k=num_updates, largest=True).indices]
            update_mask[batch_index, selected] = True
        return update_mask


def build_diffusion_blocks(generation_length: int, block_length: int, total_steps: int) -> list[DiffusionBlock]:
    """Split a generation region into blocks and distribute all steps.

    Earlier blocks receive one additional step when the budget is not evenly
    divisible.  Supporting a shorter final block is useful for evaluation
    lengths that are not exact multiples of the training block size.
    """
    if generation_length <= 0:
        raise ValueError("generation_length must be positive")
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    num_blocks = math.ceil(generation_length / block_length)
    if total_steps < num_blocks:
        raise ValueError("total_steps must allocate at least one step to every block")

    base_steps, remainder = divmod(total_steps, num_blocks)
    blocks = []
    for index in range(num_blocks):
        start = index * block_length
        end = min(start + block_length, generation_length)
        blocks.append(
            DiffusionBlock(
                index=index,
                start=start,
                end=end,
                num_steps=base_steps + (1 if index < remainder else 0),
            )
        )
    return blocks
