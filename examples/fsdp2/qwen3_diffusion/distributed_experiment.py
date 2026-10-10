#!/usr/bin/env python3
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Small torchrun helper for data-parallel AR/diffusion experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: str

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def all_gather_object(self, value):
        if not self.enabled:
            return [value]
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        return gathered

    def close(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


def initialize_distributed(requested_device: str | None = None) -> DistributedContext:
    """Initialize one replicated model process per NPU when launched by torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        return DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            device=requested_device or "",
        )

    if requested_device and requested_device not in {
        "npu",
        f"npu:{local_rank}",
    }:
        raise ValueError(
            "Under torchrun, --device must be omitted or match LOCAL_RANK "
            f"(expected npu:{local_rank}, got {requested_device})."
        )
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("8-NPU evaluation requires torch_npu") from exc

    torch.npu.set_device(f"npu:{local_rank}")
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=f"npu:{local_rank}",
    )


def shard_sequence(values, rank: int, world_size: int):
    """Deterministically assign every global item to exactly one rank."""
    return values[rank::world_size]

