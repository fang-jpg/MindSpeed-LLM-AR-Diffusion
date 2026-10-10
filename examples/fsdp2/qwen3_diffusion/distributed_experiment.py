#!/usr/bin/env python3
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Small torchrun helper for data-parallel AR/diffusion experiments."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist


def atomic_write_json(path: str | Path, value) -> None:
    """Publish JSON atomically so other ranks never observe a partial file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.pid{os.getpid()}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _native_collective(function):
    """Return the underlying torch collective below MindSpeed wrappers.

    MindSpeed registers high-availability wrappers around selected
    ``torch.distributed`` functions.  Those wrappers call Megatron ``get_args``
    and therefore cannot be used by these small standalone experiment entry
    points, which deliberately do not initialize Megatron.  The wrappers use
    ``functools.wraps``, so following ``__wrapped__`` reaches the original
    HCCL-backed torch function without changing the process group.
    """
    while hasattr(function, "__wrapped__"):
        function = function.__wrapped__
    return function


def _native_all_gather_object(output, value) -> None:
    """Run PyTorch object gathering without MindSpeed's nested wrapper.

    Unwrapping ``dist.all_gather_object`` alone is insufficient: PyTorch's
    Python implementation then looks up ``all_gather`` in its module globals,
    and MindSpeed patches that inner function separately.  Temporarily expose
    the unwrapped inner collective to this process-local call and restore it
    immediately afterwards.  Experiment entry points are single-threaded, so
    no other local collective can observe this short-lived substitution.
    """
    object_collective = _native_collective(dist.all_gather_object)
    function_globals = getattr(object_collective, "__globals__", None)
    wrapped_all_gather = (
        function_globals.get("all_gather") if function_globals else None
    )
    if wrapped_all_gather is None:
        object_collective(output, value)
        return

    function_globals["all_gather"] = _native_collective(wrapped_all_gather)
    try:
        object_collective(output, value)
    finally:
        function_globals["all_gather"] = wrapped_all_gather


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
            _native_collective(dist.barrier)()

    def all_gather_object(self, value):
        if not self.enabled:
            return [value]
        gathered = [None] * self.world_size
        _native_all_gather_object(gathered, value)
        return gathered

    def all_gather_json_files(
        self,
        value,
        exchange_dir: str | Path,
        timeout_seconds: float = 3600.0,
        poll_interval_seconds: float = 1.0,
    ):
        """Exchange JSON payloads through shared files instead of HCCL.

        This is intended for post-benchmark result collection, where ranks can
        finish far apart in time.  Avoiding a collective here prevents a fast
        rank from hitting an HCCL connection timeout while slower ranks are
        still decoding.
        """
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")

        directory = Path(exchange_dir)
        directory.mkdir(parents=True, exist_ok=True)
        rank_path = directory / f"rank{self.rank:02d}.json"
        atomic_write_json(
            rank_path,
            {
                "schema_version": 1,
                "rank": self.rank,
                "world_size": self.world_size,
                "value": value,
            },
        )

        expected_paths = [
            directory / f"rank{rank:02d}.json"
            for rank in range(self.world_size)
        ]
        started = time.monotonic()
        next_progress_report = 60.0
        while True:
            missing_ranks = [
                rank
                for rank, path in enumerate(expected_paths)
                if not path.is_file()
            ]
            if not missing_ranks:
                break

            elapsed = time.monotonic() - started
            if elapsed >= timeout_seconds:
                raise TimeoutError(
                    f"Timed out after {elapsed:.1f}s waiting for rank JSON "
                    f"files in {directory}; missing ranks: {missing_ranks}"
                )
            if self.is_main and elapsed >= next_progress_report:
                print(
                    f"> Waiting for benchmark rank files; elapsed={elapsed:.0f}s, "
                    f"missing_ranks={missing_ranks}",
                    flush=True,
                )
                next_progress_report += 60.0
            time.sleep(
                min(poll_interval_seconds, timeout_seconds - elapsed)
            )

        gathered = []
        for expected_rank, path in enumerate(expected_paths):
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if envelope.get("rank") != expected_rank:
                raise ValueError(
                    f"Rank payload identity mismatch in {path}: "
                    f"expected {expected_rank}, got {envelope.get('rank')}"
                )
            if envelope.get("world_size") != self.world_size:
                raise ValueError(
                    f"Rank payload world size mismatch in {path}: expected "
                    f"{self.world_size}, got {envelope.get('world_size')}"
                )
            gathered.append(envelope["value"])
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

