"""Sharded tensor storage for diffusion sampler trajectory states."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info


SHARD_FORMAT = "mindspeed-diffusion-sampler-shard-v1"
POLICY_IDS = {"confidence": 0, "hybrid": 1}


def _torch_load(path: str | Path, map_location: Any = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


@dataclass
class SamplerTrainingRecord:
    features: torch.Tensor
    labels: torch.BoolTensor
    active_mask: torch.BoolTensor
    policy: str
    sample_index: int
    block_index: int
    step_index: int

    def validate(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("features must have shape [block, feature]")
        if self.labels.shape != self.features.shape[:1]:
            raise ValueError("labels must have shape [block]")
        if self.active_mask.shape != self.labels.shape:
            raise ValueError("active_mask must have shape [block]")
        if self.features.shape[-1] != 144:
            raise ValueError("sampler records must contain 144 features")
        if not self.active_mask.any():
            raise ValueError("a trajectory state must have an active position")
        if self.policy not in POLICY_IDS:
            raise ValueError(f"unknown trajectory policy: {self.policy}")


class SamplerShardWriter:
    """Write fixed-block records in moderately sized, independently loadable shards."""

    def __init__(
        self,
        output_dir: str | Path,
        rank: int = 0,
        records_per_shard: int = 8192,
    ) -> None:
        if records_per_shard <= 0:
            raise ValueError("records_per_shard must be positive")
        self.output_dir = Path(output_dir)
        self.rank = int(rank)
        self.records_per_shard = int(records_per_shard)
        self.buffers: dict[tuple[str, int], list[SamplerTrainingRecord]] = defaultdict(list)
        self.shard_counts: dict[tuple[str, int], int] = defaultdict(int)
        self.record_counts: dict[tuple[str, int], int] = defaultdict(int)
        self.closed = False

    def add(self, split: str, record: SamplerTrainingRecord) -> None:
        if self.closed:
            raise RuntimeError("cannot add to a closed writer")
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        record.validate()
        block_length = int(record.features.shape[0])
        key = (split, block_length)
        self.buffers[key].append(record)
        self.record_counts[key] += 1
        if len(self.buffers[key]) >= self.records_per_shard:
            self._flush(key)

    def _flush(self, key: tuple[str, int]) -> None:
        records = self.buffers[key]
        if not records:
            return
        split, block_length = key
        shard_index = self.shard_counts[key]
        directory = (
            self.output_dir
            / split
            / f"rank-{self.rank:05d}"
            / f"block-{block_length:03d}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"shard-{shard_index:06d}.pt"
        payload = {
            "format": SHARD_FORMAT,
            "block_length": block_length,
            "feature_dim": 144,
            "features": torch.stack([item.features for item in records])
            .to(torch.float16)
            .contiguous(),
            "labels": torch.stack([item.labels for item in records])
            .to(torch.bool)
            .contiguous(),
            "active_mask": torch.stack([item.active_mask for item in records])
            .to(torch.bool)
            .contiguous(),
            "policy": torch.tensor(
                [POLICY_IDS[item.policy] for item in records], dtype=torch.int8
            ),
            "sample_index": torch.tensor(
                [item.sample_index for item in records], dtype=torch.int64
            ),
            "block_index": torch.tensor(
                [item.block_index for item in records], dtype=torch.int32
            ),
            "step_index": torch.tensor(
                [item.step_index for item in records], dtype=torch.int16
            ),
        }
        torch.save(payload, path)
        self.buffers[key] = []
        self.shard_counts[key] += 1

    def close(self) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("writer is already closed")
        for key in list(self.buffers):
            self._flush(key)
        self.closed = True
        summary = {
            "format": SHARD_FORMAT,
            "rank": self.rank,
            "records_per_shard": self.records_per_shard,
            "partitions": [
                {
                    "split": split,
                    "block_length": block_length,
                    "records": self.record_counts[(split, block_length)],
                    "shards": self.shard_counts[(split, block_length)],
                }
                for split, block_length in sorted(self.record_counts)
            ],
        }
        manifest_dir = self.output_dir / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"rank-{self.rank:05d}.json"
        manifest_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary

    def __enter__(self) -> "SamplerShardWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None and not self.closed:
            self.close()


class SamplerShardDataset(IterableDataset):
    """Stream sampler records, sharding files across ranks and data workers."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        max_block_length: int = 32,
        shuffle: bool = True,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_block_length = int(max_block_length)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0

        self.shards = sorted((self.data_dir / split).glob("**/shard-*.pt"))
        if not self.shards:
            raise FileNotFoundError(
                f"no sampler shards found under {self.data_dir / split}"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _assigned_shards(self) -> list[Path]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        global_worker = self.rank * num_workers + worker_id
        global_workers = self.world_size * num_workers
        shards = list(self.shards)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(shards)
        return shards[global_worker::global_workers]

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch * 1009 + self.rank)
        for path in self._assigned_shards():
            payload = _torch_load(path, map_location="cpu")
            if payload.get("format") != SHARD_FORMAT:
                raise ValueError(f"unsupported sampler shard: {path}")
            features = payload["features"]
            labels = payload["labels"]
            active_mask = payload["active_mask"]
            block_length = int(payload["block_length"])
            if block_length > self.max_block_length:
                raise ValueError(
                    f"shard block length {block_length} exceeds "
                    f"max_block_length={self.max_block_length}"
                )
            order = (
                torch.randperm(features.shape[0], generator=generator)
                if self.shuffle
                else torch.arange(features.shape[0])
            )
            for index in order.tolist():
                padded_features = torch.zeros(
                    self.max_block_length,
                    features.shape[-1],
                    dtype=features.dtype,
                )
                padded_labels = torch.zeros(
                    self.max_block_length, dtype=torch.bool
                )
                padded_active = torch.zeros(
                    self.max_block_length, dtype=torch.bool
                )
                padded_features[:block_length] = features[index]
                padded_labels[:block_length] = labels[index]
                padded_active[:block_length] = active_mask[index]
                yield {
                    "features": padded_features,
                    "labels": padded_labels,
                    "active_mask": padded_active,
                }
