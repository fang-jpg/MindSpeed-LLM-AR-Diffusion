#!/usr/bin/env python3
"""Train the lightweight diffusion acceptance sampler on collected states."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from mindspeed_llm.fsdp2.models.qwen3_diffusion.sampler.data import (
    SamplerShardDataset,
)
from mindspeed_llm.fsdp2.models.qwen3_diffusion.sampler.features import EmbeddingPCA
from mindspeed_llm.fsdp2.models.qwen3_diffusion.sampler.metrics import BinaryAUCMeter
from mindspeed_llm.fsdp2.models.qwen3_diffusion.sampler.model import (
    DiffusionSampler,
    SamplerConfig,
)
from mindspeed_llm.fsdp2.models.qwen3_diffusion.sampler.runtime import (
    DTYPE_MAP,
    detect_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--pca-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--auc-bins", type=int, default=4096)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=DTYPE_MAP, default="bfloat16")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def _setup_distributed(args: argparse.Namespace):
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    device = torch.device(args.device or detect_device(local_rank))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.set_device(device)
    if world_size > 1:
        backend = args.backend
        if backend is None:
            backend = (
                "hccl"
                if device.type == "npu"
                else "nccl" if device.type == "cuda" else "gloo"
            )
        torch.distributed.init_process_group(backend=backend)
    return rank, world_size, device


def _autocast(device: torch.device, dtype: torch.dtype):
    if device.type == "cpu" or dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _set_learning_rate(optimizer, base_lr: float, step: int, warmup: int) -> None:
    scale = 1.0 if warmup <= 0 else min(1.0, (step + 1) / warmup)
    for group in optimizer.param_groups:
        group["lr"] = base_lr * scale


def _reduce_pair(values: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(values)
    return values


@torch.no_grad()
def validate(
    model,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    auc_bins: int,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    meter = BinaryAUCMeter(auc_bins, device)
    loss_and_count = torch.zeros(2, dtype=torch.float32, device=device)
    correct_and_count = torch.zeros(2, dtype=torch.float32, device=device)
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        features = batch["features"].to(device, non_blocking=True).float()
        labels = batch["labels"].to(device, non_blocking=True)
        active = batch["active_mask"].to(device, non_blocking=True)
        with _autocast(device, amp_dtype):
            logits = model(features, active)
        selected_logits = logits[active].float()
        selected_labels = labels[active]
        losses = F.binary_cross_entropy_with_logits(
            selected_logits,
            selected_labels.float(),
            reduction="sum",
        )
        probabilities = torch.sigmoid(selected_logits)
        meter.update(probabilities, selected_labels)
        loss_and_count += torch.tensor(
            [losses.item(), selected_labels.numel()],
            dtype=torch.float32,
            device=device,
        )
        predictions = probabilities >= 0.5
        correct_and_count += torch.tensor(
            [predictions.eq(selected_labels).sum().item(), selected_labels.numel()],
            dtype=torch.float32,
            device=device,
        )
    meter.synchronize()
    _reduce_pair(loss_and_count)
    _reduce_pair(correct_and_count)
    return {
        "loss": float((loss_and_count[0] / loss_and_count[1].clamp_min(1)).item()),
        "auc": meter.compute(),
        "accuracy_at_0.5": float(
            (correct_and_count[0] / correct_and_count[1].clamp_min(1)).item()
        ),
        "tokens": int(loss_and_count[1].item()),
    }


def _checkpoint_payload(
    model,
    optimizer,
    config: SamplerConfig,
    pca: EmbeddingPCA,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    validation: dict[str, float],
) -> dict:
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    return {
        "format": "mindspeed-diffusion-sampler-v1",
        "model_state": {
            key: value.detach().cpu() for key, value in unwrapped.state_dict().items()
        },
        "optimizer_state": optimizer.state_dict(),
        "sampler_config": asdict(config),
        "feature_spec": {
            "total_dim": 144,
            "top_predictions": 3,
            "pca_dim_per_prediction": pca.output_dim,
            "statistics": [
                "top1_probability",
                "top2_probability",
                "top3_probability",
                "top1_top2_margin",
                "top3_probability_mass",
                "entropy",
            ],
        },
        "pca": {
            "mean": pca.mean,
            "components": pca.components,
            "explained_variance": pca.explained_variance,
            "source_rows": pca.source_rows,
            "seed": pca.seed,
            "metadata": pca.metadata or {},
        },
        "train_args": vars(args),
        "epoch": epoch,
        "global_step": global_step,
        "validation": validation,
    }


def _save_checkpoint(payload: dict, destination: Path) -> None:
    # Direct writes work on Linux training hosts as well as Windows workspaces
    # where sandboxed directories may deny rename/replace operations.
    torch.save(payload, destination)


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if args.gradient_accumulation <= 0:
        raise ValueError("gradient-accumulation must be positive")
    rank, world_size, device = _setup_distributed(args)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    pca = EmbeddingPCA.load(args.pca_path)
    if pca.output_dim != 46:
        raise ValueError("paper-aligned 144-d features require --pca-dim 46")
    config = SamplerConfig(dropout=args.dropout)
    model = DiffusionSampler(config).to(device)
    if world_size > 1:
        device_ids = [device.index] if device.type in {"cuda", "npu"} else None
        model = DistributedDataParallel(model, device_ids=device_ids)
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    if rank == 0:
        print(
            f"> sampler parameters: {unwrapped.parameter_count():,} "
            f"({unwrapped.parameter_count() / 1e6:.3f}M)"
        )

    train_data = SamplerShardDataset(
        args.data_dir,
        split="train",
        max_block_length=config.max_block_length,
        shuffle=True,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
    )
    val_data = SamplerShardDataset(
        args.data_dir,
        split="val",
        max_block_length=config.max_block_length,
        shuffle=False,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type in {"cuda", "npu"},
    }
    train_loader = DataLoader(train_data, **loader_kwargs)
    val_loader = DataLoader(val_data, **loader_kwargs)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        torch.distributed.barrier()
    metrics_path = output_dir / "metrics.jsonl"
    global_step = 0
    best_auc = -math.inf
    stale_epochs = 0
    stop_training = False
    amp_dtype = DTYPE_MAP[args.dtype]

    for epoch in range(args.epochs):
        train_data.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_batches = 0
        micro_batches_since_step = 0
        join_context = (
            model.join if isinstance(model, DistributedDataParallel) else contextlib.nullcontext
        )
        with join_context():
            for batch_index, batch in enumerate(train_loader):
                features = batch["features"].to(device, non_blocking=True).float()
                labels = batch["labels"].to(device, non_blocking=True)
                active = batch["active_mask"].to(device, non_blocking=True)
                with _autocast(device, amp_dtype):
                    logits = model(features, active)
                    loss = F.binary_cross_entropy_with_logits(
                        logits[active], labels[active].float()
                    )
                    scaled_loss = loss / args.gradient_accumulation
                scaled_loss.backward()
                running_loss += loss.detach().item()
                running_batches += 1
                micro_batches_since_step += 1

                should_step = (
                    micro_batches_since_step >= args.gradient_accumulation
                )
                if should_step:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )
                    _set_learning_rate(
                        optimizer,
                        args.learning_rate,
                        global_step,
                        args.warmup_steps,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    micro_batches_since_step = 0
                    global_step += 1
                    if rank == 0 and args.log_every > 0:
                        if global_step % args.log_every == 0:
                            print(
                                f"> epoch={epoch} step={global_step} "
                                f"loss={running_loss / max(running_batches, 1):.6f}"
                            )
                            running_loss = 0.0
                            running_batches = 0
                    if args.max_steps is not None and global_step >= args.max_steps:
                        stop_training = True
                        break

            if micro_batches_since_step > 0 and not stop_training:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                _set_learning_rate(
                    optimizer,
                    args.learning_rate,
                    global_step,
                    args.warmup_steps,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        validation = validate(
            unwrapped,
            val_loader,
            device=device,
            amp_dtype=amp_dtype,
            auc_bins=args.auc_bins,
            max_batches=args.max_validation_batches,
        )
        validation.update(epoch=epoch, global_step=global_step)
        if rank == 0:
            print(json.dumps(validation, ensure_ascii=False))
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(validation, ensure_ascii=False) + "\n")
            payload = _checkpoint_payload(
                model,
                optimizer,
                config,
                pca,
                args,
                epoch,
                global_step,
                validation,
            )
            _save_checkpoint(payload, output_dir / "last.pt")
            if validation["auc"] > best_auc + args.min_delta:
                best_auc = validation["auc"]
                stale_epochs = 0
                _save_checkpoint(payload, output_dir / "best.pt")
            else:
                stale_epochs += 1
            stop_training = stop_training or stale_epochs >= args.patience

        if world_size > 1:
            stop_tensor = torch.tensor(
                int(stop_training), dtype=torch.int32, device=device
            )
            torch.distributed.broadcast(stop_tensor, src=0)
            stop_training = bool(stop_tensor.item())
        if stop_training:
            break

    if world_size > 1:
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
