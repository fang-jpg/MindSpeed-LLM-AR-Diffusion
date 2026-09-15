# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

"""Nemotron-style loss reporting and global-token gradient normalization."""

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.distributed as dist


def _report_pair(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Build the ``[loss_sum, token_count]`` tensor used by Megatron logging."""
    numerator = numerator.detach().reshape(1).to(torch.float32)
    denominator = torch.as_tensor(denominator, device=numerator.device).detach().reshape(1).to(torch.float32)
    return torch.cat((numerator, denominator))


def build_diffusion_loss_report(
    weighted_loss_sum: torch.Tensor,
    total_token_count: torch.Tensor,
    ar_loss_sum: torch.Tensor,
    ar_token_count: torch.Tensor,
    dlm_loss_sum: torch.Tensor,
    dlm_token_count: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Build the same loss report emitted by NemotronLabsDiffusion.

    ``dlm_loss_sum`` is the unweighted diffusion loss after the ``1 / p_mask``
    correction. ``weighted_loss_sum`` already includes the configured DLM and
    AR objective weights.
    """
    return {
        "lm loss": _report_pair(weighted_loss_sum, total_token_count),
        "ar loss": _report_pair(ar_loss_sum, ar_token_count),
        "dlm loss": _report_pair(dlm_loss_sum, dlm_token_count),
        "num_tokens_dlm": torch.as_tensor(dlm_token_count, device=weighted_loss_sum.device)
        .detach()
        .reshape(1)
        .to(torch.float32),
    }


def reduce_diffusion_loss_reports(
    reports: Iterable[Dict[str, torch.Tensor]],
    group: Optional[dist.ProcessGroup] = None,
) -> Dict[str, torch.Tensor]:
    """Reduce reports across micro-batches and the data-parallel group.

    Two-element values are summed across micro-batches and DP ranks, then
    normalized by their token count. One-element values retain Megatron's
    legacy behavior and are averaged over local micro-batches only.
    """
    reports = list(reports)
    if not reports:
        return {}

    reduced = {}
    for key in reports[0]:
        values = [report[key].reshape(-1).to(torch.float32) for report in reports]
        if values[0].numel() == 2:
            value = torch.stack(values).sum(dim=0)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(value, op=dist.ReduceOp.SUM, group=group)
            reduced[key] = value[0] / value[1]
        elif values[0].numel() == 1:
            reduced[key] = torch.cat(values).mean()
        else:
            raise ValueError(f"Invalid loss report shape for {key}: {values[0].shape}")

    return reduced


@torch.no_grad()
def finalize_global_token_gradients(
    model: torch.nn.Module,
    local_token_counts: Iterable[torch.Tensor],
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize averaged FSDP gradients by the global token count.

    With per-token loss enabled, backward accumulates unnormalized loss sums
    over the local micro-batches. MindSpeed FSDP averages those gradients over
    its process group. Multiplying by ``group_size / global_token_count`` is
    therefore equivalent to Megatron's SUM reduction followed by division by
    the number of contributing tokens in the global batch.

    Returns:
        A pair containing the global token count and applied gradient scale.
    """
    token_counts = [torch.as_tensor(count).detach().reshape(()) for count in local_token_counts]
    if not token_counts:
        raise ValueError("Global token loss requires at least one local token count.")

    global_token_count = torch.stack(
        [count.to(device=token_counts[0].device, dtype=torch.float32) for count in token_counts]
    ).sum()
    group_size = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_token_count, op=dist.ReduceOp.SUM, group=group)
        group_size = dist.get_world_size(group)

    if global_token_count.item() > 0:
        gradient_scale = global_token_count.new_tensor(float(group_size)) / global_token_count
        scale_value = gradient_scale.item()
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale_value)
    else:
        gradient_scale = global_token_count.new_tensor(1.0)

    return global_token_count, gradient_scale


__all__ = [
    "build_diffusion_loss_report",
    "finalize_global_token_gradients",
    "reduce_diffusion_loss_reports",
]
