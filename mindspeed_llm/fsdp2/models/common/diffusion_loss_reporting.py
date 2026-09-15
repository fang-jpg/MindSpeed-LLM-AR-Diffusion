# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

"""Nemotron-style loss reporting for block-diffusion language models."""

from typing import Dict, Iterable, Optional

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


__all__ = ["build_diffusion_loss_report", "reduce_diffusion_loss_reports"]
