"""Write unreduced token losses without changing the training objective."""

import json
import logging
import math
from pathlib import Path

import torch


logger = logging.getLogger(__name__)


def _json_number(value):
    """Keep non-finite diagnostics explicit while emitting standard JSON."""
    value = float(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _cpu_values(tensor):
    # Transfer the complete tensor once, rather than synchronizing per token.
    return tensor.detach().cpu().tolist()


def _summarize_on_device(tensors):
    """Reduce per sample before transferring to CPU; never decode token IDs."""
    valid = tensors["valid_mask"].detach().bool()
    counts = valid.sum(dim=-1)
    names = ["losses"]
    if "weighted_losses" in tensors:
        names.append("weighted_losses")
    sums = torch.stack([
        torch.where(valid, tensors[name].detach().float(), 0.0).sum(dim=-1)
        for name in names
    ], dim=-1)
    cpu_counts = counts.cpu().tolist()
    cpu_sums = sums.cpu().tolist()
    summaries = []
    for count, totals in zip(cpu_counts, cpu_sums):
        summary = {
            "valid_token_count": count,
            "mean_loss": _json_number(totals[0] / count) if count else None,
        }
        if len(names) > 1:
            summary["mean_weighted_loss"] = _json_number(totals[1] / count) if count else None
        summaries.append(summary)
    return summaries


def write_token_loss_logs(
    details, input_ids, *, output_dir, step, micro_step, tokenizer=None, position_ids=None,
    include_token_details=False,
):
    """Append per-sample CE summaries, with optional full token diagnostics.

    ``details`` maps each component (for example ``ar`` or ``diffusion``) to
    ``losses``, ``target_ids`` and ``valid_mask`` tensors of shape [batch, length].
    Optional ``weighted_losses`` and ``p_mask`` tensors have the same shape.
    Inputs and targets must already be aligned by the caller, including any AR
    target shift. Invalid targets are retained with a null loss, never silently
    omitted or decoded. Position is zero-based within the local input sequence.
    Optional position_ids of shape [batch, length] or [1, length] additionally
    record the model's position_id. Step and micro_step are the caller's
    one-based training counters.

    Each rank appends to ``output_dir/token_losses/rank_<global_rank>.jsonl``.
    Rank zero also emits each record at INFO through the training logger. A
    per-batch/component summary reports the mean raw CE over valid positions;
    this local diagnostic need not equal a globally reduced training objective.
    No collectives are introduced and all inspected tensors are detached.

    By default only summaries are written and only device-reduced statistics
    are copied to CPU. Set include_token_details=True for short diagnostic runs
    to additionally write every token (including invalid positions).

    Returns the log Path, or None when there are no components to log.
    """
    if not details:
        return None
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, length]")
    expected_shape = input_ids.shape
    batch_size, sequence_length = expected_shape
    if position_ids is not None and (
        position_ids.ndim != 2
        or position_ids.shape[0] not in (1, batch_size)
        or position_ids.shape[1] != sequence_length
    ):
        raise ValueError("position_ids must have shape [batch, length] or [1, length]")
    for component, tensors in details.items():
        for name in ("losses", "target_ids", "valid_mask"):
            if name not in tensors:
                raise ValueError(f"{component} token losses are missing {name}")
        for name in ("losses", "target_ids", "valid_mask", "weighted_losses", "p_mask"):
            if name in tensors and tensors[name].shape != expected_shape:
                raise ValueError(
                    f"{component}.{name} shape {tuple(tensors[name].shape)} "
                    f"does not match input_ids shape {tuple(expected_shape)}"
                )

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    log_dir = Path(output_dir) / "token_losses"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"rank_{rank}.jsonl"
    inputs = _cpu_values(input_ids) if include_token_details else None
    positions = _cpu_values(position_ids) if include_token_details and position_ids is not None else None

    with log_path.open("a", encoding="utf-8") as log_file:
        def emit(record):
            line = json.dumps(record, ensure_ascii=False, allow_nan=False)
            log_file.write(line + "\n")
            if rank == 0:
                logger.info("token_loss %s", line)

        for component, tensors in details.items():
            if not include_token_details:
                for batch_index, statistics in enumerate(_summarize_on_device(tensors)):
                    emit({
                        "record_type": "summary",
                        "step": step,
                        "micro_step": micro_step,
                        "rank": rank,
                        "component": component,
                        "batch_index": batch_index,
                        "token_count": sequence_length,
                        **statistics,
                    })
                continue
            values = {
                name: _cpu_values(tensors[name])
                for name in ("losses", "target_ids", "valid_mask", "weighted_losses", "p_mask")
                if name in tensors
            }
            target_tokens = None
            if tokenizer is not None:
                valid_ids = [
                    values["target_ids"][batch_index][position]
                    for batch_index in range(batch_size)
                    for position in range(sequence_length)
                    if values["valid_mask"][batch_index][position]
                ]
                # A single batch conversion preserves duplicate IDs and order.
                target_tokens = iter(tokenizer.convert_ids_to_tokens(valid_ids)) if valid_ids else iter(())

            for batch_index in range(batch_size):
                common = {
                    "step": step,
                    "micro_step": micro_step,
                    "rank": rank,
                    "component": component,
                    "batch_index": batch_index,
                }
                valid_losses = []
                valid_weighted_losses = []
                for position in range(sequence_length):
                    valid = bool(values["valid_mask"][batch_index][position])
                    raw_loss = values["losses"][batch_index][position]
                    record = {
                        "record_type": "token",
                        **common,
                        "position": position,
                        "input_id": inputs[batch_index][position],
                        "target_id": values["target_ids"][batch_index][position],
                        "valid": valid,
                        "loss": _json_number(raw_loss) if valid else None,
                    }
                    if positions is not None:
                        position_batch = 0 if len(positions) == 1 else batch_index
                        record["position_id"] = positions[position_batch][position]
                    if valid:
                        valid_losses.append(raw_loss)
                    if "weighted_losses" in values:
                        weighted_loss = values["weighted_losses"][batch_index][position]
                        record["weighted_loss"] = _json_number(weighted_loss) if valid else None
                        if valid:
                            valid_weighted_losses.append(weighted_loss)
                    if "p_mask" in values:
                        record["p_mask"] = _json_number(values["p_mask"][batch_index][position])
                    if target_tokens is not None:
                        record["target_token"] = next(target_tokens) if valid else None
                    emit(record)

                summary = {
                    "record_type": "summary",
                    **common,
                    "token_count": sequence_length,
                    "valid_token_count": len(valid_losses),
                    "mean_loss": _json_number(sum(valid_losses) / len(valid_losses)) if valid_losses else None,
                }
                if "weighted_losses" in values:
                    summary["mean_weighted_loss"] = (
                        _json_number(sum(valid_weighted_losses) / len(valid_weighted_losses))
                        if valid_weighted_losses else None
                    )
                emit(summary)
        log_file.flush()
    return log_path
