#!/usr/bin/env python3
"""Plot MindSpeed-LLM loss and gradient-norm curves from a text log.

The parser supports the regular MindSpeed-LLM/FSDP2 metric line, for example::

    iteration 12/100 | lm loss: 2.31E+00 | grad norm: 1.23 |

It also understands the Qwen3 block-diffusion component prints::

    diff_loss:1234.5
    ar_loss:2345.6

Those component values are sums rather than token-normalized losses.  When an
optimizer step contains several component prints (gradient accumulation), the
plot uses their mean and the CSV contains both the mean and sum.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
ITERATION_RE = re.compile(
    r"\biteration\s*:?[ \t]*(?P<iteration>\d+)"
    r"(?:[ \t]*/[ \t]*(?P<total>\d+))?",
    re.IGNORECASE,
)


def _field_pattern(label: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?<![A-Za-z0-9]){label}[ \t]*:[ \t]*(?P<value>{FLOAT_PATTERN})",
        re.IGNORECASE,
    )


LM_LOSS_RE = _field_pattern(r"lm[ _-]+loss")
GRAD_NORM_RE = _field_pattern(r"grad(?:ient)?[ _-]+norm")
DIFF_LOSS_RE = _field_pattern(r"(?:diff|diffusion)[ _-]+loss")
AR_LOSS_RE = _field_pattern(r"(?:ar|autoregressive)[ _-]+loss")


@dataclass
class TrainingRecord:
    iteration: int
    total_iterations: Optional[int]
    run: int
    line_number: int
    lm_loss: Optional[float]
    grad_norm: Optional[float]
    diffusion_loss_print_mean: Optional[float]
    ar_loss_print_mean: Optional[float]
    diffusion_loss_print_sum: Optional[float]
    ar_loss_print_sum: Optional[float]
    diffusion_print_count: int
    ar_print_count: int


@dataclass
class ParseResult:
    records: list[TrainingRecord]
    unmatched_diffusion_values: int = 0
    unmatched_ar_values: int = 0


def _extract(pattern: re.Pattern[str], line: str) -> Optional[float]:
    match = pattern.search(line)
    return float(match.group("value")) if match else None


def _mean_or_none(values: list[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def _sum_or_none(values: list[float]) -> Optional[float]:
    return math.fsum(values) if values else None


def parse_lines(lines: Iterable[str]) -> ParseResult:
    """Parse iteration metrics and associate preceding component prints.

    Qwen3 block-diffusion prints component sums during each forward pass and
    emits the iteration metric after the optimizer step.  Pending component
    values are therefore attached to the next recognized iteration line.
    """

    records: list[TrainingRecord] = []
    pending_diffusion: list[float] = []
    pending_ar: list[float] = []
    run = 1
    previous_iteration: Optional[int] = None

    for line_number, raw_line in enumerate(lines, start=1):
        line = ANSI_ESCAPE_RE.sub("", raw_line)

        diffusion_loss = _extract(DIFF_LOSS_RE, line)
        ar_loss = _extract(AR_LOSS_RE, line)
        if diffusion_loss is not None:
            pending_diffusion.append(diffusion_loss)
        if ar_loss is not None:
            pending_ar.append(ar_loss)

        iteration_match = ITERATION_RE.search(line)
        if not iteration_match:
            continue

        lm_loss = _extract(LM_LOSS_RE, line)
        grad_norm = _extract(GRAD_NORM_RE, line)
        is_metric_line = (
            lm_loss is not None
            or grad_norm is not None
            or "consumed samples" in line.lower()
            or "elapsed time per iteration" in line.lower()
        )
        if not is_metric_line:
            continue

        iteration = int(iteration_match.group("iteration"))
        total_group = iteration_match.group("total")
        total_iterations = int(total_group) if total_group else None
        if previous_iteration is not None and iteration <= previous_iteration:
            run += 1

        records.append(
            TrainingRecord(
                iteration=iteration,
                total_iterations=total_iterations,
                run=run,
                line_number=line_number,
                lm_loss=lm_loss,
                grad_norm=grad_norm,
                diffusion_loss_print_mean=_mean_or_none(pending_diffusion),
                ar_loss_print_mean=_mean_or_none(pending_ar),
                diffusion_loss_print_sum=_sum_or_none(pending_diffusion),
                ar_loss_print_sum=_sum_or_none(pending_ar),
                diffusion_print_count=len(pending_diffusion),
                ar_print_count=len(pending_ar),
            )
        )
        pending_diffusion.clear()
        pending_ar.clear()
        previous_iteration = iteration

    return ParseResult(
        records=records,
        unmatched_diffusion_values=len(pending_diffusion),
        unmatched_ar_values=len(pending_ar),
    )


def parse_log(path: Path) -> ParseResult:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return parse_lines(handle)


def write_csv(records: list[TrainingRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(TrainingRecord.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: getattr(record, name) for name in fieldnames})


def _has_values(records: list[TrainingRecord], field: str) -> bool:
    return any(getattr(record, field) is not None for record in records)


def _rolling_mean(
    records: list[TrainingRecord], field: str, window: int
) -> list[Optional[float]]:
    smoothed: list[Optional[float]] = []
    history: list[float] = []
    current_run: Optional[int] = None
    for record in records:
        if current_run != record.run:
            history = []
            current_run = record.run
        value = getattr(record, field)
        if value is None:
            smoothed.append(None)
            continue
        history.append(value)
        smoothed.append(statistics.fmean(history[-window:]))
    return smoothed


def _plot_series(
    ax,
    records: list[TrainingRecord],
    field: str,
    label: str,
    color: str,
    smooth_window: int,
) -> None:
    """Plot each appended/restarted run separately to avoid backward joins."""

    smooth_values = _rolling_mean(records, field, smooth_window)
    run_ids = sorted({record.run for record in records})
    for index, run_id in enumerate(run_ids):
        points = [
            (record.iteration, getattr(record, field), smooth_values[position])
            for position, record in enumerate(records)
            if record.run == run_id and getattr(record, field) is not None
        ]
        if not points:
            continue
        x_values = [point[0] for point in points]
        raw_values = [point[1] for point in points]
        smoothed = [point[2] for point in points]
        raw_label = f"{label} (raw)" if index == 0 else None
        ax.plot(x_values, raw_values, color=color, alpha=0.22, linewidth=0.8, label=raw_label)
        if smooth_window > 1:
            smooth_label = f"{label} ({smooth_window}-step mean)" if index == 0 else None
            ax.plot(x_values, smoothed, color=color, linewidth=1.8, label=smooth_label)


def plot_records(
    records: list[TrainingRecord], output_path: Path, smooth_window: int, title: str
) -> None:
    try:
        import matplotlib
    except ImportError as error:
        raise SystemExit(
            "matplotlib is required to draw the curves; install it with "
            "`python -m pip install matplotlib`."
        ) from error

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels: list[tuple[str, list[tuple[str, str, str]]]] = []
    if _has_values(records, "lm_loss"):
        panels.append(("Normalized training loss", [("lm_loss", "LM/total loss", "#2563eb")]))

    component_series = []
    if _has_values(records, "diffusion_loss_print_mean"):
        component_series.append(
            ("diffusion_loss_print_mean", "Diffusion printed-sum mean", "#dc2626")
        )
    if _has_values(records, "ar_loss_print_mean"):
        component_series.append(("ar_loss_print_mean", "AR printed-sum mean", "#059669"))
    if component_series:
        panels.append(("Qwen3 loss components (not token-normalized)", component_series))

    if _has_values(records, "grad_norm"):
        panels.append(("Gradient norm", [("grad_norm", "Grad norm", "#7c3aed")]))

    if not panels:
        raise ValueError("No loss or grad-norm values were found in recognized iteration lines.")

    figure, axes = plt.subplots(
        len(panels),
        1,
        figsize=(13, 3.6 * len(panels)),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    figure.suptitle(title, fontsize=14, fontweight="bold")

    for ax, (panel_title, series) in zip(axes[:, 0], panels):
        for field, label, color in series:
            _plot_series(ax, records, field, label, color, smooth_window)
        ax.set_title(panel_title, loc="left", fontsize=11)
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.25, linewidth=0.7)
        ax.legend(loc="best", frameon=False)

    axes[-1, 0].set_xlabel("Optimizer iteration")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _count(records: list[TrainingRecord], field: str) -> int:
    return sum(getattr(record, field) is not None for record in records)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="MindSpeed-LLM text log")

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="PNG output path",
    )

    parser.add_argument(
        "--csv",
        type=Path,
        help="Parsed CSV path",
    )

    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write parsed CSV data",
    )

    parser.add_argument(
        "--smooth",
        type=int,
        default=20,
        help="Rolling-mean window; 1 disables smoothing",
    )

    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Only plot iterations <= this value",
    )

    parser.add_argument("--title")

    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    if args.smooth < 1:
        raise SystemExit("--smooth must be at least 1")
    if not args.log.is_file():
        raise SystemExit(f"Log file not found: {args.log}")

    output_path = args.output or args.log.with_name(f"{args.log.stem}_curves.png")
    csv_path = args.csv or output_path.with_suffix(".csv")
    result = parse_log(args.log)
    if args.max_step is not None:
        result.records = [
            record
            for record in result.records
            if record.iteration <= args.max_step
    ]
    if not result.records:
        raise SystemExit(
            "No MindSpeed-LLM iteration metric lines were found. Expected fields such as "
            "`iteration`, `lm loss`, and `grad norm`."
        )

    plot_records(result.records, output_path, args.smooth, args.title or args.log.name)
    if not args.no_csv:
        write_csv(result.records, csv_path)

    print(f"Parsed iteration records: {len(result.records)}")
    print(f"  lm loss: {_count(result.records, 'lm_loss')}")
    print(f"  grad norm: {_count(result.records, 'grad_norm')}")
    print(f"  diffusion component groups: {_count(result.records, 'diffusion_loss_print_mean')}")
    print(f"  AR component groups: {_count(result.records, 'ar_loss_print_mean')}")
    if result.unmatched_diffusion_values or result.unmatched_ar_values:
        print(
            "Ignored component prints after the final completed iteration: "
            f"diff={result.unmatched_diffusion_values}, ar={result.unmatched_ar_values}"
        )
    print(f"Wrote plot: {output_path.resolve()}")
    if not args.no_csv:
        print(f"Wrote CSV: {csv_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
