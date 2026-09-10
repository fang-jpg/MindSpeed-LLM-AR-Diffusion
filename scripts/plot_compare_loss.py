#!/usr/bin/env python3

"""
Compare loss curves from two MindSpeed-LLM training logs.

Example:

python plot_compare_loss.py \
    --log1 /path/to/train1.log \
    --log2 /path/to/train2.log \
    --label1 "Baseline" \
    --label2 "Diffusion" \
    --output loss_compare.png \
    --smooth 20
    --max-step 600
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
from pathlib import Path
from typing import Optional


FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"

ANSI_ESCAPE_RE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))"
)

ITERATION_RE = re.compile(
    r"\biteration\s*:?[ \t]*(?P<iteration>\d+)"
    r"(?:[ \t]*/[ \t]*(?P<total>\d+))?",
    re.IGNORECASE,
)

LM_LOSS_RE = re.compile(
    rf"(?<![A-Za-z0-9])lm[ _-]+loss[ \t]*:[ \t]*"
    rf"(?P<value>{FLOAT_PATTERN})",
    re.IGNORECASE,
)


def extract_loss(line: str) -> Optional[float]:
    match = LM_LOSS_RE.search(line)
    if match:
        return float(match.group("value"))
    return None


def parse_log(path: Path):
    """
    Extract:
        iteration -> lm loss

    from a MindSpeed training log.
    """

    records = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for line_number, raw_line in enumerate(f, start=1):

            line = ANSI_ESCAPE_RE.sub("", raw_line)

            iteration_match = ITERATION_RE.search(line)

            if not iteration_match:
                continue

            loss = extract_loss(line)

            if loss is None:
                continue

            iteration = int(
                iteration_match.group("iteration")
            )

            records.append(
                (
                    iteration,
                    loss,
                )
            )

    return records


def rolling_mean(values, window):
    """
    Calculate rolling mean.

    Example:

    values = [1, 2, 3, 4]
    window = 2

    result = [1, 1.5, 2.5, 3.5]
    """

    result = []

    history = []

    for value in values:

        history.append(value)

        current = history[-window:]

        result.append(
            statistics.fmean(current)
        )

    return result


def plot_compare(
    records1,
    records2,
    label1,
    label2,
    output_path,
    smooth_window=20,
    title="Training Loss Comparison",
    max_step=None,
):

    try:
        import matplotlib

    except ImportError as error:

        raise SystemExit(
            "matplotlib is required.\n"
            "Install it with:\n"
            "python -m pip install matplotlib"
        ) from error

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    # --------------------------------------------------
    # Limit maximum iteration
    # --------------------------------------------------

    if max_step is not None:

        records1 = [
            item
            for item in records1
            if item[0] <= max_step
        ]

        records2 = [
            item
            for item in records2
            if item[0] <= max_step
        ]

    if not records1:
        raise ValueError(
            "No loss values found in training log 1."
        )

    if not records2:
        raise ValueError(
            "No loss values found in training log 2."
        )

    # --------------------------------------------------
    # Separate x / y
    # --------------------------------------------------

    x1 = [item[0] for item in records1]
    y1 = [item[1] for item in records1]

    x2 = [item[0] for item in records2]
    y2 = [item[1] for item in records2]

    # --------------------------------------------------
    # Smooth
    # --------------------------------------------------

    if smooth_window > 1:

        smooth1 = rolling_mean(
            y1,
            smooth_window,
        )

        smooth2 = rolling_mean(
            y2,
            smooth_window,
        )

    else:

        smooth1 = y1
        smooth2 = y2

    # --------------------------------------------------
    # Create figure
    # --------------------------------------------------

    figure, ax = plt.subplots(
        figsize=(13, 7)
    )

    # --------------------------------------------------
    # Raw curves
    # --------------------------------------------------

    ax.plot(
        x1,
        y1,
        alpha=0.20,
        linewidth=0.8,
        label=f"{label1} (raw)",
    )

    ax.plot(
        x2,
        y2,
        alpha=0.20,
        linewidth=0.8,
        label=f"{label2} (raw)",
    )

    # --------------------------------------------------
    # Smoothed curves
    # --------------------------------------------------

    if smooth_window > 1:

        ax.plot(
            x1,
            smooth1,
            linewidth=2.0,
            label=f"{label1} ({smooth_window}-step mean)",
        )

        ax.plot(
            x2,
            smooth2,
            linewidth=2.0,
            label=f"{label2} ({smooth_window}-step mean)",
        )

    # --------------------------------------------------
    # Labels
    # --------------------------------------------------

    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Optimizer iteration",
        fontsize=12,
    )

    ax.set_ylabel(
        "Training loss",
        fontsize=12,
    )

    # --------------------------------------------------
    # Grid
    # --------------------------------------------------

    ax.grid(
        True,
        alpha=0.25,
        linewidth=0.7,
    )

    # --------------------------------------------------
    # Legend
    # --------------------------------------------------

    ax.legend(
        loc="best",
        frameon=False,
    )

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def build_argument_parser():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--log1",
        type=Path,
        required=True,
        help="First training log",
    )

    parser.add_argument(
        "--log2",
        type=Path,
        required=True,
        help="Second training log",
    )

    parser.add_argument(
        "--label1",
        default="Training 1",
        help="Label for first training",
    )

    parser.add_argument(
        "--label2",
        default="Training 2",
        help="Label for second training",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("loss_compare.png"),
        help="Output PNG path",
    )

    parser.add_argument(
        "--smooth",
        type=int,
        default=20,
        help="Rolling mean window",
    )

    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Only plot iterations <= this value",
    )

    parser.add_argument(
        "--title",
        default="Training Loss Comparison",
        help="Figure title",
    )

    return parser


def main():

    args = build_argument_parser().parse_args()

    if args.smooth < 1:

        raise SystemExit(
            "--smooth must be at least 1"
        )

    if not args.log1.is_file():

        raise SystemExit(
            f"Training log 1 not found: {args.log1}"
        )

    if not args.log2.is_file():

        raise SystemExit(
            f"Training log 2 not found: {args.log2}"
        )

    # --------------------------------------------------
    # Parse
    # --------------------------------------------------

    records1 = parse_log(args.log1)

    records2 = parse_log(args.log2)

    print(
        f"{args.label1}: "
        f"{len(records1)} loss points"
    )

    print(
        f"{args.label2}: "
        f"{len(records2)} loss points"
    )

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    plot_compare(
        records1=records1,
        records2=records2,
        label1=args.label1,
        label2=args.label2,
        output_path=args.output,
        smooth_window=args.smooth,
        title=args.title,
        max_step=args.max_step,
    )

    print(
        f"Wrote plot: "
        f"{args.output.resolve()}"
    )


if __name__ == "__main__":
    main()