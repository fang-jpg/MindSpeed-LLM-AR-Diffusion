#!/usr/bin/env python3
"""Plot per-step raw AR/Diffusion CE from token_loss summaries in .log or JSONL.

Accept a training log, rank JSONL files, or a token_losses directory from ONE
training run. Each step pools all selected samples, microbatches and ranks:
sum(mean_loss * valid_token_count) / sum(valid_token_count).
Objective weights and diffusion's 1/p_mask correction are never used.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import sys
from typing import Iterable


SUMMARY_RE = re.compile(r'"record_type"\s*:\s*"summary"')
PREFIX_RE = re.compile(r'\btoken_loss\s+(?=\{)')
ANSI_RE = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
COMPONENTS = ("ar", "diffusion")


@dataclass(frozen=True)
class Summary:
    step: int
    rank: int
    micro_step: int
    batch_index: int
    component: str
    valid_token_count: int
    mean_loss: float | None

    @property
    def key(self):
        return self.step, self.rank, self.micro_step, self.batch_index, self.component


@dataclass
class StepLoss:
    step: int
    ar_loss: float | None
    diffusion_loss: float | None
    ar_valid_tokens: int
    diffusion_valid_tokens: int
    ar_samples: int
    diffusion_samples: int
    ar_ranks: str
    diffusion_ranks: str


@dataclass
class ParseResult:
    summaries: list[Summary]
    malformed_lines: int = 0


def _integer(record, name, minimum):
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def parse_lines(lines: Iterable[str], source="<log>") -> ParseResult:
    """Read summaries only; token detail lines are neither decoded nor retained."""
    summaries = []
    malformed = 0
    for line_number, line in enumerate(lines, 1):
        if not SUMMARY_RE.search(line):
            continue
        line = ANSI_RE.sub("", line).strip()
        prefix = PREFIX_RE.search(line)
        payload = line[prefix.end():] if prefix else line
        if not payload.startswith("{"):
            continue
        try:
            record = json.loads(payload)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if not isinstance(record, dict) or record.get("record_type") != "summary":
            continue
        if record.get("component") not in COMPONENTS:
            continue
        try:
            count = _integer(record, "valid_token_count", 0)
            mean = record.get("mean_loss")
            if count > 0:
                if mean is None or isinstance(mean, bool):
                    raise ValueError("mean_loss is required for a nonempty summary")
                mean = float(mean)  # Also preserves explicit NaN/Infinity diagnostics.
            else:
                mean = None
            summaries.append(Summary(
                step=_integer(record, "step", 1),
                rank=_integer(record, "rank", 0),
                micro_step=_integer(record, "micro_step", 1),
                batch_index=_integer(record, "batch_index", 0),
                component=record["component"],
                valid_token_count=count,
                mean_loss=mean,
            ))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{source}:{line_number}: {error}") from error
    return ParseResult(summaries, malformed)


def resolve_inputs(inputs: list[Path]) -> list[Path]:
    paths = []
    for path in inputs:
        if path.is_dir():
            matches = sorted(path.glob("rank_*.jsonl"))
            if not matches:
                raise ValueError(f"No rank_*.jsonl files in {path}")
        elif path.is_file():
            matches = [path]
        else:
            raise ValueError(f"Log path does not exist: {path}")
        for match in matches:
            resolved = match.resolve()
            if resolved not in paths:
                paths.append(resolved)
    return paths


def _same_mean(left, right):
    return left == right or (
        left is not None and right is not None and math.isnan(left) and math.isnan(right)
    )


def read_summaries(paths: list[Path], ranks: set[int] | None = None):
    """Deduplicate console/JSONL copies; reject conflicting records from reruns."""
    unique = {}
    malformed = duplicates = 0
    for path in paths:
        with path.open(encoding="utf-8-sig", errors="replace") as handle:
            parsed = parse_lines(handle, str(path))
        malformed += parsed.malformed_lines
        for summary in parsed.summaries:
            if ranks is not None and summary.rank not in ranks:
                continue
            previous = unique.get(summary.key)
            if previous is not None:
                if (previous.valid_token_count != summary.valid_token_count
                        or not _same_mean(previous.mean_loss, summary.mean_loss)):
                    raise ValueError(
                        f"Conflicting summaries for {summary.key} in {path}. "
                        "Select logs from one training run; split appended restarted runs first."
                    )
                duplicates += 1
            unique[summary.key] = summary
    if not unique:
        raise ValueError(
            "No matching token_loss summaries found. Use logs produced with "
            "log_per_token_loss=True. Old ar_loss:/diff_loss: sums alone do not "
            "contain enough information to reconstruct raw per-token means."
        )
    return list(unique.values()), malformed, duplicates


def aggregate_steps(summaries: list[Summary]) -> list[StepLoss]:
    grouped = {}
    for summary in summaries:
        grouped.setdefault(summary.step, {}).setdefault(summary.component, []).append(summary)
    rows = []
    for step, components in sorted(grouped.items()):
        values = {}
        for component in COMPONENTS:
            samples = components.get(component, [])
            valid = [sample for sample in samples if sample.valid_token_count > 0]
            count = sum(sample.valid_token_count for sample in valid)
            contributions = [sample.mean_loss * sample.valid_token_count for sample in valid]
            # Retain a non-finite step as a plot gap; never silently drop bad samples.
            total = (math.fsum(contributions) if all(math.isfinite(x) for x in contributions)
                     else sum(contributions))
            values[f"{component}_loss"] = total / count if count else None
            values[f"{component}_valid_tokens"] = count
            values[f"{component}_samples"] = len(samples)
            values[f"{component}_ranks"] = ";".join(str(rank) for rank in sorted({s.rank for s in samples}))
        rows.append(StepLoss(step=step, **values))
    return rows


def write_csv(rows: list[StepLoss], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(StepLoss.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def plot_losses(rows: list[StepLoss], output: Path, title="AR and Diffusion loss"):
    try:
        import matplotlib
    except ImportError as error:
        raise ValueError("Plotting requires matplotlib: python -m pip install matplotlib") from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_ranks = sorted({int(rank) for row in rows for component in COMPONENTS
                        for rank in getattr(row, f"{component}_ranks").split(";") if rank})
    figure, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    for component, label, color in (("ar", "AR", "#087f8c"), ("diffusion", "Diffusion", "#df6b32")):
        values = [getattr(row, f"{component}_loss") for row in rows]
        if not any(value is not None for value in values):
            continue
        values = [value if value is not None and math.isfinite(value) else math.nan for value in values]
        ax.plot([row.step for row in rows], values, label=label, color=color,
                linewidth=1.5, marker=".", markersize=3)
    figure.suptitle(title, fontsize=15, fontweight="bold")
    ax.set_title(f"Raw CE, pooled by valid token count | logged ranks: {', '.join(map(str, all_ranks))}",
                 fontsize=10, color="#555555", pad=14)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Mean cross-entropy (nats / valid token)")
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    if ax.lines:
        ax.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="Training .log, rank JSONL files, or token_losses directory")
    parser.add_argument("-o", "--output", type=Path, help="Curve PNG (default: raw_token_loss.png beside input)")
    parser.add_argument("--csv", type=Path, help="Per-step CSV (default: same stem as the PNG)")
    parser.add_argument("--rank", type=int, action="append", help="Include only this global rank; may be repeated")
    parser.add_argument("--title", default="AR and Diffusion loss")
    args = parser.parse_args(argv)
    try:
        paths = resolve_inputs(args.logs)
        summaries, malformed, duplicates = read_summaries(paths, set(args.rank) if args.rank else None)
        rows = aggregate_steps(summaries)
        output = args.output or paths[0].parent / "raw_token_loss.png"
        csv_path = args.csv or output.with_suffix(".csv")
        if output.resolve() in paths or csv_path.resolve() in paths or output.resolve() == csv_path.resolve():
            raise ValueError("PNG and CSV output paths must differ from each other and the input logs.")
        plot_losses(rows, output, args.title)
        write_csv(rows, csv_path)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Parsed {len(summaries)} summaries across {len(rows)} logged optimizer steps.")
    print("Mean = sum(raw mean_loss * valid_token_count) / sum(valid_token_count). No objective weights or 1/p_mask.")
    for component in COMPONENTS:
        ranks = sorted({summary.rank for summary in summaries if summary.component == component})
        print(f"{component}: ranks {ranks}, {sum(getattr(row, component + '_valid_tokens') for row in rows)} valid tokens")
    if duplicates:
        print(f"Deduplicated {duplicates} identical summaries.")
    if malformed:
        print(f"Warning: skipped {malformed} malformed summary lines; affected steps may be incomplete.", file=sys.stderr)
    if any(getattr(row, component + '_loss') is not None
           and not math.isfinite(getattr(row, component + '_loss')) for row in rows for component in COMPONENTS):
        print("Warning: non-finite loss retained in CSV and shown as gaps in the plot.", file=sys.stderr)
    print(f"PNG: {output.resolve()}")
    print(f"CSV: {csv_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())