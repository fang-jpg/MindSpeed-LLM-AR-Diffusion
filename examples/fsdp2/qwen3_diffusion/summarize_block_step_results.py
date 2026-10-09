#!/usr/bin/env python3
"""Merge speed and GSM8K JSON outputs into a fill-ready Markdown table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CONFIGS = [
    ("AR-BASE", 1, 1),
    ("B04-S01", 4, 1),
    ("B04-S02", 4, 2),
    ("B04-S04", 4, 4),
    ("B08-S02", 8, 2),
    ("B08-S04", 8, 4),
    ("B08-S08", 8, 8),
    ("B16-S04", 16, 4),
    ("B16-S08", 16, 8),
    ("B16-S16", 16, 16),
    ("B32-S08", 32, 8),
    ("B32-S16", 32, 16),
    ("B32-S32", 32, 32),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def display(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main():
    args = parse_args()
    root = Path(args.results_root)
    output_dir = Path(args.output_dir) if args.output_dir else root
    ar_quality = read_json(root / "gsm8k" / "AR-BASE" / "summary.json")
    ar_accuracy = ar_quality.get("accuracy_percent") if ar_quality else None

    rows = []
    for exp_id, block_length, steps in CONFIGS:
        speed = read_json(root / "speed" / f"{exp_id}.json")
        quality = read_json(root / "gsm8k" / exp_id / "summary.json")
        accuracy = quality.get("accuracy_percent") if quality else None
        rows.append(
            {
                "exp_id": exp_id,
                "block_length": block_length,
                "steps_per_block": steps,
                "decode_tokens_per_second": (
                    speed.get("decode_tokens_per_second") if speed else None
                ),
                "speedup_vs_ar": speed.get("speedup_vs_ar") if speed else None,
                "mean_ms_per_prompt": (
                    speed.get("mean_ms_per_prompt") if speed else None
                ),
                "mean_total_nfe": speed.get("mean_total_nfe") if speed else None,
                "peak_memory_gib": speed.get("peak_memory_gib") if speed else None,
                "gsm8k_accuracy_percent": accuracy,
                "delta_gsm8k_percentage_points": (
                    accuracy - ar_accuracy
                    if accuracy is not None and ar_accuracy is not None
                    else None
                ),
                "formal_full_test": (
                    quality.get("formal_full_test") if quality else None
                ),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "experiment_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    headers = [
        "Exp ID",
        "B",
        "S",
        "Decode tok/s",
        "Speedup",
        "平均 ms",
        "平均 NFE",
        "GSM8K %",
        "Δ vs AR (pp)",
        "峰值显存 GiB",
        "完整 test",
    ]
    lines = [
        "# AR-Diffusion 实验结果汇总",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["exp_id"],
                    str(row["block_length"]),
                    str(row["steps_per_block"]),
                    display(row["decode_tokens_per_second"], 2),
                    display(row["speedup_vs_ar"], 3),
                    display(row["mean_ms_per_prompt"], 2),
                    display(row["mean_total_nfe"], 2),
                    display(row["gsm8k_accuracy_percent"], 2),
                    display(row["delta_gsm8k_percentage_points"], 2),
                    display(row["peak_memory_gib"], 2),
                    display(row["formal_full_test"], 0),
                ]
            )
            + " |"
        )
    markdown_path = output_dir / "experiment_summary.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"> Wrote {csv_path}")
    print(f"> Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
