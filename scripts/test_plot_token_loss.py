"""Checks of raw-loss aggregation, log formats, duplicate handling and PNG/CSV output."""

import contextlib
import csv
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import plot_token_loss as plot


def summary(step=1, component="ar", mean=2.0, count=2, **kwargs):
    return dict(record_type="summary", step=step, component=component, mean_loss=mean,
                valid_token_count=count, token_count=4096, rank=0, micro_step=1,
                batch_index=0, mean_weighted_loss=99999.0, **kwargs)


def line(**kwargs):
    return json.dumps(summary(**kwargs))


class PlotTokenLossTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def write(self, name, records):
        path = self.root / name
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
        return path

    def test_unweighted_ce_pooled_across_microbatches_samples_and_ranks(self):
        records = [summary(mean=2, count=1), summary(mean=8, count=3),
                   summary(mean=4, count=2), summary(mean=10, count=4),
                   summary(component="diffusion", mean=6, count=1),
                   summary(component="diffusion", mean=2, count=3)]
        records[1]["batch_index"] = 1
        records[2]["micro_step"] = 2
        records[3]["rank"] = 1
        records[5]["rank"] = 1
        rows = plot.aggregate_steps(plot.parse_lines(map(json.dumps, records)).summaries)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].ar_loss, (2 + 24 + 8 + 40) / 10)
        self.assertEqual(rows[0].diffusion_loss, 3)
        self.assertEqual(rows[0].ar_valid_tokens, 10)
        self.assertEqual(rows[0].diffusion_valid_tokens, 4)
        self.assertEqual(rows[0].ar_ranks, "0;1")
        self.assertEqual(rows[0].ar_samples, 4)

    def test_plain_jsonl_and_prefixed_console_logs_match(self):
        record = line()
        console = "\x1b[32m[rank0]:INFO [2026-09-10 12:00:00] >> token_loss " + record + "\x1b[0m"
        parsed = plot.parse_lines(["ar_loss:999", "diff_loss:999", console])
        self.assertEqual(parsed.summaries, plot.parse_lines([record]).summaries)

    def test_token_lines_do_not_double_count_and_need_no_json_decoding(self):
        parsed = plot.parse_lines(iter([
            '{"record_type": "token", "loss": 5000, this line is deliberately incomplete', line(),
        ]))
        self.assertEqual(len(parsed.summaries), 1)
        self.assertEqual(parsed.malformed_lines, 0)

    def test_invalid_summary_is_reported_with_location(self):
        record = summary()
        record["valid_token_count"] = -3
        with self.assertRaisesRegex(ValueError, "run.log:1: valid_token_count"):
            plot.parse_lines([json.dumps(record)], "run.log")
        record["valid_token_count"] = 3
        record["mean_loss"] = None
        with self.assertRaisesRegex(ValueError, "mean_loss is required"):
            plot.parse_lines([json.dumps(record)])

    def test_truncated_summary_is_counted(self):
        parsed = plot.parse_lines([line(), '{"record_type": "summary", "step":'])
        self.assertEqual(parsed.malformed_lines, 1)
        self.assertEqual(len(parsed.summaries), 1)

    def test_empty_components_are_not_filled_with_zero(self):
        parsed = plot.parse_lines([line(step=3, count=0, mean=None),
                                   line(step=1, component="diffusion", mean=5, count=2)])
        rows = plot.aggregate_steps(parsed.summaries)
        self.assertEqual([row.step for row in rows], [1, 3])
        self.assertIsNone(rows[0].ar_loss)
        self.assertIsNone(rows[1].ar_loss)
        self.assertIsNone(rows[1].diffusion_loss)
        self.assertEqual(rows[1].ar_valid_tokens, 0)

    def test_nonfinite_sample_is_preserved_not_dropped(self):
        record = summary(mean="NaN", count=1)
        normal = summary(mean=2, count=3)
        normal["rank"] = 1
        rows = plot.aggregate_steps(plot.parse_lines(map(json.dumps, [record, normal])).summaries)
        self.assertTrue(math.isnan(rows[0].ar_loss))
        self.assertEqual(rows[0].ar_valid_tokens, 4)

    def test_copies_of_same_summary_are_deduplicated(self):
        path1 = self.write("rank_0.jsonl", [summary()])
        path2 = self.write("train.log", [summary()])
        summaries, bad, duplicate = plot.read_summaries([path1, path2])
        self.assertEqual((len(summaries), bad, duplicate), (1, 0, 1))
        self.assertEqual(plot.aggregate_steps(summaries)[0].ar_valid_tokens, 2)

    def test_conflicting_restarted_steps_fail(self):
        path = self.write("rank_0.jsonl", [summary(mean=2), summary(mean=3)])
        with self.assertRaisesRegex(ValueError, "Conflicting summaries"):
            plot.read_summaries([path])

    def test_rank_directory_and_rank_filter(self):
        first = self.write("rank_0.jsonl", [summary()])
        other = summary(mean=4)
        other["rank"] = 2
        second = self.write("rank_2.jsonl", [other])
        self.write("unrelated.jsonl", [summary()])
        paths = plot.resolve_inputs([self.root, first])
        self.assertEqual(paths, [first.resolve(), second.resolve()])
        selected, _, _ = plot.read_summaries(paths, {2})
        self.assertEqual(plot.aggregate_steps(selected)[0].ar_loss, 4)
        self.assertEqual(plot.aggregate_steps(selected)[0].ar_ranks, "2")

    def test_old_aggregate_logs_are_not_misinterpreted(self):
        path = self.root / "old.log"
        path.write_text("diff_loss:8192\nar_loss:16384\niteration 1 | lm loss: 3.0\n")
        with self.assertRaisesRegex(ValueError, "sums alone"):
            plot.read_summaries([path])

    def test_cli_produces_png_and_exact_step_csv(self):
        path = self.write("rank_0.jsonl", [summary(), summary(step=2, mean=3),
                                          summary(component="diffusion", mean=4)])
        png = self.root / "curves.png"
        with contextlib.redirect_stdout(io.StringIO()):
            result = plot.main([str(path), "--output", str(png)])
        self.assertEqual(result, 0)
        self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        with png.with_suffix(".csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual(float(rows[0]["ar_loss"]), 2)
        self.assertEqual(float(rows[0]["diffusion_loss"]), 4)
        self.assertEqual(rows[1]["diffusion_loss"], "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
