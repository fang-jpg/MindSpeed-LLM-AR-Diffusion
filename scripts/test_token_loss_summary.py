"""CPU regressions for compact loss summaries and explicit token-dump opt-in."""

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

import plot_token_loss


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summary_logger_under_test", ROOT / "mindspeed_llm/fsdp2/models/common/token_loss_logging.py"
)
logging_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(logging_module)


def read_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TokenLossSummaryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)
        self.inputs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        losses = torch.tensor([[1., 3., 99., 5.], [2., 88., 4., 66.]])
        valid = torch.tensor([[True, True, False, True], [True, False, True, False]])
        self.details = {
            "ar": {"losses": losses, "valid_mask": valid,
                   "target_ids": self.inputs.masked_fill(~valid, -100), "weighted_losses": losses * 0.7},
            "diffusion": {"losses": losses + 1, "valid_mask": valid,
                          "target_ids": self.inputs, "weighted_losses": (losses + 1) * 2,
                          "p_mask": torch.full((2, 4), 0.25)},
        }

    def write(self, **kwargs):
        args = dict(output_dir=self.output, step=1, micro_step=1)
        args.update(kwargs)
        return logging_module.write_token_loss_logs(self.details, self.inputs, **args)

    def test_default_writes_only_summaries(self):
        records = read_records(self.write())
        self.assertEqual(len(records), 4)
        self.assertTrue(all(record["record_type"] == "summary" for record in records))
        self.assertEqual([r["valid_token_count"] for r in records], [3, 2, 3, 2])
        self.assertEqual([r["mean_loss"] for r in records], [3, 3, 4, 4])

    def test_summary_never_copies_full_tokens_or_decodes_them(self):
        tokenizer = Mock()
        with patch.object(logging_module, "_cpu_values", side_effect=AssertionError("Full tensor copy")):
            self.write(tokenizer=tokenizer, position_ids=torch.arange(4).unsqueeze(0))
        tokenizer.convert_ids_to_tokens.assert_not_called()

    def test_summary_matches_explicit_full_dump(self):
        compact = read_records(self.write())
        full_path = self.write(output_dir=self.output / "full", include_token_details=True)
        full = read_records(full_path)
        self.assertEqual(len(full), 20)
        full_summaries = [r for r in full if r["record_type"] == "summary"]
        for expected, actual in zip(full_summaries, compact):
            self.assertEqual(expected.keys(), actual.keys())
            for name in expected:
                if name.startswith("mean_"):
                    self.assertAlmostEqual(expected[name], actual[name], places=6)
                else:
                    self.assertEqual(expected[name], actual[name])
        tokens = [r for r in full if r["record_type"] == "token"]
        self.assertEqual(len(tokens), 16)
        self.assertIsNone(tokens[2]["loss"])

    def test_invalid_nan_and_inf_do_not_pollute_summary(self):
        self.details["ar"]["losses"][~self.details["ar"]["valid_mask"]] = float("nan")
        self.details["ar"]["weighted_losses"][~self.details["ar"]["valid_mask"]] = float("inf")
        for row in read_records(self.write()):
            self.assertIsInstance(row["mean_loss"], float)
            self.assertTrue(math.isfinite(row["mean_weighted_loss"]))

    def test_nonfinite_valid_loss_stays_visible(self):
        self.details["ar"]["losses"][0, 0] = float("nan")
        self.details["ar"]["losses"][1, 0] = float("inf")
        rows = read_records(self.write())
        self.assertEqual(rows[0]["mean_loss"], "NaN")
        self.assertEqual(rows[1]["mean_loss"], "Infinity")

    def test_all_invalid_summary_has_null_mean(self):
        for tensors in self.details.values():
            tensors["valid_mask"].zero_()
        for row in read_records(self.write()):
            self.assertEqual(row["valid_token_count"], 0)
            self.assertIsNone(row["mean_loss"])
            self.assertIsNone(row["mean_weighted_loss"])

    def test_summary_does_not_change_autograd(self):
        leaf = torch.arange(8, dtype=torch.float32).reshape(2, 4).requires_grad_()
        losses = leaf.square()
        self.details["ar"]["losses"] = losses
        self.write()
        self.assertIsNone(leaf.grad)
        losses.sum().backward()
        torch.testing.assert_close(leaf.grad, 2 * leaf.detach())

    def test_per_rank_files_and_microbatch_append(self):
        with patch.object(torch.distributed, "is_initialized", return_value=True), \
                patch.object(torch.distributed, "get_rank", return_value=3), \
                patch.object(logging_module.logger, "info") as console:
            path = self.write()
            self.write(micro_step=2)
        self.assertEqual(path.name, "rank_3.jsonl")
        rows = read_records(path)
        self.assertEqual(len(rows), 8)
        self.assertEqual({r["micro_step"] for r in rows}, {1, 2})
        console.assert_not_called()

    def test_existing_plotter_accepts_compact_logs(self):
        path = self.write()
        self.write(micro_step=2)
        summaries, malformed, duplicates = plot_token_loss.read_summaries([path])
        rows = plot_token_loss.aggregate_steps(summaries)
        self.assertEqual((malformed, duplicates), (0, 0))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].ar_loss, 3)
        self.assertEqual(rows[0].diffusion_loss, 4)
        self.assertEqual(rows[0].ar_valid_tokens, 10)

    def test_summary_size_is_independent_of_sequence_length(self):
        self.inputs = torch.zeros((2, 4096), dtype=torch.long)
        self.details = {component: {
            "losses": torch.full((2, 4096), 2.0), "target_ids": self.inputs,
            "valid_mask": torch.ones_like(self.inputs, dtype=torch.bool),
            "weighted_losses": torch.full((2, 4096), 4.0),
        } for component in ("ar", "diffusion")}
        compact = self.write()
        with patch.object(logging_module.logger, "info"):
            full = self.write(output_dir=self.output / "full", include_token_details=True)
        self.assertEqual(len(read_records(compact)), 4)
        self.assertLess(compact.stat().st_size * 1000, full.stat().st_size)
        print(f"\n8192 input tokens, two branches: full={full.stat().st_size} bytes; "
              f"summary={compact.stat().st_size} bytes")


if __name__ == "__main__":
    unittest.main(verbosity=2)
