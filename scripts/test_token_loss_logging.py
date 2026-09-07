#!/usr/bin/env python3
"""CPU-only tests: python scripts/test_token_loss_logging.py."""

import importlib.util
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "mindspeed_llm/fsdp2/models/common/token_loss_logging.py"
SPEC = importlib.util.spec_from_file_location("token_loss_logging", MODULE_PATH)
token_loss_logging = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(token_loss_logging)
write_token_loss_logs = token_loss_logging.write_token_loss_logs


def read_records(path):
    def reject_non_json_constant(value):
        raise AssertionError(f"Nonstandard JSON constant: {value}")

    return [
        json.loads(line, parse_constant=reject_non_json_constant)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


class TokenLossLoggingTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.output_dir = Path(self.temp_directory.name)
        self.inputs = torch.tensor([[10, 20, 20, 30], [40, 50, 60, 70]])
        self.details = {
            "ar": {
                "losses": torch.tensor([[1.0, 2.0, 3.0, 99.0], [4.0, 99.0, 6.0, 99.0]]),
                "target_ids": torch.tensor([[20, 20, 30, -100], [50, -100, 70, -100]]),
                "valid_mask": torch.tensor([[True, True, True, False], [True, False, True, False]]),
            }
        }

    def write(self, **kwargs):
        defaults = {"output_dir": self.output_dir, "step": 1, "micro_step": 1}
        defaults.update(kwargs)
        return write_token_loss_logs(self.details, self.inputs, **defaults)

    def test_every_token_has_aligned_ids_and_masked_losses(self):
        records = read_records(self.write())
        tokens = [row for row in records if row["record_type"] == "token"]
        self.assertEqual(len(tokens), self.inputs.numel())
        self.assertEqual([(row["batch_index"], row["position"]) for row in tokens],
                         [(batch, position) for batch in range(2) for position in range(4)])
        self.assertEqual([row["input_id"] for row in tokens], [10, 20, 20, 30, 40, 50, 60, 70])
        self.assertEqual([row["target_id"] for row in tokens], [20, 20, 30, -100, 50, -100, 70, -100])
        self.assertEqual([row["loss"] for row in tokens], [1.0, 2.0, 3.0, None, 4.0, None, 6.0, None])
        self.assertEqual([row["valid"] for row in tokens], [True, True, True, False, True, False, True, False])
        summaries = [row for row in records if row["record_type"] == "summary"]
        self.assertEqual([row["mean_loss"] for row in summaries], [2.0, 5.0])
        self.assertEqual([row["valid_token_count"] for row in summaries], [3, 2])

    def test_rank_files_and_only_rank_zero_emits_info(self):
        with patch.object(torch.distributed, "is_initialized", return_value=True), \
                patch.object(torch.distributed, "get_rank", return_value=0), \
                self.assertLogs(token_loss_logging.logger, level=logging.INFO) as captured:
            rank_zero_path = self.write()
        self.assertEqual(rank_zero_path.name, "rank_0.jsonl")
        self.assertEqual(len(captured.records), 10)
        self.assertEqual(
            [json.loads(record.getMessage().removeprefix("token_loss ")) for record in captured.records],
            read_records(rank_zero_path),
        )
        with patch.object(torch.distributed, "is_initialized", return_value=True), \
                patch.object(torch.distributed, "get_rank", return_value=3), \
                patch.object(token_loss_logging.logger, "info") as log_info:
            rank_three_path = self.write()
        log_info.assert_not_called()
        self.assertEqual(rank_three_path.name, "rank_3.jsonl")
        self.assertTrue(all(row["rank"] == 3 for row in read_records(rank_three_path)))
        self.assertEqual(len(read_records(rank_zero_path)), 10)

    def test_appends_steps_and_microsteps(self):
        path = self.write(step=3, micro_step=1)
        self.write(step=3, micro_step=2)
        self.write(step=4, micro_step=1)
        records = read_records(path)
        self.assertEqual(len(records), 30)
        self.assertEqual([(records[index]["step"], records[index]["micro_step"]) for index in (0, 10, 20)],
                         [(3, 1), (3, 2), (4, 1)])

    def test_unicode_and_special_tokens_batch_conversion(self):
        tokenizer = Mock()
        tokenizer.convert_ids_to_tokens.return_value = ["你", "你", '<|end\"of\ntext|>', "\\", "世界"]
        path = self.write(tokenizer=tokenizer)
        tokenizer.convert_ids_to_tokens.assert_called_once_with([20, 20, 30, 50, 70])
        tokens = [row for row in read_records(path) if row["record_type"] == "token"]
        self.assertEqual([row["target_token"] for row in tokens],
                         ["你", "你", '<|end\"of\ntext|>', None, "\\", None, "世界", None])
        self.assertIn("世界", path.read_text(encoding="utf-8"))
        self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 10)

    def test_detach_does_not_modify_values_or_gradients(self):
        leaf = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], requires_grad=True)
        losses = leaf.square()
        self.details["ar"]["losses"] = losses
        saved = losses.detach().clone()
        self.write()
        self.assertTrue(torch.equal(losses.detach(), saved))
        self.assertTrue(losses.requires_grad)
        self.assertIsNone(leaf.grad)
        losses.sum().backward()
        self.assertTrue(torch.equal(leaf.grad, 2 * leaf.detach()))

    def test_empty_details_does_not_create_files(self):
        result = write_token_loss_logs({}, self.inputs, output_dir=self.output_dir, step=1, micro_step=1)
        self.assertIsNone(result)
        self.assertFalse((self.output_dir / "token_losses").exists())

    def test_weighted_diffusion_loss_keeps_raw_loss_distinct(self):
        self.details["diffusion"] = {
            "losses": self.details["ar"]["losses"] + 1,
            "target_ids": self.inputs,
            "valid_mask": self.details["ar"]["valid_mask"],
            "weighted_losses": (self.details["ar"]["losses"] + 1) / 0.5,
            "p_mask": torch.full_like(self.inputs, 0.5, dtype=torch.float32),
        }
        records = read_records(self.write())
        self.assertEqual(len(records), 20)
        tokens = [row for row in records if row["record_type"] == "token" and row["component"] == "diffusion"]
        self.assertEqual(tokens[0]["loss"], 2.0)
        self.assertEqual(tokens[0]["weighted_loss"], 4.0)
        self.assertEqual(tokens[0]["p_mask"], 0.5)
        self.assertIsNone(tokens[3]["loss"])
        self.assertIsNone(tokens[3]["weighted_loss"])
        summaries = [row for row in records if row["record_type"] == "summary" and row["component"] == "diffusion"]
        self.assertEqual(summaries[0]["mean_loss"], 3.0)
        self.assertEqual(summaries[0]["mean_weighted_loss"], 6.0)

    def test_nonfinite_values_produce_standard_json(self):
        self.details["ar"]["losses"][0] = torch.tensor([float("nan"), float("inf"), -float("inf"), float("nan")])
        records = read_records(self.write())
        tokens = [row for row in records if row["record_type"] == "token"]
        self.assertEqual([row["loss"] for row in tokens[:4]], ["NaN", "Infinity", "-Infinity", None])
        summary = next(row for row in records if row["record_type"] == "summary")
        self.assertEqual(summary["mean_loss"], "NaN")

    def test_all_invalid_tokens_are_present_and_not_decoded(self):
        self.details["ar"]["valid_mask"] = torch.zeros_like(self.inputs, dtype=torch.bool)
        tokenizer = Mock()
        records = read_records(self.write(tokenizer=tokenizer))
        self.assertEqual(len(records), 10)
        tokenizer.convert_ids_to_tokens.assert_not_called()
        for row in records:
            if row["record_type"] == "summary":
                self.assertEqual(row["valid_token_count"], 0)
                self.assertIsNone(row["mean_loss"])
            else:
                self.assertIsNone(row["loss"])
                self.assertIsNone(row["target_token"])

    def test_misaligned_shapes_fail_before_writing(self):
        self.details["ar"]["target_ids"] = torch.ones((2, 3), dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "does not match input_ids"):
            self.write()
        self.assertFalse((self.output_dir / "token_losses").exists())

    def test_position_ids_preserve_original_offsets_and_broadcast(self):
        path = self.write(position_ids=torch.tensor([[12, 13, 0, 1]]))
        tokens = [row for row in read_records(path) if row["record_type"] == "token"]
        self.assertEqual([row["position_id"] for row in tokens], [12, 13, 0, 1, 12, 13, 0, 1])
        self.assertEqual([row["position"] for row in tokens], [0, 1, 2, 3, 0, 1, 2, 3])
        self.write(micro_step=2, position_ids=torch.tensor([[8, 9, 10, 11], [0, 1, 0, 1]]))
        tokens = [row for row in read_records(path) if row["record_type"] == "token" and row["micro_step"] == 2]
        self.assertEqual([row["position_id"] for row in tokens], [8, 9, 10, 11, 0, 1, 0, 1])

    def test_misaligned_position_ids_fail_before_writing(self):
        with self.assertRaisesRegex(ValueError, "position_ids must have shape"):
            self.write(position_ids=torch.tensor([[0, 1, 2]]))
        self.assertFalse((self.output_dir / "token_losses").exists())

    def test_long_sequences_are_not_truncated(self):
        length = 4096
        inputs = torch.zeros((2, length), dtype=torch.long)
        details = {"ar": {
            "target_ids": inputs,
            "losses": torch.arange(2 * length, dtype=torch.float32).reshape(2, length),
            "valid_mask": torch.ones_like(inputs, dtype=torch.bool),
        }}
        with patch.object(token_loss_logging.logger, "info"):
            path = write_token_loss_logs(
                details, inputs, output_dir=self.output_dir, step=1, micro_step=1
            )
        tokens = [row for row in read_records(path) if row["record_type"] == "token"]
        self.assertEqual(len(tokens), 2 * length)
        self.assertEqual(tokens[-1]["position"], length - 1)
        self.assertEqual(tokens[-1]["batch_index"], 1)
        self.assertEqual(tokens[-1]["loss"], 2 * length - 1)


if __name__ == "__main__":
    unittest.main()
