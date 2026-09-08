"""CPU tests of the real Trainer loss/logging methods with hardware imports isolated."""

import importlib.util
import logging
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def load_trainer():
    dependencies = {
        "distributed.parallel_state": ["ParallelState"],
        "distributed.clip_grad_norm": ["clip_grad_norm"],
        "data.data_factory": ["DataManager"],
        "data.processor.processor_utils": ["IGNORE_INDEX"],
        "features.chunkloss": ["chunk_loss", "calculate_lm_loss"],
        "models.common.token_loss_logging": ["write_token_loss_logs"],
        "checkpoint.utils": ["empty_cache", "cleanup_old_checkpoints"],
        "utils.dist_op": ["all_reduce"],
        "utils.logging": ["get_logger"],
        "utils.train_monitor": ["TrainMonitor"],
        "utils.profiler": ["ProfilerConfig", "ProfilerManager"],
    }
    stubs = {}
    for suffix, names in dependencies.items():
        name = "mindspeed_llm.fsdp2." + suffix
        module = types.ModuleType(name)
        for attribute in names:
            setattr(module, attribute, Mock())
        stubs[name] = module
    stubs["mindspeed_llm.fsdp2.utils.logging"].get_logger = logging.getLogger
    spec = importlib.util.spec_from_file_location(
        "token_logging_trainer_under_test", ROOT / "mindspeed_llm/fsdp2/train/train/trainer.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class LogitsOutput(dict):
    def __init__(self, logits):
        super().__init__(logits=logits)
        self.logits = logits


class TrainerTokenLossTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_trainer()

    def setUp(self):
        self.trainer = self.module.Trainer.__new__(self.module.Trainer)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.trainer.args = types.SimpleNamespace(
            stage="pt", log_per_token_loss=True, token_loss_logging_steps=1,
            output_dir=self.directory.name, calculate_per_token_loss=False,
        )
        self.trainer.optimization_args = types.SimpleNamespace(chunk_loss_size=None)
        self.trainer.global_step = 4
        self.trainer._token_loss_micro_step = 2
        self.trainer.tokenizer = None
        self.module.ParallelState.return_value.get_group_size.return_value = 1
        self.module.write_token_loss_logs.reset_mock()
        self.inputs = {
            "input_ids": torch.tensor([[0, 1, 2], [1, 0, 1]]),
            "labels": torch.tensor([[1, -100, 0], [2, 1, 0]]),
            "position_ids": torch.tensor([[10, 11, 12]]),
        }
        self.device_patch = patch.object(torch.accelerator, "current_device", return_value="cpu", create=True)
        self.device_patch.start()
        self.addCleanup(self.device_patch.stop)

    def test_pretraining_alignment_scalar_and_gradient(self):
        torch.manual_seed(41)
        logits = torch.randn(2, 3, 3, requires_grad=True)
        reference = logits.detach().clone().requires_grad_()
        expected = F.cross_entropy(reference.reshape(-1, 3), self.inputs["labels"].reshape(-1))
        expected.backward()
        actual, details = self.trainer._compute_language_model_pretrain_loss(
            logits, self.inputs["labels"], return_token_losses=True
        )
        actual.backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(logits.grad, reference.grad)
        torch.testing.assert_close(details, F.cross_entropy(
            logits.detach().reshape(-1, 3), self.inputs["labels"].reshape(-1), reduction="none"
        ).reshape(2, 3))
        self.assertEqual(actual.ndim, 0)
        self.assertFalse(details.requires_grad)

    def test_fallback_writes_step_microbatch_and_positions(self):
        logits = torch.randn(2, 3, 3, requires_grad=True)
        self.trainer.model = Mock(return_value=LogitsOutput(logits))
        self.trainer.model.supports_token_loss_logging = False
        self.trainer.model.model.supports_token_loss_logging = False
        self.trainer.model.config.enable_diffusion_lm = False
        labels = self.inputs["labels"].clone()
        loss = self.trainer._compute_loss(self.inputs.copy())
        self.assertEqual(loss.ndim, 0)
        call = self.module.write_token_loss_logs.call_args
        torch.testing.assert_close(call.args[0]["ar"]["target_ids"], labels)
        self.assertEqual(call.kwargs["step"], 5)
        self.assertEqual(call.kwargs["micro_step"], 2)
        torch.testing.assert_close(call.kwargs["position_ids"], self.inputs["position_ids"])

    def test_wrapped_diffusion_model_and_logging_interval(self):
        details = {"ar": {"losses": torch.ones(2, 3)}}
        numerator = torch.tensor(9.0, requires_grad=True)
        self.trainer.model = Mock(return_value={
            "loss": (numerator, torch.tensor(3)), "token_loss_details": details,
        })
        self.trainer.model.supports_token_loss_logging = False
        self.trainer.model.model.supports_token_loss_logging = True
        self.trainer.model.config.enable_diffusion_lm = True
        self.trainer.args.token_loss_logging_steps = 5
        loss = self.trainer._compute_loss(self.inputs.copy())
        self.assertEqual(loss.item(), 3.0)
        self.assertTrue(self.trainer.model.call_args.kwargs["log_per_token_loss"])
        self.assertIs(self.module.write_token_loss_logs.call_args.args[0], details)
        self.module.write_token_loss_logs.reset_mock()
        self.trainer.global_step = 5
        self.trainer._compute_loss(self.inputs.copy())
        self.module.write_token_loss_logs.assert_not_called()
        self.assertNotIn("log_per_token_loss", self.trainer.model.call_args.kwargs)

    def test_tuple_loss_uses_local_count_without_distributed_scaling(self):
        self.module.ParallelState.return_value.get_group_size.return_value = 8
        for container in (tuple, list):
            with self.subTest(container=container.__name__):
                numerator = torch.tensor(9.0, requires_grad=True)
                count = torch.tensor(3.0, requires_grad=True)
                with patch.object(self.module.dist, "is_initialized", return_value=True), \
                        patch.object(self.module.dist, "all_reduce") as reduce_count:
                    loss = self.trainer._normalize_token_sum_loss(container((numerator, count)))
                reduce_count.assert_not_called()
                torch.testing.assert_close(loss, torch.tensor(3.0))
                loss.backward()
                torch.testing.assert_close(numerator.grad, torch.tensor(1.0 / 3.0))
                self.assertIsNone(count.grad)
                self.assertEqual(count.item(), 3.0)

    def test_tuple_loss_clamps_empty_count(self):
        numerator = torch.tensor(0.0, requires_grad=True)
        loss = self.trainer._normalize_token_sum_loss((numerator, 0))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertEqual(numerator.grad.item(), 1.0)

    def test_scalar_loss_is_not_normalized_twice(self):
        loss = torch.tensor(2.0, requires_grad=True)
        self.assertIs(self.trainer._normalize_token_sum_loss(loss), loss)

    def test_training_step_matches_default_rank_and_microbatch_mean(self):
        # Exercise the real training_step -> _compute_loss -> normalization path.
        # Average rank gradients explicitly: this is not a distributed backend test.
        self.trainer.args.log_per_token_loss = False
        self.trainer.optimizer = Mock(spec=[])
        coefficients = torch.tensor([[120.0, 360.0], [360.0, 300.0]])
        counts = torch.tensor([[60.0, 90.0], [120.0, 150.0]])
        self.module.ParallelState.return_value.get_group_size.return_value = 2
        # Also cover the final, shorter accumulation window (actual K = 1).
        for accumulation_steps in (2, 1):
            with self.subTest(accumulation_steps=accumulation_steps):
                self.trainer.current_gradient_accumulation_steps = accumulation_steps
                rank_gradients, rank_losses = [], []
                with patch.object(self.module.dist, "is_initialized", return_value=True), \
                        patch.object(self.module.dist, "all_reduce") as reduce_count, \
                        patch.object(torch.cuda, "device_count", return_value=0):
                    for rank in range(2):
                        parameter = torch.tensor(1.0, requires_grad=True)
                        outputs = [
                            {"loss": (coefficients[rank, micro] * parameter.square(), counts[rank, micro])}
                            for micro in range(accumulation_steps)
                        ]
                        self.trainer.model = Mock(side_effect=outputs)
                        self.trainer.model.config.enable_diffusion_lm = True
                        losses = [
                            self.trainer.training_step(self.inputs.copy(), num_items_in_batch=10000)
                            for _ in range(accumulation_steps)
                        ]
                        rank_losses.append(torch.stack(losses).sum())
                        rank_gradients.append(parameter.grad.clone())
                reduce_count.assert_not_called()
                reference = torch.tensor(1.0, requires_grad=True)
                expected = (coefficients[:, :accumulation_steps] * reference.square()
                            / counts[:, :accumulation_steps]).mean()
                expected.backward()
                torch.testing.assert_close(torch.stack(rank_losses).mean(), expected)
                torch.testing.assert_close(torch.stack(rank_gradients).mean(), reference.grad)

    def test_chunk_loss_fails_before_model_forward(self):
        self.trainer.optimization_args.chunk_loss_size = 128
        self.trainer.model = Mock()
        with self.assertRaisesRegex(ValueError, "chunk_loss_size"):
            self.trainer._compute_loss(self.inputs.copy())
        self.trainer.model.assert_not_called()

    def test_missing_model_details_is_explicit(self):
        self.trainer.model = Mock(return_value={"loss": torch.tensor(1.0)})
        self.trainer.model.config.enable_diffusion_lm = True
        with self.assertRaisesRegex(ValueError, "token_loss_details"):
            self.trainer._compute_loss(self.inputs.copy())


if __name__ == "__main__":
    unittest.main(verbosity=2)
