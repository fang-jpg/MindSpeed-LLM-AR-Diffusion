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
