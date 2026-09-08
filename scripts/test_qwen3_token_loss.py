"""CPU regression tests for the production Qwen3 diffusion loss forward path.

Run: python scripts/test_qwen3_token_loss.py

Requires torch and transformers. A small embedding backbone replaces attention;
the production forward, LMHead, loss calculations, and autograd remain intact.
These tests cover loss normalization/logging, not NPU or distributed training.
"""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config
from transformers.modeling_outputs import CausalLMOutputWithPast


ROOT = Path(__file__).resolve().parents[1]


def load_model_source():
    """Bypass the Megatron package initializer and unused hardware patch manager."""
    packages = {}
    for name in (
        "mindspeed", "mindspeed_llm", "mindspeed_llm.fsdp2",
        "mindspeed_llm.fsdp2.models", "mindspeed_llm.fsdp2.models.common",
    ):
        packages[name] = types.ModuleType(name)
        packages[name].__path__ = []
    patch_utils = types.ModuleType("mindspeed.patch_utils")
    patch_utils.MindSpeedPatchesManager = object
    packages[patch_utils.__name__] = patch_utils

    def load(name, relative_path):
        spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    with patch.dict(sys.modules, packages):
        load("mindspeed_llm.fsdp2.models.common.fusions",
             "mindspeed_llm/fsdp2/models/common/fusions.py")
        load("mindspeed_llm.fsdp2.models.common.modules",
             "mindspeed_llm/fsdp2/models/common/modules.py")
        return load("_qwen3_token_loss_test_source",
                    "mindspeed_llm/fsdp2/models/qwen3_diffusion/modeling_qwen3_diffusion.py")


SOURCE = load_model_source()


class EmbeddingBackbone(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.cache_marker = object()

    def forward(self, input_ids, **kwargs):
        hidden = self.embed_tokens(input_ids)
        return types.SimpleNamespace(
            last_hidden_state=hidden,
            past_key_values=self.cache_marker,
            hidden_states=(hidden,),
            attentions=None,
        )


def make_model(paradigm):
    torch.manual_seed(17)
    model = SOURCE.Qwen3DiffusionForCausalLM.__new__(SOURCE.Qwen3DiffusionForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = Qwen3Config(vocab_size=11, hidden_size=5)
    model.config.dlm_paradigm = paradigm
    model.config.block_size = 2
    model.config.dlm_loss_weight = 0.35
    model.config.ar_loss_weight = 0.7
    model.mask_token_id = 10
    model.bd_mask = None
    model.block_diff_position_ids = None
    model.model = EmbeddingBackbone(model.config)
    model.lm_head = SOURCE.LMHead(5, 11, bias=False)
    return model


def target_nll(logits, targets):
    """Independent negative-log-probability reference, including ignored labels."""
    valid = targets.ne(-100)
    safe_targets = targets.masked_fill(~valid, 0)
    losses = -logits.float().log_softmax(-1).gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return losses.masked_fill(~valid, 0)


class TokenLossTests(unittest.TestCase):
    def setUp(self):
        # Targets deliberately differ from inputs and are already shifted.
        self.input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        self.labels = torch.tensor([[2, 3, -100, 9], [6, -100, 8, 1]])
        self.mask = torch.tensor([[True, False, True, False], [False, True, False, True]])
        self.probabilities = torch.tensor([[0.2] * 4, [0.00001] * 4])

    def fixed_mask(self, model, mask=None):
        selected = self.mask if mask is None else mask

        def forward_process(input_ids, eps=1e-3, loss_mask=None):
            valid = selected.clone()
            if loss_mask is not None:
                valid &= loss_mask.ne(0)
            return (input_ids.masked_fill(valid, model.mask_token_id),
                    valid, self.probabilities.clone())

        model.forward_process = forward_process

    def run_forward(self, model, enabled=True, **kwargs):
        arguments = dict(input_ids=self.input_ids, labels=self.labels,
                         log_per_token_loss=enabled)
        arguments.update(kwargs)
        # The existing joint path also prints aggregate loss values.
        with contextlib.redirect_stdout(io.StringIO()):
            return model(**arguments)

    def assert_detached_details(self, details):
        for objective in details.values():
            for tensor in objective.values():
                self.assertEqual(tensor.shape, self.input_ids.shape)
                self.assertFalse(tensor.requires_grad)
                self.assertIsNone(tensor.grad_fn)

    def test_ar_uses_already_shifted_targets_and_ignores_minus_100(self):
        output = self.run_forward(make_model("autoregressive"))
        ar = output.token_loss_details["ar"]
        expected = target_nll(output.logits, self.labels)
        torch.testing.assert_close(ar["losses"], expected)
        torch.testing.assert_close(ar["target_ids"], self.labels)
        torch.testing.assert_close(ar["valid_mask"], self.labels.ne(-100))
        torch.testing.assert_close(output.loss, expected.sum() / self.labels.ne(-100).sum())
        self.assertEqual(output.loss.ndim, 0)
        self.assert_detached_details(output.token_loss_details)

    def test_logging_preserves_loss_and_gradients_for_all_paradigms(self):
        for paradigm in ("autoregressive", "bidirectional", "block_diff"):
            with self.subTest(paradigm=paradigm):
                model = make_model(paradigm)
                self.fixed_mask(model)
                results = []
                for enabled in (False, True):
                    model.zero_grad(set_to_none=True)
                    output = self.run_forward(model, enabled)
                    scalar = output.loss
                    if isinstance(scalar, tuple):
                        scalar = scalar[0] / scalar[1]
                    scalar.backward()
                    results.append((scalar.detach(), [p.grad.clone() for p in model.parameters()]))
                    self.assertEqual("token_loss_details" in output, enabled)
                torch.testing.assert_close(results[0][0], results[1][0])
                for original, logged in zip(results[0][1], results[1][1]):
                    torch.testing.assert_close(original, logged)

    def test_joint_details_reconstruct_weighted_training_numerator(self):
        model = make_model("block_diff")
        self.fixed_mask(model)
        output = self.run_forward(model)
        ar = output.token_loss_details["ar"]
        diffusion = output.token_loss_details["diffusion"]
        ar_nll = target_nll(output.causal_logits, self.labels)
        diff_nll = target_nll(output.logits, self.input_ids)
        torch.testing.assert_close(ar["losses"], ar_nll)
        torch.testing.assert_close(ar["weighted_losses"], ar_nll * model.config.ar_loss_weight)
        torch.testing.assert_close(diffusion["losses"][self.mask], diff_nll[self.mask])
        torch.testing.assert_close(diffusion["valid_mask"], self.mask)
        torch.testing.assert_close(diffusion["target_ids"], self.input_ids)
        torch.testing.assert_close(diffusion["p_mask"], self.probabilities)
        expected_diff = diff_nll / self.probabilities.clamp_min(1e-3) * model.config.dlm_loss_weight
        torch.testing.assert_close(diffusion["weighted_losses"][self.mask], expected_diff[self.mask])
        torch.testing.assert_close(output.loss[0],
                                   diffusion["weighted_losses"].sum() + ar["weighted_losses"].sum())
        torch.testing.assert_close(output.loss[1],
                                   self.mask.sum().float() + self.labels.numel())
        torch.testing.assert_close(output.ar_loss_sum, ar_nll.sum())
        torch.testing.assert_close(output.diffusion_loss_sum, expected_diff[self.mask].sum())
        self.assert_detached_details(output.token_loss_details)

    def test_joint_default_denominator_and_gradients_for_ar_weights(self):
        for ar_weight in (0.0, 1.0, 2.0):
            with self.subTest(ar_weight=ar_weight):
                model = make_model("block_diff")
                model.config.ar_loss_weight = ar_weight
                self.fixed_mask(model)
                output = self.run_forward(model, enabled=False)
                ar_nll = target_nll(output.causal_logits, self.labels).sum()
                diffusion_nll = (target_nll(output.logits, self.input_ids)[self.mask]
                                 / self.probabilities[self.mask].clamp_min(1e-3)).sum()
                # Official DGPTStep counts every AR label, including -100 slots.
                count = self.mask.sum().float() + self.labels.numel()
                expected = (model.config.dlm_loss_weight * diffusion_nll + ar_weight * ar_nll) / count
                actual = output.loss[0] / output.loss[1]
                torch.testing.assert_close(output.loss[1], count)
                self.assertEqual(output.ar_token_count.item(), self.labels.numel())
                torch.testing.assert_close(actual, expected)
                parameters = list(model.parameters())
                expected_gradients = torch.autograd.grad(expected, parameters, retain_graph=True)
                actual.backward()
                for parameter, gradient in zip(parameters, expected_gradients):
                    torch.testing.assert_close(parameter.grad, gradient)

    def test_bidirectional_logs_input_reconstruction_only(self):
        model = make_model("bidirectional")
        self.fixed_mask(model)
        output = self.run_forward(model)
        self.assertEqual(set(output.token_loss_details), {"diffusion"})
        details = output.token_loss_details["diffusion"]
        torch.testing.assert_close(details["target_ids"], self.input_ids)
        torch.testing.assert_close(details["valid_mask"], self.mask)
        expected = target_nll(output.logits, self.input_ids)
        torch.testing.assert_close(details["losses"][self.mask], expected[self.mask])
        torch.testing.assert_close(output.loss, details["weighted_losses"].sum())

    def test_no_masked_tokens_keeps_empty_diffusion_observations(self):
        for paradigm in ("bidirectional", "block_diff"):
            with self.subTest(paradigm=paradigm):
                model = make_model(paradigm)
                self.fixed_mask(model, torch.zeros_like(self.mask))
                output = self.run_forward(model)
                details = output.token_loss_details["diffusion"]
                self.assertFalse(details["valid_mask"].any())
                self.assertEqual(details["weighted_losses"].sum().item(), 0)
                scalar = output.loss[0] if isinstance(output.loss, tuple) else output.loss
                self.assertTrue(torch.isfinite(scalar))
                scalar.backward()

    def test_all_ignored_ar_labels_preserve_existing_mean_behavior(self):
        model = make_model("autoregressive")
        labels = torch.full_like(self.labels, -100)
        original = self.run_forward(model, False, labels=labels)
        logged = self.run_forward(model, True, labels=labels)
        # Existing CE(mean) is NaN when no targets contribute; logging preserves it.
        self.assertTrue(torch.isnan(original.loss))
        self.assertTrue(torch.isnan(logged.loss))
        self.assertFalse(logged.token_loss_details["ar"]["valid_mask"].any())
        self.assertEqual(logged.token_loss_details["ar"]["losses"].sum().item(), 0)

    def test_real_mask_sampler_respects_loss_mask(self):
        model = make_model("bidirectional")
        loss_mask = torch.zeros_like(self.input_ids)
        output = self.run_forward(model, loss_mask=loss_mask)
        self.assertFalse(output.token_loss_details["diffusion"]["valid_mask"].any())
        self.assertEqual(output.loss.item(), 0)

    def test_no_labels_preserves_inference_output_and_cache(self):
        for paradigm in ("autoregressive", "bidirectional", "block_diff"):
            with self.subTest(paradigm=paradigm):
                model = make_model(paradigm).eval()
                output = self.run_forward(model, labels=None)
                self.assertIsInstance(output, CausalLMOutputWithPast)
                self.assertNotIn("token_loss_details", output)
                self.assertIsNone(output.loss)
                self.assertEqual(output.logits.shape, (2, 4, 11))
                self.assertIs(output.past_key_values, model.model.cache_marker)
                self.assertEqual(output.hidden_states[0].shape, (2, 4, 5))

    def test_chunk_loss_logging_reports_incompatibility(self):
        for paradigm in ("autoregressive", "bidirectional", "block_diff"):
            with self.subTest(paradigm=paradigm):
                with self.assertRaisesRegex(ValueError, "chunk_loss_size"):
                    self.run_forward(make_model(paradigm), loss_ctx=lambda *args: None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
