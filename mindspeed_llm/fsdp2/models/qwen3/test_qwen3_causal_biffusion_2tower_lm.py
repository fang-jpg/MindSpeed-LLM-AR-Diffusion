import pytest
import torch
from transformers import Qwen3Config
from transformers.modeling_outputs import CausalLMOutputWithPast

from mindspeed_llm.fsdp2.models.model_registry import ModelRegistry
from mindspeed_llm.fsdp2.models.qwen3.qwen3 import Qwen3ForCausalLM
from mindspeed_llm.fsdp2.models.qwen3.qwen3_causal_biffusion_2tower_lm import (
    Qwen3CausalBiffusion2TowerLm,
)


@pytest.fixture
def tiny_qwen3_config():
    return Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=False,
        block_size=4,
    )


def test_two_towers_are_complete_and_independent(tiny_qwen3_config):
    model = Qwen3CausalBiffusion2TowerLm(tiny_qwen3_config)

    assert model.mask_token_id == 32
    assert model.config.vocab_size == 33
    assert model.encoder.get_input_embeddings().num_embeddings == 33
    assert model.encoder.get_output_embeddings().out_features == 33
    assert model.decoder.get_input_embeddings().num_embeddings == 33
    assert model.decoder.get_output_embeddings().out_features == 33
    assert isinstance(model.encoder, Qwen3ForCausalLM)
    assert isinstance(model.decoder, Qwen3ForCausalLM)
    assert model.encoder.model is not model.decoder.model
    assert model.encoder.lm_head is not model.decoder.lm_head
    assert model.encoder.model.embed_tokens.weight is not model.decoder.model.embed_tokens.weight
    assert model.encoder.lm_head.weight is not model.decoder.lm_head.weight


def test_only_decoder_parameters_are_trainable(tiny_qwen3_config):
    model = Qwen3CausalBiffusion2TowerLm(tiny_qwen3_config)

    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.decoder.parameters())

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    optimized_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert optimized_parameters == {id(parameter) for parameter in model.decoder.parameters()}
    assert optimized_parameters.isdisjoint(
        id(parameter) for parameter in model.encoder.parameters()
    )


def test_model_is_registered():
    assert (
        ModelRegistry.get_model_class("qwen3_causal_biffusion_2tower")
        is Qwen3CausalBiffusion2TowerLm
    )


def test_forward_returns_standard_causal_lm_output(tiny_qwen3_config):
    torch.manual_seed(7)
    model = Qwen3CausalBiffusion2TowerLm(tiny_qwen3_config)
    input_ids = torch.randint(0, 31, (2, 7))
    labels = input_ids.clone()
    encoder_weights_before = {
        name: parameter.detach().clone()
        for name, parameter in model.encoder.named_parameters()
    }

    outputs = model(input_ids=input_ids, labels=labels)
    outputs.loss.backward()

    assert isinstance(outputs, CausalLMOutputWithPast)
    assert outputs.logits.shape == (2, 7, tiny_qwen3_config.vocab_size)
    assert outputs.loss.ndim == 0
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.decoder.parameters())
    assert all(
        torch.equal(parameter, encoder_weights_before[name])
        for name, parameter in model.encoder.named_parameters()
    )


def test_each_block_samples_one_probability_per_sequence(tiny_qwen3_config):
    torch.manual_seed(11)
    model = Qwen3CausalBiffusion2TowerLm(tiny_qwen3_config)
    input_ids = torch.randint(0, 31, (2, 10))

    _, _, p_mask = model._sample_block_mask(input_ids)

    for start in range(0, input_ids.shape[1], tiny_qwen3_config.block_size):
        block_probabilities = p_mask[:, start:start + tiny_qwen3_config.block_size]
        assert torch.all(block_probabilities == block_probabilities[:, :1])


def test_block_attention_is_bidirectional_inside_and_causal_between_blocks(
    tiny_qwen3_config,
):
    model = Qwen3CausalBiffusion2TowerLm(tiny_qwen3_config)
    input_ids = torch.ones((1, 10), dtype=torch.long)

    attention_mask = model._build_block_causal_attention_mask(
        input_ids,
        attention_mask=None,
    )
    can_attend = attention_mask[0, 0] == 0

    # Block 0 is fully bidirectional.
    assert can_attend[:4, :4].all()
    # Block 1 sees block 0 and itself in both directions.
    assert can_attend[4:8, :8].all()
    # Earlier blocks cannot see later blocks.
    assert not can_attend[:4, 4:].any()
    assert not can_attend[4:8, 8:].any()
    # The final partial block sees all preceding and same-block tokens.
    assert can_attend[8:, :].all()
