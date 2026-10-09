import importlib.util
import sys
import types
from pathlib import Path

import torch


def load_generation_module_without_mindspeed_runtime():
    module_names = [
        "mindspeed_llm",
        "mindspeed_llm.fsdp2",
        "mindspeed_llm.fsdp2.models",
        "mindspeed_llm.fsdp2.models.qwen3",
        "mindspeed_llm.fsdp2.models.qwen3.utils",
        "mindspeed_llm.fsdp2.models.qwen3_diffusion",
        "mindspeed_llm.fsdp2.models.qwen3_diffusion.generation",
    ]
    saved = {name: sys.modules.get(name) for name in module_names}
    try:
        for name in module_names[:-2]:
            package = types.ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package
        utils = types.ModuleType("mindspeed_llm.fsdp2.models.qwen3.utils")
        utils.set_diffusion_lm = lambda model, enabled: setattr(
            model, "diffusion_enabled", enabled
        )
        sys.modules[utils.__name__] = utils
        diffusion_package = types.ModuleType(
            "mindspeed_llm.fsdp2.models.qwen3_diffusion"
        )
        diffusion_package.__path__ = []
        sys.modules[diffusion_package.__name__] = diffusion_package

        path = (
            Path(__file__).resolve().parents[3]
            / "mindspeed_llm"
            / "fsdp2"
            / "models"
            / "qwen3_diffusion"
            / "generation.py"
        )
        name = "mindspeed_llm.fsdp2.models.qwen3_diffusion.generation"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


generation = load_generation_module_without_mindspeed_runtime()


class FakeCache:
    def __init__(self):
        self.length = 0

    def get_seq_length(self):
        return self.length

    def crop(self, max_length):
        self.length = max_length


class FakeBackbone:
    def __call__(
        self,
        input_ids,
        position_ids=None,
        past_key_values=None,
        use_cache=True,
        cache_position=None,
        **kwargs,
    ):
        if cache_position is not None and cache_position.numel() > 0:
            past_key_values.length = max(
                past_key_values.length, int(cache_position[-1].item()) + 1
            )
        hidden = torch.zeros((*input_ids.shape, 4), dtype=torch.float32)
        return types.SimpleNamespace(
            last_hidden_state=hidden,
            past_key_values=past_key_values,
        )


class FakeModel:
    def __init__(self):
        self.mask_token_id = 9
        self.config = types.SimpleNamespace(eos_token_id=2)
        self.model = FakeBackbone()
        self.diffusion_enabled = False

    def lm_head(self, hidden_states):
        logits = torch.zeros((*hidden_states.shape[:-1], 10), dtype=torch.float32)
        logits[..., 1] = 10
        return logits, None


def test_transfer_schedule_distributes_every_mask_once():
    schedule = generation._get_num_transfer_tokens(
        torch.tensor([[True, True, True, True, True]]), steps=2
    )
    assert schedule.tolist() == [[3, 2]]


def test_block_generation_decouples_steps_and_handles_short_final_block(monkeypatch):
    monkeypatch.setattr(generation, "DynamicCache", FakeCache)
    prompt = torch.tensor([[3, 4, 5]], dtype=torch.long)
    output, stats = generation.block_diff_generate(
        model=FakeModel(),
        prompt_ids=prompt,
        max_new_tokens=10,
        block_length=4,
        steps_per_block=2,
        seed_mode="none",
        ignore_eos=True,
        return_stats=True,
    )

    assert output.shape == (1, 13)
    assert output[:, 3:].tolist() == [[1] * 10]
    assert stats.prefill_nfe == 1
    assert stats.denoise_nfe == 6
    assert stats.refresh_nfe == 2
    assert stats.total_nfe == 9
    assert stats.generated_tokens == 10
    assert sum(stats.committed_per_step) == 10
