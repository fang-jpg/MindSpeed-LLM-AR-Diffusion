import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
SAMPLER_DIR = (
    ROOT
    / "mindspeed_llm"
    / "fsdp2"
    / "models"
    / "qwen3_diffusion"
    / "sampler"
)


def _load_module(short_name):
    package_names = [
        "mindspeed_llm",
        "mindspeed_llm.fsdp2",
        "mindspeed_llm.fsdp2.models",
        "mindspeed_llm.fsdp2.models.qwen3_diffusion",
        "mindspeed_llm.fsdp2.models.qwen3_diffusion.sampler",
    ]
    for name in package_names:
        if name not in sys.modules:
            package = importlib.util.module_from_spec(
                importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            )
            package.__path__ = []
            sys.modules[name] = package
    full_name = (
        "mindspeed_llm.fsdp2.models.qwen3_diffusion.sampler." + short_name
    )
    spec = importlib.util.spec_from_file_location(
        full_name, SAMPLER_DIR / f"{short_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


model_module = _load_module("model")
features_module = _load_module("features")
data_module = _load_module("data")
metrics_module = _load_module("metrics")


def _load_trajectory_module():
    names = [
        "transformers",
        "transformers.cache_utils",
        "mindspeed_llm.fsdp2.models.qwen3",
        "mindspeed_llm.fsdp2.models.qwen3.utils",
        "mindspeed_llm.fsdp2.models.qwen3_diffusion.generation",
        "mindspeed_llm.fsdp2.models.qwen3_diffusion.sampler.trajectory",
    ]
    saved = {name: sys.modules.get(name) for name in names}
    try:
        transformers_package = types.ModuleType("transformers")
        transformers_package.__path__ = []
        sys.modules[transformers_package.__name__] = transformers_package
        cache_utils = types.ModuleType("transformers.cache_utils")
        cache_utils.DynamicCache = _FakeCache
        sys.modules[cache_utils.__name__] = cache_utils

        qwen3_package = types.ModuleType("mindspeed_llm.fsdp2.models.qwen3")
        qwen3_package.__path__ = []
        sys.modules[qwen3_package.__name__] = qwen3_package
        qwen3_utils = types.ModuleType("mindspeed_llm.fsdp2.models.qwen3.utils")
        qwen3_utils.set_diffusion_lm = lambda model, enabled: setattr(
            model, "diffusion_enabled", enabled
        )
        sys.modules[qwen3_utils.__name__] = qwen3_utils
        generation_module = types.ModuleType(
            "mindspeed_llm.fsdp2.models.qwen3_diffusion.generation"
        )
        generation_module._crop_dynamic_cache = (
            lambda cache, length: cache.crop(length)
        )
        sys.modules[generation_module.__name__] = generation_module
        return _load_module("trajectory")
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_paper_default_sampler_is_about_4p8m_parameters():
    sampler = model_module.DiffusionSampler()
    assert sampler.parameter_count() == 4_804_897


def test_feature_extractor_produces_exactly_144_features():
    hidden_size = 48
    pca = features_module.EmbeddingPCA(
        mean=torch.zeros(hidden_size),
        components=torch.eye(hidden_size)[:46],
        explained_variance=torch.ones(46),
        source_rows=64,
        seed=1,
    )
    embeddings = torch.arange(8 * hidden_size, dtype=torch.float32).reshape(
        8, hidden_size
    )
    extractor = features_module.SamplerFeatureExtractor(
        pca,
        embeddings,
        invalid_token_id=7,
        table_dtype=torch.float32,
        projection_batch_size=3,
    )
    logits = torch.tensor(
        [[[0.0, 1.0, 4.0, 3.0, 2.0, -1.0, -2.0, 100.0]]]
    )
    result = extractor(logits)
    assert result.features.shape == (1, 1, 144)
    assert result.top1_ids.item() == 2
    assert 0.0 < result.top1_probability.item() < 1.0
    assert torch.isfinite(result.features).all()


def test_shards_round_trip_and_pad_mixed_block_lengths(tmp_path):
    writer = data_module.SamplerShardWriter(tmp_path, records_per_shard=1)
    for block_length in (8, 32):
        active = torch.zeros(block_length, dtype=torch.bool)
        active[:2] = True
        writer.add(
            "train",
            data_module.SamplerTrainingRecord(
                features=torch.ones(block_length, 144),
                labels=active.clone(),
                active_mask=active,
                policy="confidence",
                sample_index=0,
                block_index=0,
                step_index=0,
            ),
        )
    writer.close()
    dataset = data_module.SamplerShardDataset(
        tmp_path, "train", max_block_length=32, shuffle=False
    )
    records = list(dataset)
    assert len(records) == 2
    assert all(record["features"].shape == (32, 144) for record in records)
    assert [int(record["active_mask"].sum()) for record in records] == [2, 2]


def test_load_megatron_prefixes_from_fsdp2_yaml(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
data:
  dataset:
    file_name:
      - /data/math_text_document.bin
      - /data/code_text_document
  max_samples: 123
  data_manager_type: mg
""".strip(),
        encoding="utf-8",
    )
    config = data_module.load_megatron_data_config(config_path)
    assert config.prefixes == (
        "/data/math_text_document",
        "/data/code_text_document",
    )
    assert config.max_samples == 123


class _FakeIndexedDataset:
    def __init__(self, sequences, document_indices):
        self.sequences = [np.asarray(item, dtype=np.int32) for item in sequences]
        self.document_indices = np.asarray(document_indices, dtype=np.int64)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]


def test_megatron_documents_preserve_boundaries_and_shard_across_ranks():
    datasets = {
        "math": _FakeIndexedDataset(
            sequences=[[1, 2], [3], [4, 5]],
            document_indices=[0, 2, 3],
        ),
        "code": _FakeIndexedDataset(
            sequences=[[6, 7, 8], [9], [10, 11]],
            document_indices=[0, 1, 3],
        ),
    }
    factory = datasets.__getitem__
    complete = list(
        data_module.iter_megatron_documents(
            ["math", "code"],
            seed=7,
            dataset_factory=factory,
        )
    )
    assert len(complete) == 4
    by_key = {item.key: item.token_ids.tolist() for item in complete}
    assert by_key == {
        "math:0": [1, 2, 3],
        "math:1": [4, 5],
        "code:0": [6, 7, 8],
        "code:1": [9, 10, 11],
    }

    rank_documents = []
    for rank in (0, 1):
        rank_documents.extend(
            data_module.iter_megatron_documents(
                ["math", "code"],
                rank=rank,
                world_size=2,
                max_examples=3,
                seed=7,
                dataset_factory=factory,
            )
        )
    assert len(rank_documents) == 3
    assert {item.sample_index for item in rank_documents} == {0, 1, 2}
    assert len({item.key for item in rank_documents}) == 3


def test_auc_meter_handles_ties():
    meter = metrics_module.BinaryAUCMeter(128, torch.device("cpu"))
    meter.update(
        torch.tensor([0.1, 0.2, 0.8, 0.9]),
        torch.tensor([False, False, True, True]),
    )
    assert meter.compute() == 1.0


class _FakeCache:
    def __init__(self):
        self.length = 0

    def get_seq_length(self):
        return self.length

    def crop(self, length):
        self.length = length


class _FakeBackbone:
    def __init__(self):
        self.layers = []
        self.last_input_ids = None

    def __call__(
        self,
        input_ids,
        past_key_values,
        cache_position,
        **kwargs,
    ):
        self.last_input_ids = input_ids.clone()
        if cache_position.numel():
            past_key_values.length = int(cache_position[-1].item()) + 1
        return types.SimpleNamespace(
            last_hidden_state=torch.zeros((*input_ids.shape, 4)),
            past_key_values=past_key_values,
        )


class _FakeDiffusionModel:
    def __init__(self):
        self.mask_token_id = 5
        self.model = _FakeBackbone()
        self.diffusion_enabled = False

    def lm_head(self, hidden_states):
        input_ids = self.model.last_input_ids
        logits = torch.full((*input_ids.shape, 6), -5.0)
        if input_ids[0, 0].item() == self.mask_token_id:
            logits[0, 0, 1] = 8.0
            logits[0, 1, 2] = 7.0
        else:
            logits[0, 0, 1] = 8.0
            logits[0, 1, 3] = 8.0
        return logits, None


def test_confidence_trajectory_labels_against_eventual_tokens(monkeypatch):
    trajectory_module = _load_trajectory_module()
    monkeypatch.setattr(trajectory_module, "DynamicCache", _FakeCache)
    pca = features_module.EmbeddingPCA(
        mean=torch.zeros(48),
        components=torch.eye(48)[:46],
        explained_variance=torch.ones(46),
        source_rows=64,
        seed=1,
    )
    extractor = features_module.SamplerFeatureExtractor(
        pca,
        torch.randn(6, 48),
        invalid_token_id=5,
        table_dtype=torch.float32,
    )
    records, final_ids = trajectory_module.collect_block_trajectory(
        model=_FakeDiffusionModel(),
        prefix_ids=torch.tensor([[4]]),
        block_length=2,
        feature_extractor=extractor,
        policy="confidence",
        sample_index=0,
        block_index=0,
    )
    assert final_ids.tolist() == [1, 3]
    assert len(records) == 2
    assert records[0].active_mask.tolist() == [True, True]
    assert records[0].labels.tolist() == [True, False]
    assert records[1].active_mask.tolist() == [False, True]
    assert records[1].labels.tolist() == [False, True]
