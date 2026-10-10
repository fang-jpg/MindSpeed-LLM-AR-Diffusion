import importlib.util
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "fsdp2"
    / "qwen3_diffusion"
    / "distributed_experiment.py"
)
SPEC = importlib.util.spec_from_file_location(
    "distributed_experiment_for_test", MODULE_PATH
)
distributed = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = distributed
SPEC.loader.exec_module(distributed)


def test_shard_sequence_covers_items_once():
    values = list(range(13))
    shards = [distributed.shard_sequence(values, rank, 8) for rank in range(8)]
    assert sorted(item for shard in shards for item in shard) == values
    assert shards[0] == [0, 8]
    assert shards[7] == [7]


def test_single_process_context_does_not_initialize_collective(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    context = distributed.initialize_distributed("cpu")
    assert context.world_size == 1
    assert context.device == "cpu"
    assert context.is_main
    assert context.all_gather_object({"ok": True}) == [{"ok": True}]

