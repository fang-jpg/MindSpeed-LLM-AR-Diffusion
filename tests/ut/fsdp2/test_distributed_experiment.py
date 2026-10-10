import importlib.util
import sys
from functools import wraps
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


def test_context_bypasses_megatron_dependent_collective_wrappers(monkeypatch):
    calls = []

    def native_barrier():
        calls.append("barrier")

    @wraps(native_barrier)
    def mindspeed_barrier_wrapper():
        raise AssertionError("args is not initialized")

    def native_all_gather_object(output, value):
        calls.append("all_gather_object")
        for rank in range(len(output)):
            output[rank] = {"rank": rank, "value": value}

    @wraps(native_all_gather_object)
    def mindspeed_all_gather_wrapper(output, value):
        raise AssertionError("args is not initialized")

    monkeypatch.setattr(distributed.dist, "barrier", mindspeed_barrier_wrapper)
    monkeypatch.setattr(
        distributed.dist,
        "all_gather_object",
        mindspeed_all_gather_wrapper,
    )
    context = distributed.DistributedContext(
        rank=0, local_rank=0, world_size=2, device="npu:0"
    )

    context.barrier()
    gathered = context.all_gather_object("sample")

    assert calls == ["barrier", "all_gather_object"]
    assert gathered == [
        {"rank": 0, "value": "sample"},
        {"rank": 1, "value": "sample"},
    ]

