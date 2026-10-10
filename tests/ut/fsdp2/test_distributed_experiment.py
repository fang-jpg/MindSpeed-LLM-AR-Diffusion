import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
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

    def native_all_gather(output, value):
        calls.append("all_gather_object")
        for rank in range(len(output)):
            output[rank] = {"rank": rank, "value": value}

    @wraps(native_all_gather)
    def mindspeed_inner_all_gather_wrapper(output, value):
        raise AssertionError("args is not initialized")

    object_collective_namespace = {
        "all_gather": mindspeed_inner_all_gather_wrapper
    }
    exec(
        "def native_all_gather_object(output, value):\n"
        "    all_gather(output, value)\n",
        object_collective_namespace,
    )
    native_all_gather_object = object_collective_namespace[
        "native_all_gather_object"
    ]

    @wraps(native_all_gather_object)
    def mindspeed_outer_all_gather_wrapper(output, value):
        raise AssertionError("args is not initialized")

    monkeypatch.setattr(distributed.dist, "barrier", mindspeed_barrier_wrapper)
    monkeypatch.setattr(
        distributed.dist,
        "all_gather_object",
        mindspeed_outer_all_gather_wrapper,
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
    assert (
        native_all_gather_object.__globals__["all_gather"]
        is mindspeed_inner_all_gather_wrapper
    )


def test_rank_payloads_can_be_gathered_through_shared_json_files(tmp_path):
    contexts = [
        distributed.DistributedContext(
            rank=rank,
            local_rank=rank,
            world_size=2,
            device=f"npu:{rank}",
        )
        for rank in range(2)
    ]
    exchange_dir = tmp_path / "rank_payloads" / "run-123"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                context.all_gather_json_files,
                {"rank": context.rank, "value": context.rank + 10},
                exchange_dir,
                2.0,
                0.01,
            )
            for context in contexts
        ]
        results = [future.result() for future in futures]

    expected = [
        {"rank": 0, "value": 10},
        {"rank": 1, "value": 11},
    ]
    assert results == [expected, expected]
    assert sorted(path.name for path in exchange_dir.glob("rank*.json")) == [
        "rank00.json",
        "rank01.json",
    ]

