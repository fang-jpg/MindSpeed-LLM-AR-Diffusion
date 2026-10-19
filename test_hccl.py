import os
import torch
import torch_npu
import torch.distributed as dist
from datetime import timedelta

local_rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(local_rank)

dist.init_process_group(
    backend="hccl",
    timeout=timedelta(seconds=120),
)

rank = dist.get_rank()
world_size = dist.get_world_size()

x = torch.tensor(
    [rank + 1.0],
    device=f"npu:{local_rank}"
)

print(f"[rank {rank}] before all_reduce", flush=True)

dist.all_reduce(x)

print(f"[rank {rank}] result={x.item()}", flush=True)

dist.destroy_process_group()