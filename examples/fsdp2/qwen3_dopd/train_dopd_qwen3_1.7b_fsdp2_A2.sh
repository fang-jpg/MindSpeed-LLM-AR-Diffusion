#!/bin/bash
set -euo pipefail

source examples/fsdp2/env_config.sh

NPUS_PER_NODE=${NPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-6502}
NNODES=1
NODE_RANK=0

torchrun \
  --nproc_per_node "${NPUS_PER_NODE}" \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  train_dopd_fsdp2.py \
  examples/fsdp2/qwen3_dopd/dopd_qwen3_1.7b_fsdp2_A2.yaml

