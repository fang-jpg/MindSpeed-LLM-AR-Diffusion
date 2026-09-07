#!/bin/bash

source examples/fsdp2/env_config.sh

NPUS_PER_NODE=${NPUS_PER_NODE:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-42323}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MODEL_PATH=${MODEL_PATH:-/path/to/qwen3-diffusion-checkpoint}

torchrun \
  --nproc_per_node "${NPUS_PER_NODE}" \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  inference_fsdp2_diffusion.py \
  examples/fsdp2/qwen3_diffusion/inference_qwen3_diffusion_810.yaml \
  --model.model_name_or_path "${MODEL_PATH}" \
  --parallel.fsdp_size "${NPUS_PER_NODE}" \
  "$@"
