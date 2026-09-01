#!/bin/bash

source examples/fsdp2/env_config.sh

MODEL_PATH=${MODEL_PATH:-/home/data/Qwen3-8B}
NPUS_PER_NODE=${NPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
FSDP_SIZE=${FSDP_SIZE:-$((NPUS_PER_NODE * NNODES))}

torchrun \
    --nproc_per_node "${NPUS_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    inference_fsdp2.py examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A2.yaml \
    --model.model_name_or_path "${MODEL_PATH}" \
    --model.init_model_with_meta_device true \
    --parallel.fsdp_size "${FSDP_SIZE}" \
    --parallel.recompute false \
    --inference.infer_backend huggingface \
    --inference.max_new_tokens 512 \
    --inference.do_sample false
