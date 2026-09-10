#!/bin/bash
set -euo pipefail

source examples/fsdp2/env_config.sh

NPUS_PER_NODE=${NPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6511}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$((NPUS_PER_NODE * NNODES))

MODEL_PATH=${MODEL_PATH:-/home/c00633840/models/Qwen3-1.7B-Base/}
TOKENIZER_PATH=${TOKENIZER_PATH:-"${MODEL_PATH}"}
INPUT_TEXT=${INPUT_TEXT:-hello}
OUTPUT_DIR=${OUTPUT_DIR:-./output/qwen3_text}
FSDP_SIZE=${FSDP_SIZE:-"${WORLD_SIZE}"}

DISTRIBUTED_ARGS=(
    --nproc_per_node "${NPUS_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

mkdir -p logs "${OUTPUT_DIR}"

torchrun "${DISTRIBUTED_ARGS[@]}" evaluation_fsdp2.py \
    examples/fsdp2/qwen3/evaluate_qwen3_text_fsdp2.yaml \
    --model.model_name_or_path "${MODEL_PATH}" \
    --model.tokenizer_name_or_path "${TOKENIZER_PATH}" \
    --parallel.fsdp_size "${FSDP_SIZE}" \
    --evaluation.input_text "${INPUT_TEXT}" \
    --evaluation.output_dir "${OUTPUT_DIR}" \
    2>&1 | tee logs/evaluate_qwen3_text_fsdp2.log
