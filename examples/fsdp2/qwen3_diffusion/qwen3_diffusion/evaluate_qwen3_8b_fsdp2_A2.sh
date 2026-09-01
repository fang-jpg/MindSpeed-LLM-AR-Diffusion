#!/bin/bash

# FSDP2/HuggingFace equivalent of:
# examples/mcore/qwen3/evaluate_qwen3_1point7b_ptd.sh
set -eo pipefail

source examples/fsdp2/env_config.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# HuggingFace model directory. It should contain config.json and
# model.safetensors/model.safetensors.index.json.
MODEL_PATH=${MODEL_PATH:-/share/dataset/x00840191/models/qwen31.7}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}}
DATA_PATH=${DATA_PATH:-/home/c00633840/testdata/mmlu/data/test}
OUTPUT_PATH=${OUTPUT_PATH:-./outputs/qwen3_1point7b_fsdp2_mmlu.json}

MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
NPUS_PER_NODE=${NPUS_PER_NODE:-8}
FSDP_SIZE=${FSDP_SIZE:-$((NPUS_PER_NODE * NNODES))}

mkdir -p logs "$(dirname "${OUTPUT_PATH}")"

torchrun \
    --nproc_per_node "${NPUS_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    evaluation_fsdp2.py examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A2.yaml \
    --model.model_name_or_path "${MODEL_PATH}" \
    --model.tokenizer_name_or_path "${TOKENIZER_PATH}" \
    --model.model_id qwen3 \
    --model.trust_remote_code false \
    --model.train_from_scratch false \
    --model.init_model_with_meta_device false \
    --model.use_fast_tokenizer false \
    --parallel.fsdp_size "${FSDP_SIZE}" \
    --parallel.tp_size 1 \
    --parallel.cp_size 1 \
    --parallel.ep_size 1 \
    --parallel.ep_fsdp_size 1 \
    --parallel.recompute false \
    --inference.max_new_tokens 256 \
    --inference.do_sample false \
    --inference.enable_thinking false \
    --evaluation.task mmlu \
    --evaluation.task_data_path "${DATA_PATH}" \
    --evaluation.use_chat_template true \
    --evaluation.seed 42 \
    --evaluation.output_path "${OUTPUT_PATH}" 