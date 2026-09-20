#!/usr/bin/env bash

# 1. 激活共享盘 CANN 9.1
source /share/f50058111/activate_cann_910b_9.1.sh

# 2. MindSpeed FSDP2 环境
source examples/fsdp2/env_config.sh

# 3. 单机 8 卡分布式配置
NPUS_PER_NODE=8
NNODES=1
NODE_RANK=0

MASTER_ADDR=127.0.0.1
MASTER_PORT=6501

# 4. 路径
MODEL_PATH=/share/model_weights/Qwen3-1.7B-Base

OUTPUT_DIR=/share/dataset/x00840191/model_train/ckpt/qwen3_1.7b_ar_cpt_baseline_0920

LOG_DIR=/share/f50058111/logs
LOG_FILE=${LOG_DIR}/qwen3_1.7b_ar_cpt_baseline_0920.log

CONFIG=examples/fsdp2/qwen3/ar_cpt_qwen3_1.7b_baseline_fsdp2_A2.yaml

ENTRYPOINT=examples/fsdp2/qwen3/train_ar_cpt_fsdp2.py

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "======================================"
echo "Native AR CPT baseline"
echo "NPUs       : ${NPUS_PER_NODE}"
echo "NNODES     : ${NNODES}"
echo "NODE_RANK  : ${NODE_RANK}"
echo "MASTER     : ${MASTER_ADDR}:${MASTER_PORT}"
echo "MODEL      : ${MODEL_PATH}"
echo "CONFIG     : ${CONFIG}"
echo "OUTPUT     : ${OUTPUT_DIR}"
echo "LOG        : ${LOG_FILE}"
echo "======================================"

torchrun \
    --nproc_per_node ${NPUS_PER_NODE} \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    ${ENTRYPOINT} \
    ${CONFIG} \
    --model.model_name_or_path "${MODEL_PATH}" \
    --training.output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_FILE}"
