#!/bin/bash
set -euo pipefail

source examples/fsdp2/env_config.sh

NPUS_PER_NODE=${NPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-80.5.5.133}
MASTER_PORT=${MASTER_PORT:-6502}
NNODES=${NNODES:-2}
MODEL_PATH=${MODEL_PATH:-/mnt/pipeline-data/f50058111/model_weights/Qwen3-1.7B-Base}
DOPD_DATA_PATH=${DOPD_DATA_PATH:-/mnt/pipeline-data/f50058111/dataset/opdlm_train_data/opdlm_train.json}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/pipeline-data/f50058111/model_train/opd/qwen3_1.7b_dopd_paper_2node_16card}
export PYTHONUNBUFFERED=1

if [[ -z "${NODE_RANK:-}" ]]; then
  HOST_IPV4=" $(hostname -I 2>/dev/null || true) "
  if [[ "${HOST_IPV4}" == *" 80.5.5.133 "* ]]; then
    NODE_RANK=0
  elif [[ "${HOST_IPV4}" == *" 80.5.5.118 "* ]]; then
    NODE_RANK=1
  else
    echo "无法自动判断 NODE_RANK。请在 80.5.5.133 上设置 NODE_RANK=0，在 80.5.5.118 上设置 NODE_RANK=1。" >&2
    exit 2
  fi
fi
if [[ "${NODE_RANK}" != "0" && "${NODE_RANK}" != "1" ]]; then
  echo "NODE_RANK 必须为 0（80.5.5.133）或 1（80.5.5.118）。" >&2
  exit 2
fi

if [[ ! -f "${DOPD_DATA_PATH}" ]]; then
  echo "未找到官方 OPDLM 数据集：${DOPD_DATA_PATH}" >&2
  exit 2
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "未找到 Qwen3-1.7B-Base 模型目录：${MODEL_PATH}" >&2
  exit 2
fi
python -c 'import math_verify; assert math_verify' >/dev/null

mkdir -p "${OUTPUT_DIR}/logs"
LOG_FILE="${OUTPUT_DIR}/logs/node_${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "开始论文对齐的 d-OPD 训练：node_rank=${NODE_RANK}/${NNODES}，主节点=${MASTER_ADDR}:${MASTER_PORT}"
echo "训练数据：${DOPD_DATA_PATH}"
echo "模型目录：${MODEL_PATH}"
echo "输出目录：${OUTPUT_DIR}"

torchrun \
  --nproc_per_node "${NPUS_PER_NODE}" \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  train_dopd_fsdp2.py \
  examples/fsdp2/qwen3_dopd/dopd_qwen3_1.7b_fsdp2_A2.yaml \
  --model.model_name_or_path="${MODEL_PATH}" \
  --dopd.teacher_model_name_or_path="${MODEL_PATH}" \
  --dopd.train_data_path="${DOPD_DATA_PATH}" \
  --training.output_dir="${OUTPUT_DIR}"

