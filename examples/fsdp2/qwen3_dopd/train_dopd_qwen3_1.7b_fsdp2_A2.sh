#!/bin/bash
set -euo pipefail

source examples/fsdp2/env_config.sh

NPUS_PER_NODE=${NPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-6502}
STUDENT_MODEL_PATH=${STUDENT_MODEL_PATH:-${MODEL_PATH:-/mnt/pipeline-data/f50058111/model_weights/Qwen3-1.7B-a2d-init}}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-/mnt/pipeline-data/f50058111/model_weights/Qwen3-1.7B}
DOPD_DATA_PATH=${DOPD_DATA_PATH:-/mnt/pipeline-data/f50058111/dataset/opdlm_train_data/opdlm_train.json}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/pipeline-data/f50058111/model_train/opd/qwen3_1.7b_dopd_paper_1node_8card_1019}
export PYTHONUNBUFFERED=1

if [[ "${NPUS_PER_NODE}" != "8" ]]; then
  echo "论文对齐的单机配置要求 NPUS_PER_NODE=8，当前为 ${NPUS_PER_NODE}。" >&2
  exit 2
fi
if [[ ! -f "${DOPD_DATA_PATH}" ]]; then
  echo "未找到官方 OPDLM 数据集：${DOPD_DATA_PATH}" >&2
  exit 2
fi
if [[ ! -f "${STUDENT_MODEL_PATH}/config.json" ]]; then
  echo "未找到 a2d-init 学生模型配置：${STUDENT_MODEL_PATH}/config.json" >&2
  exit 2
fi
if [[ ! -f "${TEACHER_MODEL_PATH}/config.json" ]]; then
  echo "未找到 Qwen3-1.7B 教师模型配置：${TEACHER_MODEL_PATH}/config.json" >&2
  exit 2
fi
python -c 'import json,sys; s=json.load(open(sys.argv[1])); t=json.load(open(sys.argv[2])); assert s.get("model_type")=="a2d-qwen3", "学生 model_type 应为 a2d-qwen3，实际为 %r" % s.get("model_type"); assert t.get("model_type")=="qwen3", "教师 model_type 应为 qwen3，实际为 %r" % t.get("model_type")' "${STUDENT_MODEL_PATH}/config.json" "${TEACHER_MODEL_PATH}/config.json"
python -c 'import math_verify; assert math_verify' >/dev/null

mkdir -p "${OUTPUT_DIR}/logs"
LOG_FILE="${OUTPUT_DIR}/logs/node_0_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "开始论文对齐的 d-OPD 单机训练：8 张 A2，主节点=${MASTER_ADDR}:${MASTER_PORT}"
echo "训练数据：${DOPD_DATA_PATH}"
echo "学生模型（A2D 初始化）：${STUDENT_MODEL_PATH}"
echo "教师模型（AR）：${TEACHER_MODEL_PATH}"
echo "输出目录：${OUTPUT_DIR}"

torchrun \
  --nproc_per_node "${NPUS_PER_NODE}" \
  --nnodes 1 \
  --node_rank 0 \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  train_dopd_fsdp2.py \
  examples/fsdp2/qwen3_dopd/dopd_qwen3_1.7b_fsdp2_A2.yaml \
  --model.model_name_or_path="${STUDENT_MODEL_PATH}" \
  --dopd.teacher_model_name_or_path="${TEACHER_MODEL_PATH}" \
  --dopd.train_data_path="${DOPD_DATA_PATH}" \
  --training.output_dir="${OUTPUT_DIR}"

