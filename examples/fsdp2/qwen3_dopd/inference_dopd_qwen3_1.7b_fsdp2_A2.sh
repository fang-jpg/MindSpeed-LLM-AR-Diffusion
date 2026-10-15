#!/bin/bash
# Inspect d-OPD block-size/threshold decoding on one 8-card A2 node.
#
# Common overrides:
#   BLOCK_SIZES=1,2,4,8,16 THRESHOLDS=fixed,0.7,0.8,0.9 \
#   MAX_NEW_TOKENS=128 PROMPT='Explain why the sky is blue.' bash "$0"
#
# If dopd_config.json records an old Base-model path, set MODEL_PATH explicitly.

set -euo pipefail

# env_config.sh appends to PYTHONPATH, so define it for shells launched with -u.
export PYTHONPATH=${PYTHONPATH:-}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

source "${REPO_ROOT}/examples/fsdp2/env_config.sh"

NPUS_PER_NODE=${NPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6503}
NNODES=1
NODE_RANK=0

CHECKPOINT_PATH=${CHECKPOINT_PATH:-/share/dataset/x00840191/model_train/ckpt/qwen3_1.7b_dopd_1010/checkpoints/global_step_483_final}
MODEL_PATH=${MODEL_PATH:-}
MASK_TOKEN_ID=${MASK_TOKEN_ID:-151669}
RUN_DIR=$(dirname "$(dirname "${CHECKPOINT_PATH}")")
OUTPUT_JSON=${OUTPUT_JSON:-${RUN_DIR}/dopd_decode_analysis.json}
LOG_PATH=${LOG_PATH:-${RUN_DIR}/dopd_decode_analysis.log}

PROMPT=${PROMPT:-"Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for \$2 per fresh duck egg. How much in dollars does she make every day?"}
BLOCK_SIZES=${BLOCK_SIZES:-4,8,16}
THRESHOLDS=${THRESHOLDS:-fixed,0.8,0.9}
STEPS_PER_BLOCK=${STEPS_PER_BLOCK:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
SEED_MODE=${SEED_MODE:-ar}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_K=${TOP_K:-0}
TOP_P=${TOP_P:-1.0}
SEED=${SEED:-42}
WARMUP=${WARMUP:-0}
REPEATS=${REPEATS:-1}
USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE:-0}
ENABLE_THINKING=${ENABLE_THINKING:-0}
IGNORE_EOS=${IGNORE_EOS:-0}

mkdir -p "$(dirname "${OUTPUT_JSON}")" "$(dirname "${LOG_PATH}")"

ARGS=(
  --checkpoint "${CHECKPOINT_PATH}"
  --output-json "${OUTPUT_JSON}"
  --prompt "${PROMPT}"
  --block-sizes "${BLOCK_SIZES}"
  --thresholds "${THRESHOLDS}"
  --steps-per-block "${STEPS_PER_BLOCK}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --mask-token-id "${MASK_TOKEN_ID}"
  --seed-mode "${SEED_MODE}"
  --temperature "${TEMPERATURE}"
  --top-k "${TOP_K}"
  --top-p "${TOP_P}"
  --seed "${SEED}"
  --warmup "${WARMUP}"
  --repeats "${REPEATS}"
)

if [[ -n "${MODEL_PATH}" ]]; then
  ARGS+=(--model-path "${MODEL_PATH}")
fi
if [[ "${USE_CHAT_TEMPLATE}" == "1" ]]; then
  ARGS+=(--use-chat-template)
fi
if [[ "${ENABLE_THINKING}" == "1" ]]; then
  ARGS+=(--enable-thinking)
fi
if [[ "${IGNORE_EOS}" == "1" ]]; then
  ARGS+=(--ignore-eos)
fi

echo "Checkpoint : ${CHECKPOINT_PATH}"
echo "Base model : ${MODEL_PATH:-auto from dopd_config.json}"
echo "Blocks     : ${BLOCK_SIZES}"
echo "Thresholds : ${THRESHOLDS}"
echo "Output     : ${OUTPUT_JSON}"
echo "Log        : ${LOG_PATH}"

torchrun \
  --nproc_per_node "${NPUS_PER_NODE}" \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  examples/fsdp2/qwen3_dopd/inference_dopd_decode_analysis.py \
  "${ARGS[@]}" 2>&1 | tee "${LOG_PATH}"
