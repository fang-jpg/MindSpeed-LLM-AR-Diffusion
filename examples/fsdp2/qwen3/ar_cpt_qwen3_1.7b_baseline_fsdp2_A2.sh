#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# Optional site-specific CANN activation. Override or clear CANN_ENV_SCRIPT as needed.
CANN_ENV_SCRIPT="${CANN_ENV_SCRIPT-/share/f50058111/activate_cann_910b_9.1.sh}"
if [[ -n "${CANN_ENV_SCRIPT}" && -f "${CANN_ENV_SCRIPT}" ]]; then
    # shellcheck disable=SC1090
    source "${CANN_ENV_SCRIPT}"
fi

: "${PYTHONPATH:=}"
source examples/fsdp2/env_config.sh

NPUS_PER_NODE="${NPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-6501}"

WORLD_SIZE=$((NPUS_PER_NODE * NNODES))
FSDP_SIZE=8
if (( WORLD_SIZE % FSDP_SIZE != 0 )); then
    echo "ERROR: WORLD_SIZE=${WORLD_SIZE} must be divisible by fsdp_size=${FSDP_SIZE}." >&2
    exit 2
fi

MODEL_PATH="${MODEL_PATH:-/share/model_weights/Qwen3-1.7B-Base}"
OUTPUT_DIR="${OUTPUT_DIR:-/share/dataset/x00840191/model_train/ckpt/qwen3_1.7b_ar_cpt_baseline_0916}"
LOG_DIR="${LOG_DIR:-/share/f50058111/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/qwen3_1.7b_ar_cpt_baseline_0916_rank${NODE_RANK}.log}"
CONFIG="${SCRIPT_DIR}/ar_cpt_qwen3_1.7b_baseline_fsdp2_A2.yaml"
ENTRYPOINT="${SCRIPT_DIR}/train_ar_cpt_fsdp2.py"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "ERROR: MODEL_PATH does not contain config.json: ${MODEL_PATH}" >&2
    exit 2
fi
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "Native AR CPT baseline"
echo "  model       : ${MODEL_PATH}"
echo "  world size  : ${WORLD_SIZE} (${NNODES} node(s) x ${NPUS_PER_NODE} NPU(s))"
echo "  output      : ${OUTPUT_DIR}"
echo "  log         : ${LOG_FILE}"

torchrun \
    --nproc_per_node "${NPUS_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${ENTRYPOINT}" \
    "${CONFIG}" \
    --model.model_name_or_path "${MODEL_PATH}" \
    --training.output_dir "${OUTPUT_DIR}" \
    "$@" \
    2>&1 | tee "${LOG_FILE}"
