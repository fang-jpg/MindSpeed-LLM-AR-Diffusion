#!/bin/bash
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
# Phase 6: Evaluate script for Qwen3-1.7B-Diffusion.
#
# Evaluates the model on a test dataset (MMLU-style CSV) using one of
# three inference modes: ar, diffusion, or linear_spec.
#
# Usage:
#   # AR mode (default)
#   bash evaluate_qwen3_diffusion.sh ar
#
#   # Diffusion mode
#   bash evaluate_qwen3_diffusion.sh diffusion
#
#   # Linear-Spec mode
#   bash evaluate_qwen3_diffusion.sh linear_spec

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${SCRIPT_DIR}/../../.."  # repo root: MindSpeed-LLM/

# ── Configuration (override via env vars) ──
MODEL_PATH=${MODEL_PATH:-/home/c00633840/ckpt/phase4_joint_cpt/}
DATA_PATH=${DATA_PATH:-/home/c00633840/testdata/mmlu/data/test/}
DEVICE=${DEVICE:-""}           # auto-detect if empty
DTYPE=${DTYPE:-bfloat16}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-10}
BLOCK_LENGTH=${BLOCK_LENGTH:-32}
TEMPERATURE=${TEMPERATURE:-0.0}
THRESHOLD=${THRESHOLD:-0.0}
VERBOSE=${VERBOSE:-false}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-}  # empty = all samples

# ── Parse arguments ──
MODE=${1:-ar}

if [[ "$MODE" != "ar" && "$MODE" != "diffusion" && "$MODE" != "linear_spec" ]]; then
    echo "Usage: $0 <ar|diffusion|linear_spec>"
    echo ""
    echo "Environment variables:"
    echo "  MODEL_PATH       - Path to model checkpoint (default: /home/c00633840/ckpt/phase4_joint_cpt/)"
    echo "  DATA_PATH        - Path to test data (default: /home/c00633840/testdata/mmlu/data/test/)"
    echo "  MAX_NEW_TOKENS   - Max tokens to generate per sample (default: 10)"
    echo "  BLOCK_LENGTH     - Block size for diffusion/linear_spec (default: 32)"
    echo "  TEMPERATURE      - Sampling temperature (default: 0.0)"
    echo "  THRESHOLD        - Confidence threshold (default: 0.0)"
    echo "  DTYPE            - Model dtype: bfloat16|float16|float32 (default: bfloat16)"
    echo "  VERBOSE          - Print per-sample details: true|false (default: false)"
    echo "  MAX_EVAL_SAMPLES - Limit samples per subject for debugging (default: all)"
    exit 1
fi

echo "=== Qwen3-1.7B Diffusion Evaluation ==="
echo "  Mode:            ${MODE}"
echo "  Model:           ${MODEL_PATH}"
echo "  Data path:       ${DATA_PATH}"
echo "  Max new tokens:  ${MAX_NEW_TOKENS}"
echo "  Block length:    ${BLOCK_LENGTH}"
echo "  Temperature:     ${TEMPERATURE}"
echo "  Threshold:       ${THRESHOLD}"
echo "  Max eval samples: ${MAX_EVAL_SAMPLES:-all}"
echo ""

# ── Build command ──
CMD="python examples/fsdp2/qwen3_diffusion/evaluate_qwen3_diffusion.py \
    --model-path ${MODEL_PATH} \
    --data-path ${DATA_PATH} \
    --mode ${MODE} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --block-length ${BLOCK_LENGTH} \
    --temperature ${TEMPERATURE} \
    --threshold ${THRESHOLD} \
    --dtype ${DTYPE}"

if [[ -n "${DEVICE}" ]]; then
    CMD="${CMD} --device ${DEVICE}"
fi

if [[ "${VERBOSE}" == "true" ]]; then
    CMD="${CMD} --verbose"
fi

if [[ -n "${MAX_EVAL_SAMPLES}" ]]; then
    CMD="${CMD} --max-eval-samples ${MAX_EVAL_SAMPLES}"
fi

# Optional: inject config_diffusion.json if it exists in the model directory
CONFIG_DIFF="${MODEL_PATH}/config_diffusion.json"
if [[ -f "${CONFIG_DIFF}" ]]; then
    CMD="${CMD} --config-diffusion ${CONFIG_DIFF}"
fi

echo "  Command: ${CMD}"
echo ""

# ── Run evaluation ──
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/evaluate_qwen3_diffusion_${MODE}_${TIMESTAMP}.log"

echo "  Log: ${LOG_FILE}"
echo ""

eval ${CMD} 2>&1 | tee -a "${LOG_FILE}"
