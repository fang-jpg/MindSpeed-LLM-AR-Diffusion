#!/bin/bash
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
# Phase 5: Three-mode inference launch script for Qwen3-1.7B-Diffusion.
#
# Usage:
#   # AR mode (single card)
#   bash inference_qwen3_diffusion.sh ar "Explain quantum computing."
#
#   # Diffusion mode
#   bash inference_qwen3_diffusion.sh diffusion "Explain quantum computing."
#
#   # Linear-Spec mode
#   bash inference_qwen3_diffusion.sh linear_spec "Explain quantum computing."

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${SCRIPT_DIR}/../../.."  # repo root: MindSpeed-LLM/

# ── Configuration (override via env vars) ──
MODEL_PATH=${MODEL_PATH:-/share/dataset/x00840191/model_train/ckpt/phase4_cpt_ar_duffision_0909/models/global_step_500/}
DEVICE=${DEVICE:-""}           # auto-detect if empty
DTYPE=${DTYPE:-bfloat16}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
BLOCK_LENGTH=${BLOCK_LENGTH:-8}
TEMPERATURE=${TEMPERATURE:-0.7}
THRESHOLD=${THRESHOLD:-0.0}
TOPK=${TOPK:-50}
TOPP=${TOPP:-0.0}

# ── Parse arguments ──
MODE=${1:-ar}
PROMPT=${2:-"The meaning of life is"}

if [[ "$MODE" != "ar" && "$MODE" != "diffusion" && "$MODE" != "linear_spec" ]]; then
    echo "Usage: $0 <ar|diffusion|linear_spec> [prompt]"
    exit 1
fi

echo "=== Qwen3-1.7B Diffusion Inference ==="
echo "  Mode:   ${MODE}"
echo "  Model:  ${MODEL_PATH}"
echo "  Prompt: ${PROMPT}"

# ── Run inference ──
CMD="python examples/fsdp2/qwen3_diffusion/inference_qwen3_diffusion.py \
    --model-path ${MODEL_PATH} \
    --mode ${MODE} \
    --prompt \"${PROMPT}\" \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --block-length ${BLOCK_LENGTH} \
    --temperature ${TEMPERATURE} \
    --threshold ${THRESHOLD} \
    --dtype ${DTYPE} \
    --top-k ${TOPK} \
    --top-p ${TOPP}"

if [[ -n "${DEVICE}" ]]; then
    CMD="${CMD} --device ${DEVICE}"
fi

# # Optional: inject config_diffusion.json if it exists in the model directory
# CONFIG_DIFF="/home/c00633840/MindSpeed-LLM/examples/fsdp2/qwen3_diffusion/config_diffusion.json"
# if [[ -f "${CONFIG_DIFF}" ]]; then
#     CMD="${CMD} --config-diffusion ${CONFIG_DIFF}"
# fi

echo "  Command: ${CMD}"
echo ""

eval ${CMD} 2>&1 | tee -a /tmp/qwen3_diffusion_inference_${MODE}.log