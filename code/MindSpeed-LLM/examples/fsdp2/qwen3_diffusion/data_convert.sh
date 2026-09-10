#!/bin/bash
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
# Phase 4: Data preprocessing for Qwen3-Diffusion joint CPT.
#
# Reuses the same pretraining data pipeline as Phase 0 (Alpaca or similar
# plain text). The <mask> token is NOT added at preprocessing time — it
# only appears inside the model's forward_process at training time, so
# the data bin/idx files are identical to standard Qwen3 CPT data.

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${SCRIPT_DIR}/../../.."  # repo root: MindSpeed-LLM/

# ── Paths (override via env vars if needed) ──
INPUT_PARQUET=${INPUT_PARQUET:-/data/dataset/alpaca.parquet}
TOKENIZER_PATH=${TOKENIZER_PATH:-/data/models/Qwen3-1.7B-Base/}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-/data/dataset/qwen3_diff_cpt}
WORKERS=${WORKERS:-4}

echo "=== Phase 4: Data preprocessing for Qwen3-Diffusion CPT ==="
echo "  Input:       ${INPUT_PARQUET}"
echo "  Tokenizer:   ${TOKENIZER_PATH}"
echo "  Output:      ${OUTPUT_PREFIX}_text_document.bin/.idx"
echo "  Workers:     ${WORKERS}"
echo ""

python preprocess_data.py \
    --input "${INPUT_PARQUET}" \
    --tokenizer-name-or-path "${TOKENIZER_PATH}" \
    --tokenizer-type PretrainedFromHF \
    --handler-name GeneralPretrainHandler \
    --output-prefix "${OUTPUT_PREFIX}" \
    --json-keys text \
    --workers "${WORKERS}"

echo ""
echo "=== Done. Produced: ==="
ls -lh "${OUTPUT_PREFIX}"_text_document.{bin,idx}
