#!/usr/bin/env bash
# Reproducible AR vs. block-size/denoising-step experiment launcher.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

LM_EVAL_ROOT=${LM_EVAL_ROOT:-/home/f50058111/lm-eval-harness}
HF_CACHE_ROOT=${HF_CACHE_ROOT:-/home/f50058111/huggingface_cache}
MODEL_PATH=${MODEL_PATH:-/data2/model_train/ckpt/phase4_cpt_ar_diffusion_0923_resume/models/global_step_102500}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/outputs/ar_diffusion_gsm8k}
PHASE=${PHASE:-prepare}
OVERWRITE=${OVERWRITE:-0}
NUM_NPUS=${NUM_NPUS:-8}
RANK_FILE_TIMEOUT_SECONDS=${RANK_FILE_TIMEOUT_SECONDS:-3600}

if ! [[ "$NUM_NPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_NPUS must be a positive integer, got: $NUM_NPUS" >&2
  exit 2
fi

if [[ -x "$LM_EVAL_ROOT/.venv-diffusion/bin/python" ]]; then
  PYTHON_BIN=${PYTHON_BIN:-$LM_EVAL_ROOT/.venv-diffusion/bin/python}
else
  PYTHON_BIN=${PYTHON_BIN:-python}
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$HF_CACHE_ROOT"
export HF_DATASETS_CACHE="$HF_CACHE_ROOT/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

DATA_DIR="$OUTPUT_ROOT/data"
PROMPTS_FILE="$DATA_DIR/speed_prompts_100.jsonl"
FEWSHOT_FILE="$DATA_DIR/fewshot_examples.json"
PROTOCOL_FILE="$DATA_DIR/gsm8k_protocol.json"
SPEED_DIR="$OUTPUT_ROOT/speed"
QUALITY_DIR="$OUTPUT_ROOT/gsm8k"

CONFIGS=(
  "B04-S01:4:1"
  "B04-S02:4:2"
  "B04-S04:4:4"
  "B08-S02:8:2"
  "B08-S04:8:4"
  "B08-S08:8:8"
  "B16-S04:16:4"
  "B16-S08:16:8"
  "B16-S16:16:16"
  "B32-S08:32:8"
  "B32-S16:32:16"
  "B32-S32:32:32"
)

run_experiment_python() {
  local script=$1
  shift
  echo "> Launching $(basename "$script") with $NUM_NPUS process(es)"
  if (( NUM_NPUS > 1 )); then
    "$PYTHON_BIN" -m torch.distributed.run \
      --standalone \
      --nnodes=1 \
      --nproc_per_node="$NUM_NPUS" \
      "$script" "$@"
  else
    "$PYTHON_BIN" "$script" "$@"
  fi
}

prepare_data() {
  local overwrite_args=()
  if [[ "$OVERWRITE" == "1" ]]; then
    overwrite_args+=(--overwrite)
  elif [[ -f "$PROMPTS_FILE" && -f "$FEWSHOT_FILE" && -f "$PROTOCOL_FILE" ]]; then
    echo "> Fixed 100-prompt subset already exists: $PROMPTS_FILE"
    return
  fi
  "$PYTHON_BIN" "$SCRIPT_DIR/prepare_gsm8k_data.py" \
    --cache-dir "$HF_CACHE_ROOT" \
    --output-dir "$DATA_DIR" \
    --num-speed-prompts 100 \
    --num-fewshot 5 \
    --seed 42 \
    "${overwrite_args[@]}"
}

check_model() {
  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Model directory not found: $MODEL_PATH" >&2
    echo "Set MODEL_PATH to the actual exported step-102500 model directory." >&2
    exit 2
  fi
}

run_speed() {
  check_model
  prepare_data
  mkdir -p "$SPEED_DIR"
  local ar_json="$SPEED_DIR/AR-BASE.json"
  if [[ "$OVERWRITE" == "1" || ! -f "$ar_json" ]]; then
    run_experiment_python "$SCRIPT_DIR/benchmark_decode.py" \
      --model-path "$MODEL_PATH" \
      --prompts-file "$PROMPTS_FILE" \
      --output-json "$ar_json" \
      --mode ar \
      --max-new-tokens 128 \
      --temperature 0 \
      --warmup 5 \
      --repeats 20 \
      --rank-file-timeout-seconds "$RANK_FILE_TIMEOUT_SECONDS" \
      --seed 42
  else
    echo "> Skipping existing speed result: $ar_json"
  fi

  local spec exp_id block_length steps output_json
  for spec in "${CONFIGS[@]}"; do
    IFS=: read -r exp_id block_length steps <<< "$spec"
    output_json="$SPEED_DIR/$exp_id.json"
    if [[ "$OVERWRITE" != "1" && -f "$output_json" ]]; then
      echo "> Skipping existing speed result: $output_json"
      continue
    fi
    run_experiment_python "$SCRIPT_DIR/benchmark_decode.py" \
      --model-path "$MODEL_PATH" \
      --prompts-file "$PROMPTS_FILE" \
      --output-json "$output_json" \
      --ar-baseline-json "$ar_json" \
      --mode diffusion \
      --block-length "$block_length" \
      --steps-per-block "$steps" \
      --seed-mode none \
      --max-new-tokens 128 \
      --temperature 0 \
      --warmup 5 \
      --repeats 20 \
      --rank-file-timeout-seconds "$RANK_FILE_TIMEOUT_SECONDS" \
      --seed 42
  done
}

run_smoke() {
  check_model
  prepare_data
  local result_dir="$OUTPUT_ROOT/smoke/B08-S04"
  local overwrite_args=()
  if [[ "$OVERWRITE" == "1" ]]; then
    overwrite_args+=(--overwrite)
  elif [[ -f "$result_dir/summary.json" ]]; then
    echo "> Skipping existing smoke result: $result_dir/summary.json"
    return
  fi

  run_experiment_python "$SCRIPT_DIR/evaluate_gsm8k_diffusion.py" \
    --model-path "$MODEL_PATH" \
    --cache-dir "$HF_CACHE_ROOT" \
    --fewshot-file "$FEWSHOT_FILE" \
    --data-protocol "$PROTOCOL_FILE" \
    --output-dir "$result_dir" \
    --mode diffusion \
    --block-length 8 \
    --steps-per-block 4 \
    --seed-mode none \
    --num-fewshot 5 \
    --max-new-tokens 512 \
    --temperature 0 \
    --seed 42 \
    --max-eval-samples 16 \
    --resume \
    "${overwrite_args[@]}"
}

run_quality_case() {
  local exp_id=$1
  local mode=$2
  local block_length=$3
  local steps=$4
  local result_dir="$QUALITY_DIR/$exp_id"
  local overwrite_args=()
  if [[ "$OVERWRITE" == "1" ]]; then
    overwrite_args+=(--overwrite)
  elif [[ -f "$result_dir/summary.json" ]]; then
    echo "> Skipping existing GSM8K result: $result_dir/summary.json"
    return
  fi

  local mode_args=(--mode "$mode")
  if [[ "$mode" == "diffusion" ]]; then
    mode_args+=(
      --block-length "$block_length"
      --steps-per-block "$steps"
      --seed-mode none
    )
  fi
  run_experiment_python "$SCRIPT_DIR/evaluate_gsm8k_diffusion.py" \
    --model-path "$MODEL_PATH" \
    --cache-dir "$HF_CACHE_ROOT" \
    --fewshot-file "$FEWSHOT_FILE" \
    --data-protocol "$PROTOCOL_FILE" \
    --output-dir "$result_dir" \
    "${mode_args[@]}" \
    --num-fewshot 5 \
    --max-new-tokens 512 \
    --temperature 0 \
    --seed 42 \
    --resume \
    "${overwrite_args[@]}"
}

run_quality() {
  check_model
  prepare_data
  mkdir -p "$QUALITY_DIR"
  run_quality_case "AR-BASE" "ar" 1 1
  local spec exp_id block_length steps
  for spec in "${CONFIGS[@]}"; do
    IFS=: read -r exp_id block_length steps <<< "$spec"
    run_quality_case "$exp_id" "diffusion" "$block_length" "$steps"
  done
}

summarize_results() {
  "$PYTHON_BIN" "$SCRIPT_DIR/summarize_block_step_results.py" \
    --results-root "$OUTPUT_ROOT"
}

case "$PHASE" in
  prepare)
    prepare_data
    ;;
  smoke)
    run_smoke
    ;;
  speed)
    run_speed
    ;;
  quality)
    run_quality
    ;;
  all)
    run_speed
    run_quality
    summarize_results
    ;;
  summarize)
    summarize_results
    ;;
  *)
    echo "Unknown PHASE=$PHASE; use prepare, smoke, speed, quality, summarize, or all." >&2
    exit 2
    ;;
esac

echo "> Completed phase '$PHASE'. Results: $OUTPUT_ROOT"
