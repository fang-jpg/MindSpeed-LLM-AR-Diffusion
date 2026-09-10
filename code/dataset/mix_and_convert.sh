#!/bin/bash
# ============================================================
# 多数据集按比例混合转换脚本
# 将多个 parquet 数据集按指定概率比例 interleave 混合，
# 生成一个统一的 .bin / .idx 文件
# ============================================================

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# 项目根目录（MindSpeed-LLM 所在目录）
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../MindSpeed-LLM" && pwd)

# ====================== 用户配置区域 ======================

# Tokenizer 路径（根据你的模型修改）
TOKENIZER_PATH=${TOKENIZER_PATH:-/data/models/Qwen3-1.7B-Base/}

# 输出前缀（生成的文件名为 ${OUTPUT_PREFIX}_text_document.bin/.idx）
OUTPUT_PREFIX=${OUTPUT_PREFIX:-/data/dataset/mixed_dataset}

# 并行工作进程数
WORKERS=${WORKERS:-4}

# ---- 数据集配置 ----
# 在这里添加你要混合的 parquet 数据集路径
# DATASETS 数组：每个元素是一个 parquet 文件的完整路径
# PROBS 数组：每个数据集对应的采样概率（必须一一对应，且总和为 1.0）
#
# 示例：A.parquet 占 30%, B.parquet 占 50%, C.parquet 占 20%
#   DATASETS=(
#     "/path/to/A.parquet"
#     "/path/to/B.parquet"
#     "/path/to/C.parquet"
#   )
#   PROBS=(0.3 0.5 0.2)

# ↓↓↓ 请将下面的路径和比例修改为你自己的数据 ↓↓↓

# 当前目录下的示例数据
DATASETS=(
  "${SCRIPT_DIR}/alpaca.parquet"
)

# 每个数据集对应的采样概率（必须与 DATASETS 一一对应，总和为 1.0）
PROBS=(1.0)

# ↑↑↑ 请将上面的路径和比例修改为你自己的数据 ↑↑↑

# ====================== 以下无需修改 ======================

# 校验：数据集数量和概率数量必须一致
if [ ${#DATASETS[@]} -ne ${#PROBS[@]} ]; then
    echo "错误: DATASETS 数量 (${#DATASETS[@]}) 与 PROBS 数量 (${#PROBS[@]}) 不一致！"
    exit 1
fi

# 校验：所有输入文件是否存在
for ds in "${DATASETS[@]}"; do
    if [ ! -f "$ds" ]; then
        echo "错误: 文件不存在: $ds"
        exit 1
    fi
done

# 校验：概率和是否为 1.0
sum=0
for p in "${PROBS[@]}"; do
    sum=$(echo "$sum + $p" | bc 2>/dev/null || python3 -c "print($sum + $p)")
done
if [ "$(echo "$sum == 1.0" | bc 2>/dev/null || python3 -c "print(abs($sum - 1.0) < 0.001)")" != "1" ]; then
    echo "警告: 概率之和为 $sum，不等于 1.0。将自动归一化。"
fi

# 将概率列表转为逗号分隔字符串（preprocess_data.py 要求的格式）
PROBS_STR=$(IFS=,; echo "${PROBS[*]}")

echo ""
echo "========================================================"
echo "  多数据集混合转换"
echo "========================================================"
echo "  数据集列表:"
for i in "${!DATASETS[@]}"; do
    echo "    [$i] $(basename "${DATASETS[$i]}")  →  概率: ${PROBS[$i]}"
done
echo "  混合策略:  interleave_under（概率采样，欠采样）"
echo "  输出前缀:  ${OUTPUT_PREFIX}"
echo "  Tokenizer: ${TOKENIZER_PATH}"
echo "  Workers:   ${WORKERS}"
echo "========================================================"
echo ""

cd "${PROJECT_ROOT}"

python preprocess_data.py \
    --input "${DATASETS[0]}" \
    --datasets "${DATASETS[@]}" \
    --interleave-probs "${PROBS_STR}" \
    --mix-strategy interleave_under \
    --tokenizer-name-or-path "${TOKENIZER_PATH}" \
    --tokenizer-type PretrainedFromHF \
    --handler-name GeneralPretrainHandler \
    --output-prefix "${OUTPUT_PREFIX}" \
    --json-keys text \
    --workers "${WORKERS}"

echo ""
echo "========================================================"
echo "  转换完成！"
echo "========================================================"
ls -lh "${OUTPUT_PREFIX}"_text_document.{bin,idx}
echo "========================================================"
