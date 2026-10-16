#!/bin/bash
set -euo pipefail

DATA_DIR=${DATA_DIR:-/share/dataset/x00840191/data/opdlm_train_data}
mkdir -p "${DATA_DIR}"

if command -v hf >/dev/null 2>&1; then
  hf download divelab/opdlm_train_data \
    --repo-type dataset \
    --local-dir "${DATA_DIR}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download divelab/opdlm_train_data \
    --repo-type dataset \
    --local-dir "${DATA_DIR}"
else
  echo "请先安装 huggingface_hub；当前环境中没有可用的 'hf' 或 'huggingface-cli' 命令。" >&2
  exit 2
fi

DATA_FILE="${DATA_DIR}/opdlm_train.json"
python -c 'import json,sys; p=sys.argv[1]; rows=json.load(open(p, encoding="utf-8")); assert len(rows)==61816, f"应有 61816 条记录，实际为 {len(rows)} 条"; required={"question","ground_truth_answer","domain","source","tests_json"}; assert all(required <= set(row) for row in rows), "数据中缺少必要字段"; print(f"已验证 {len(rows)} 条 OPDLM 记录：{p}")' "${DATA_FILE}"
