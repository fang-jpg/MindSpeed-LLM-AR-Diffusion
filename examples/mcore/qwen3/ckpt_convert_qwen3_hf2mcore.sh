
# 修改 ascend-toolkit 路径
export CUDA_DEVICE_MAX_CONNECTIONS=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh

HF_MODEL_PATH=${HF_MODEL_PATH:-/share/dataset/x00840191/model_train/ckpt/phase4_cpt_ar_duffision_0807/models/global_step_24000/}
MCORE_OUTPUT_PATH=${MCORE_OUTPUT_PATH:-/share/dataset/x00840191/models/ar_diff_0807_24000/}
TARGET_TP=${TARGET_TP:-1}
TARGET_PP=${TARGET_PP:-1}

python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size "${TARGET_TP}" \
    --target-pipeline-parallel-size "${TARGET_PP}" \
    --load-dir "${HF_MODEL_PATH}" \
    --save-dir "${MCORE_OUTPUT_PATH}" \
    --model-type-hf qwen3