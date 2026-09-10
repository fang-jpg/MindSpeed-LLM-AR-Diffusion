# 修改 ascend-toolkit 路径
export CUDA_DEVICE_MAX_CONNECTIONS=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 8 \
    --load-dir /share/dataset/x00840191/models/qwen31.7/ \
    --save-dir /home/c00633840/ckpt/qwen3_1point7b_mcore/ \
    --model-type-hf qwen3