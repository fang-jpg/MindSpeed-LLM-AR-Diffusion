export CUDA_DEVICE_MAX_CONNECTIONS=1

# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 开启 --qlora-nf4 选项使用 QLoRA
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 1 \
    --load-dir ./model_from_hf/llama-2-70b-hf/ \
    --save-dir ./model_weights/Llama2-mcore/ \
    --tokenizer-model ./model_from_hf/llama-2-70b-hf/tokenizer.model \
    --use-mcore-models \
    --qlora-nf4 \
    --model-type-hf llama2
