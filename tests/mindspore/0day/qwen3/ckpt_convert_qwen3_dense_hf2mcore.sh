# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置需要的权重转换参数
python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 4 \
    --target-pipeline-parallel-size 2 \
    --target-expert-parallel-size 1 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --load-dir ./model_from_hf/qwen3_8b_hf/ \
    --save-dir ./model_weights/qwen3_8b_mcore/ \
    --tokenizer-model ./model_from_hf/qwen3_38b_hf/tokenizer.json \
    --model-type-hf qwen3 \
    --params-dtype bf16