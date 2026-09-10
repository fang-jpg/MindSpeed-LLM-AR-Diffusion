export CUDA_DEVICE_MAX_CONNECTIONS=1
# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换
python convert_ckpt.py \
    --use-mcore-models \
    --model-type-hf phi3.5-moe \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 4 \
    --target-pipeline-parallel-size 4 \
    --target-expert-parallel-size 1 \
    --load-dir ./model_from_hf/Phi3.5-MoE-instruct-hf \
    --save-dir ./model_weights/phi3.5-moe-mcore \
    --tokenizer-model ./model_from_hf/Phi3.5-MoE-hf/tokenizer.model \
    --spec mindspeed_llm.tasks.models.spec.phi35_moe_spec layer_spec \
    --add-qkv-bias \
    --add-dense-bias \
    --moe-grouped-gemm \
    --params-dtype bf16
#    --num-layers-per-virtual-pipeline-stage 2 \     当转换pretain使用的权重时，增加该参数