export CUDA_DEVICE_MAX_CONNECTIONS=1
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --use-mcore-models \
    --model-type-hf phi3.5-moe \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --target-expert-parallel-size 1 \
    --load-dir ./model_weights/phi3.5-moe-mcore \
    --save-dir ./model_from_hf/Phi3.5-MoE-instruct-hf \
    --spec mindspeed_llm.tasks.models.spec.phi35_moe_spec layer_spec \
    --add-qkv-bias \
    --add-dense-bias \
    --moe-grouped-gemm \
    --add-output-layer-bias \
    --params-dtype bf16
