# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换
python convert_ckpt.py \
   --model-type-hf glm4 \
   --post-norm \
   --spec mindspeed_llm.tasks.models.spec.gemma2_spec layer_spec \
   --model-type GPT \
   --load-model-type mg \
   --save-model-type hf \
   --target-tensor-parallel-size 1 \
   --target-pipeline-parallel-size 1 \
   --load-dir ./model_weights/glm-z1-mcore \
   --save-dir ./model_from_hf/GLM-Z1-32B-0414/ \
   --use-mcore-models \
   --params-dtype bf16
