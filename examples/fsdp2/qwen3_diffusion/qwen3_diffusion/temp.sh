source examples/fsdp2/env_config.sh

NPUS_PER_NODE=8
MASTER_ADDR=80.5.5.114
MASTER_PORT=6600
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
export HCCL_NPU_SOCKET_IFNAME=enp189s0f0
export HCCL_NPU_SOCKET_PORT_RANGE=50000-60000
MODEL_VARIANT=${MODEL_VARIANT:-base}
BASE_SYSTEM_PROMPT=${BASE_SYSTEM_PROMPT:-You are a helpful assistant.}

if [[ "${MODEL_VARIANT}" == "base" || "${MODEL_VARIANT}" == "base_chat" ]]; then
    USE_CHAT_TEMPLATE=false
    ENABLE_THINKING=false
    MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
    REPETITION_PENALTY=${REPETITION_PENALTY:-1.1}
    NO_REPEAT_NGRAM_SIZE=${NO_REPEAT_NGRAM_SIZE:-4}
else
    USE_CHAT_TEMPLATE=true
    ENABLE_THINKING=true
    MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
    REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
    NO_REPEAT_NGRAM_SIZE=${NO_REPEAT_NGRAM_SIZE:-0}
fi

if [[ "${MODEL_VARIANT}" == "base_chat" ]]; then
    BASE_CHAT=true
else
    BASE_CHAT=false
fi

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun $DISTRIBUTED_ARGS \
  inference_fsdp2.py \
  --model.model_name_or_path /share/dataset/x00840191/model_train/ckpt/phase4_cpt_ar_duffision_0804/models/global_step_4500/ \
  --model.model_id "qwen3_diffusion" \
  --parallel.fsdp_size 8 \
  --parallel.ep_size 1 \
  --parallel.ep_fsdp_size 1 \
  --inference.base_chat "${BASE_CHAT}" \
  --inference.base_system_prompt "${BASE_SYSTEM_PROMPT}" \
  --parallel.cp_size 1 \
  --inference.use_chat_template "${USE_CHAT_TEMPLATE}" \
  --inference.enable_thinking "${ENABLE_THINKING}" \
  --inference.max_new_tokens 256 \
  --inference.do_sample true \
  --inference.temperature 0.7 \
  --inference.top_k 50