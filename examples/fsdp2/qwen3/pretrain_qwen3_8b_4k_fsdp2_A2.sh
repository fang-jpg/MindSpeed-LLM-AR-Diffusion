source examples/fsdp2/env_config.sh

NPUS_PER_NODE=8
MASTER_ADDR=80.5.5.114
MASTER_PORT=6499
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
export HCCL_NPU_SOCKET_IFNAME=enp189s0f0
export HCCL_NPU_SOCKET_PORT_RANGE=50000-60000

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS train_fsdp2.py examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A2.yaml \
    2>&1 | tee /home/c00633840/logs/qwen3_1.7b_train_727_ar.log
