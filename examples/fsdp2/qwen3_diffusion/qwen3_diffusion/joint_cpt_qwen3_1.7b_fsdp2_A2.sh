#!/bin/bash

source /share/f50058111/activate_cann_910b_9.1.sh
source examples/fsdp2/env_config.sh

NPUS_PER_NODE=8
MASTER_ADDR=80.5.5.114
MASTER_PORT=6499
NNODES=2
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

torchrun $DISTRIBUTED_ARGS train_fsdp2.py \
    examples/fsdp2/qwen3_diffusion/joint_cpt_qwen3_1.7b_fsdp2_A2.yaml 

#!/bin/bash

# source /share/f50058111/activate_cann_910b_9.1.sh
# source examples/fsdp2/env_config.sh

# NPUS_PER_NODE=8

# MASTER_ADDR=127.0.0.1
# MASTER_PORT=6499

# NNODES=1
# NODE_RANK=0

# WORLD_SIZE=$NPUS_PER_NODE

# DISTRIBUTED_ARGS="
#     --nproc_per_node $NPUS_PER_NODE
#     --nnodes $NNODES
#     --node_rank $NODE_RANK
#     --master_addr $MASTER_ADDR
#     --master_port $MASTER_PORT
# "

# torchrun $DISTRIBUTED_ARGS train_fsdp2.py \
#     examples/fsdp2/qwen3_diffusion/joint_cpt_qwen3_1.7b_fsdp2_A2.yaml 