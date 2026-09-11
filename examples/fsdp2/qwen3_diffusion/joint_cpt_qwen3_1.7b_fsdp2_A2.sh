#!/bin/bash

source /share/f50058111/activate_cann_910b_9.1.sh
source examples/fsdp2/env_config.sh

NPUS_PER_NODE=8
MASTER_ADDR=80.5.5.114
MASTER_PORT=6499
NNODES=2
NODE_RANK=1
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
    examples/fsdp2/qwen3_diffusion/joint_cpt_qwen3_1.7b_fsdp2_A2.yaml \
    2>&1 | tee /share/f50058111/logs/qwen3_1.7b_train_ar_cpt_0910_b8.log

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
#     examples/fsdp2/qwen3_diffusion/joint_cpt_qwen3_1.7b_fsdp2_A2.yaml \
#     2>&1 | tee /share/f50058111/logs/qwen3_1.7b_train_ar_cpt_0909_b8.log
#!/bin/bash

# # 1. 激活 CANN 与环境
# source /share/f50058111/activate_cann_910b_9.1.sh
# source examples/fsdp2/env_config.sh

# # 2. 从平台预置环境变量获取分布式参数（若平台未传入则给默认兜底）
# NPUS_PER_NODE=${NPUS_PER_NODE:-8}
# NNODES=${NNODES:-2}
# NODE_RANK=${NODE_RANK:-0}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# MASTER_PORT=${MASTER_PORT:-6001}

# echo "================ Distributed Info ================"
# echo "Node Rank    : ${NODE_RANK} / ${NNODES}"
# echo "Master Addr  : ${MASTER_ADDR}:${MASTER_PORT}"
# echo "NPUs Per Node: ${NPUS_PER_NODE}"
# echo "Code Path    : ${CODE_LOAD_PATH}"
# echo "=================================================="

# # 3. 动态获取容器内实际用于通信的网卡（防止网卡名不叫 enp189s0f0）
# # 优先使用平台指定的网卡，或者根据 MASTER_ADDR 所在网段自动推导，或默认使用 eth0
# IFACE=$(ip route | grep default | awk '{print $5}' | head -n 1)
# IFACE=${IFACE:-eth0}

# export HCCL_NPU_SOCKET_IFNAME=${IFACE}
# export GLOO_SOCKET_IFNAME=${IFACE}
# export TP_SOCKET_IFNAME=${IFACE}
# export HCCL_CONNECT_TIMEOUT=1200

# # 4. 组装 torchrun 参数
# DISTRIBUTED_ARGS="
#     --nproc_per_node ${NPUS_PER_NODE} \
#     --nnodes ${NNODES} \
#     --node_rank ${NODE_RANK} \
#     --master_addr ${MASTER_ADDR} \
#     --master_port ${MASTER_PORT}
# "

# # 5. 执行训练
# torchrun ${DISTRIBUTED_ARGS} train_fsdp2.py \
#     examples/fsdp2/qwen3_diffusion/joint_cpt_qwen3_1.7b_fsdp2_A2.yaml