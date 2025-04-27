#! /bin/bash
# source /root/efs/zhaoxuncheng/cann_as/ascend-toolkit/set_env.sh
source /root/efs/jiangzhuo/Ascend/ascend-toolkit/set_env.sh

# export ASCEND_LAUNCH_BLOCKING=1
export GLOO_SOCKET_IFNAME=lo
# export HCCL_BUFFERSIZE=1024
# export ASCEND_LAUNCH_BLOCKING=1
export DEVICE_PLATFORM=npu
# export DEVICE_PLATFORM=gpu
export TRANSFER_METHOD=serial
export max_split_size_mb=32
MASTER_ADDR="localhost"
MASTER_PORT=6011  #随意
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M-%S")
DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"


torchrun $DISTRIBUTED_ARGS main.py 2>&1 | tee ./logs/${CURRENT_TIME}_new.log


