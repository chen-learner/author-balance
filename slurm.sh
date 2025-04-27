#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --partition g078t
#SBATCH --time=00:30:00
#SBATCH --comment bupthpc
#SBATCH --output=logs/slurm.log

# export GLOO_SOCKET_IFNAME=lo
# export NCCL_SHM_DISABLE=1

# export CUDA_LAUNCH_BLOCKING=1

# export DEVICE_PLATFORM=npu
export DEVICE_PLATFORM=gpu
export NCCL_ALGO=tree

export MASTER_ADDR=$(hostname)
export MASTER_PORT=6087  #随意
CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M-%S")

srun python main.py 2>&1 | tee ./logs/${CURRENT_TIME}_new.log
