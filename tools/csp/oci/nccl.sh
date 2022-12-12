#!/bin/bash
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:20:00

HPCX_PATH="/opt/hpcx-v2.11-gcc-MLNX_OFED_LINUX-5-ubuntu20.04-cuda11-gdrcopy2-nccl2.11-x86_64"

export UCX_TLS=ud,sm,self \
       UCX_NET_DEVICES=mlx5_0:1 \
       NCCL_IB_HCA="=mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_14,mlx5_15,mlx5_16,mlx5_17,mlx5_9,mlx5_10,mlx5_11,mlx5_12" \
       NCCL_IB_TIMEOUT=22 \
       NCCL_IB_RETRY_CNT=14 \
       NCCL_IB_SL=0 \
       NCCL_IB_TC=41 \
       RX_QUEUE_LEN=8192 \
       IB_RX_QUEUE_LEN=8192 \
       NCCL_DEBUG=WARN \
       OMPI_MCA_pml=ucx \
       OMPI_MCA_btl=^openib \
       OMPI_MCA_btl_tcp_if_include="192.168.0.0/16"

env | grep "SLURMD_NODENAME="
env | grep "SLURM_NODELIST="

srun --container-image="nvcr.io/nvidia/pytorch:21.09-py3" \
     --container-name=nccl \
     --container-mounts="$PWD:/nccl" \
     --ntasks-per-node=1 \
     bash -c "
     apt update &&
     apt-get install -y infiniband-diags
     "

srun --gpus-per-node=8 \
     --ntasks-per-node=8 \
     --cpu-bind=rank_ldom \
     --container-name=nccl \
     --container-mounts="$PWD:/nccl,$HPCX_PATH:/opt/hpcx" \
     bash -c "
     source /opt/hpcx/hpcx-init.sh &&
     hpcx_load && 
     /nccl/nccl-tests/build/all_reduce_perf -b1G -e10G -i $((1024*1024*1024*9))
     "
