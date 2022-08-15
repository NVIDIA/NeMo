#!/bin/bash
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=440gb 
#SBATCH --time=00:20:00

HPCX_PATH="/opt/hpcx-v2.9.0-gcc-MLNX_OFED_LINUX-5.4-1.0.3.0-ubuntu18.04-x86_64"

export UCX_IB_PCI_RELAXED_ORDERING=on \
       NCCL_IB_PCI_RELAXED_ORDERING=1 \
       UCX_TLS=rc \
       NCCL_SHARP_DISABLE=1 \
       NCCL_COLLNET_ENABLE=0 \
       NCCL_TOPO_FILE=/microsoft/ndv4-topo.xml \
       OMPI_MCA_pml=ucx \
       OMPI_MCA_btl=^openib

env | grep "SLURMD_NODENAME="
env | grep "SLURM_NODELIST="

srun --container-image="nvcr.io/nvidia/pytorch:21.09-py3" \
     --container-name=nccl \
     --container-mounts="/opt/microsoft:/microsoft,$PWD:/nccl" \
     --ntasks-per-node=1 \
     bash -c "
     apt update &&
     apt-get install -y infiniband-diags
     "

srun --gpus-per-node=8 \
     --ntasks-per-node=8 \
     --container-name=nccl \
     --mem=440gb \
     --container-mounts="/opt/microsoft:/microsoft,$PWD:/nccl,$HPCX_PATH:/opt/hpcx" \
     bash -c "
     source /opt/hpcx/hpcx-init.sh &&
     hpcx_load &&
     /nccl/nccl-tests/build/all_reduce_perf -b8 -f 2 -g 1 -e 8G
     "
