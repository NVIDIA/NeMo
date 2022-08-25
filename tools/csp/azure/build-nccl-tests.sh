#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

HPCX_PATH="/opt/hpcx-v2.9.0-gcc-MLNX_OFED_LINUX-5.4-1.0.3.0-ubuntu18.04-x86_64"

export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib

srun --container-mounts="$PWD:/nccl,$HPCX_PATH:/opt/hpcx" \
     --container-image="nvcr.io/nvidia/pytorch:21.09-py3" \
     --container-name="nccl" \
     bash -c "
     cd /nccl &&
     git clone https://github.com/NVIDIA/nccl-tests.git &&
     source /opt/hpcx/hpcx-init.sh &&
     hpcx_load &&
     cd nccl-tests &&
     make MPI=1"
