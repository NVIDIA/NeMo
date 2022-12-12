#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

srun --container-mounts="$PWD:/nccl" \
     --container-image=../../../bignlp_training.sqsh \
     bash -c "
     cd /nccl &&
     curl -fSsL --proto '=https' https://github.com/NVIDIA/nccl-tests/tarball/master | tar xz &&
     mv NVIDIA-nccl-tests* nccl-tests &&
     cd nccl-tests &&
     make -j CUDA_HOME=/usr/local/cuda MPI=1 MPI_HOME=/opt/hpcx/ompi"
