#!/bin/bash
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:20:00

export NCCL_TOPO_FILE=/nccl/p4d-24xl-topo.xml

env | grep "SLURMD_NODENAME="
env | grep "SLURM_NODELIST="

srun --container-image=../../../bignlp_training.sqsh \
     --container-mounts="$PWD:/nccl" \
     bash -c "
     /nccl/nccl-tests/build/all_reduce_perf -b 256M -e 8G -f 2 -c 1 -n 10"
