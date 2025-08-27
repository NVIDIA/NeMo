#!/bin/bash

#SBATCH -A coreai_dlalgo_llm 
#SBATCH -p interactive
#SBATCH -J coreai_dlalgo_llm-rl:verl_8b_vllm
#SBATCH -t 1:00:00 
#SBATCH -N 1 
#SBATCH --mem=0 
#SBATCH --gres=gpu:8
#SBATCH --dependency=singleton 
ACCOUNT="coreai_dlalgo_genai"
NEMO_DIR="/lustre/fsw/portfolios/coreai/users/zhiyul/NeMo"

MOUNTS="\
/lustre:/lustre,\
${NEMO_DIR}:/opt/NeMo"

# MOUNTS="\
# /lustre:/lustre"


# ORIGINAL_CONTAINER_NAME="whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6.post5-mcore0.12.1-te2.3-deepseekv3"
# NEW_CONTAINER_NAME="/lustre/fsw/portfolios/coreai/users/zhiyul/benchmark-rl/verl-ngc-cu124-vllm0.8.5-sglang0.4.6.post5-mcore0.12.1-te2.3-deepseekv3.sqsh"
# ORIGINAL_CONTAINER_NAME="/lustre/fsw/portfolios/coreai/users/zhiyul/benchmark-rl/enroot-images/gitlab-master.nvidia.com/zhiyul/nemo-ci/reinforcer:04222025.squashfs"
# NEW_CONTAINER_NAME="/lustre/fsw/portfolios/coreai/users/zhiyul/benchmark-rl/enroot-images/gitlab-master.nvidia.com/zhiyul/nemo-ci/reinforcer:06032025-nsys.squashfs"
# ORIGINAL_CONTAINER_NAME="/lustre/fsw/portfolios/coreai/users/zhiyul/benchmark-rl/verl-ngc-cu124-vllm0.8.5-sglang0.4.6.post5-mcore0.12.1-te2.3-deepseekv3.sqsh"
# NEW_CONTAINER_NAME="/lustre/fsw/portfolios/coreai/users/zhiyul/benchmark-rl/verl-ngc-cu124-vllm0.8.5-sglang0.4.6.post5-mcore0.12.1-te2.3-deepseekv3-nsys.sqsh"
# ORIGINAL_CONTAINER_NAME="/lustre/fsw/portfolios/coreai/users/zhiyul/benchmark-rl/verl-ngc-cu124-vllm0.8.5-sglang0.4.6.post5-mcore0.12.1-te2.3-deepseekv3-nsys.sqsh"
# NEW_CONTAINER_NAME="/lustre/fsw/portfolios/coreai/users/zhiyul/benchmark-rl/verl-ngc-cu124-vllm0.9.0-sglang0.4.6.post5-mcore0.12.1-te2.3-deepseekv3-nsys.sqsh"

# ORIGINAL_CONTAINER_NAME="gitlab-master.nvidia.com/zhiyul/nemo-ci/verl:app-verl0.5-vllm0.9.1-mcore0.12.2-te2.2"
# ORIGINAL_CONTAINER_NAME="verlai/verl:app-verl0.5-vllm0.9.1-mcore0.12.2-te2.2"
# NEW_CONTAINER_NAME="/lustre/fsw/portfolios/coreai/users/zhiyul/benchmark-rl/verl-ngc-cu124-vllm0.9.1-mcore0.12.2-te2.2.sqsh"

# CONTAINER_NAME="/lustre/fsw/portfolios/coreai/users/zhiyul/NeMo/nemo-dev.sqsh"
# CONTAINER_NAME="nvcr.io#nvidia/nemo:25.07"
CONTAINER_NAME="/lustre/fsw/portfolios/coreai/users/zhiyul/NeMo/nemo-25.07.sqsh"

srun -N1 \
 -n1 \
 -A ${ACCOUNT} \
 -J ${ACCOUNT}-rl:zhiyul-verl-container-setup \
 -t 2:00:00 \
 -p interactive \
 --gres=gpu:8 \
 --export=ALL,HOME=/tmp \
 --no-container-mount-home \
 --container-mounts ${MOUNTS} \
 --container-image=${CONTAINER_NAME} \
 --pty bash