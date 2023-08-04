#!/bin/bash

# Parameters
#SBATCH --account=coreai_devtech_all
#SBATCH --dependency=singleton
#SBATCH --error=llama_30b_%j.err
#SBATCH --exclusive
#SBATCH --job-name=coreai_devtech_all-gpt:gpt3_7b_4nodes_tp_1_pp_1_mbs_2_act_ckpt_0_num_mbs_act_None_act_per_pipe_None
#SBATCH --mem=0
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=llama_30b_%j.out
#SBATCH --partition=luna
#SBATCH --time=0-0:40:00

# setup
export TRANSFORMERS_OFFLINE=1
export NCCL_AVOID_RECORD_STREAMS=1

# command 1
srun --container-image nvcr.io/ea-bignlp/nemofw-training:23.05-py3 --container-mounts ???:??? --no-container-mount-home bash -c "
  wandb login ???;
  cd ???;
  git rev-parse HEAD;
  export PYTHONPATH=/path/to/apex:/path/to/NeMo:\${PYTHONPATH};
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -u /path/to/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py  \
  --config-path=/path/to/config \
  --config-name=??? "
