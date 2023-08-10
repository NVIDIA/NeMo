#!/bin/bash
#SBATCH -A account_name
#SBATCH -p partition_name
#SBATCH -N 16
#SBATCH -t 4:00:00
#SBATCH -J account_name-bloom:bloom-7b-te
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --dependency=singleton

LUSTRE=  # Path to home dir
DATA_DIR=  # "[.5,/raid/data/pile/my-gpt3_00_text_document,.5,/raid/data/pile/my-gpt3_01_text_document]"
TOKENIZER_DIR=  # /path/to/bloom-7b1

NEMO_PATH="${LUSTRE}/NeMo"
CONFIG_PATH="${NEMO_PATH}/examples/nlp/language_modeling/conf"
CONFIG_NAME="megatron_bloom_7b1_config"

# W&B
PROJECT="bloom-7b1-te"
WANDB=  # WANDB key

NAME="bloom-7b1-te-fa1-${SLURM_JOB_NUM_NODES}nodes"
RESULTS="${LUSTRE}/results/${NAME}"

mkdir -p ${RESULTS}

# Necessary Exports
export HYDRA_FULL_ERROR=1
OUTFILE="${RESULTS}/slurm-%j.out"
ERRFILE="${RESULTS}/error-%j.out"

MOUNTS="--container-mounts=${LUSTRE}:${LUSTRE},$HOME/.cache:/root/.cache"

read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& wandb login ${WANDB} \
&& echo "Starting training" \
&& cd ${NEMO_PATH} \
&& export PYTHONPATH="${NEMO_PATH}:\$PYTHONPATH" \
&& CUDA_VISIBLE_DEVICES=0,4,2,6,1,5,3,7 python ${NEMO_PATH}/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
    exp_manager.resume_if_exists=True \
    model.micro_batch_size=4 \
    model.global_batch_size=2048 \
    model.tokenizer.type=${TOKENIZER_DIR} \
    model.data.data_prefix=${DATA_DIR} \
    model.transformer_engine=True \
    model.use_flash_attention=False \
    exp_manager.explicit_log_dir=${RESULTS} \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=${NAME} \
    exp_manager.wandb_logger_kwargs.project=${PROJECT}
EOF

DOCKER_IMAGE=  # docker image
srun -o $OUTFILE -e $ERRFILE --container-image=${DOCKER_IMAGE} $MOUNTS bash -c "${cmd}"
set +x
