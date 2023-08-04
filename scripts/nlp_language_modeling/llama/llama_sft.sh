#!/bin/bash
#SBATCH -N 16 --ntasks-per-node 8 -A devtech -p luna --job-name devtech-gpt:gpt:sft_8b -t 80

<<"COMMENT"
#SBATCH -A devtech
#SBATCH -p luna
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH -J "devtech-DLFW:sft"
#SBATCH --ntasks-per-node=8
COMMENT

set -x

CONTAINER="nvcr.io/ea-bignlp/bignlp-training:23.04-py3" # use own pre-built nemo?
WANDB="???" 

# Model config: conf/megatron_gpt_config.yaml
CONFIG_PATH='conf'
CONFIG_NAME='megatron_llama_sft'

GLOBAL_BATCH_SIZE=128
VALID_GLOBAL_BATCH_SIZE=128
MICRO_BATCH_SIZE=1
ACCUMULATE_GRAD_BATCHES=1
TENSOR_MODEL_PARALLEL_SIZE=4
PIPELINE_MODEL_PARALLEL_SIZE=1
VAL_CHECK_INTERVAL=100
MAX_STEPS=1000
DATA_SPLITS_STRING="\'99982,9,9\'"
LR="1e-6"

# Model architecture
MAX_SEQ_LENGTH=4096

# Logging
PROJECT="nemo_llama_sft"
EXPNAME="llama_8b_rare_finch_1e-6_new"

# Mounts
GPFS="/path/to/NeMo"
PREPROC_DATA="/path/to/data"
#MEGATRON_PATH="/lustre/fsw/devtech/hpc-devtech/lit/software/Megatron-LM/"
RESULTS="${GPFS}/../../../results/${EXPNAME}"

CODE="${GPFS}"
#MODEL="/lustre/fsw/swdl/swdl-langspeech/sandeepsub/models"
MODEL_DIR="/path/to/pretrained_model"
#MODEL_NAME="llama-7B-tp4.nemo"
MODEL_NAME="llama-7B.nemo"

mkdir -p ${RESULTS}

MOUNTS="--container-mounts=$CODE:/code,$RESULTS:/results,$PREPROC_DATA:/preproc_data,$MODEL_DIR:/models"

TRAIN="[/preproc_data/tool_generated_sft_datasets/rare-finch/rare-finch_commercial.jsonl]"
#TRAIN="[/preproc_data/tool_generated_sft_datasets/giga-bison/giga-bison_commercial.shuf.jsonl]"

VALID="[/preproc_data/scale_ai_data/delivery_2023-04-07-val.jsonl]"

VALID_NAMES="[scale-ai]"

CONCAT_SAMPLING_PROBS="[1.0]"

# Necessary Exports
export HYDRA_FULL_ERROR=1

OUTFILE="${RESULTS}/slurm-%j-%n.out"
ERRFILE="${RESULTS}/error-%j-%n.out"

#&& git rev-parse HEAD \
read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& export WANDB_API_KEY=${WANDB} \
&& echo "Starting training" \
&& export PYTHONPATH="/code/.:${PYTHONPATH}" \
&& python /code/examples/nlp/language_modeling/tuning/megatron_gpt_sft.py \
	--config-path=/code/examples/nlp/language_modeling/tuning/conf \
	--config-name=${CONFIG_NAME} \
	+trainer.limit_val_batches=5 \
	trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
	trainer.devices=${SLURM_NTASKS_PER_NODE} \
	trainer.max_epochs=null \
	trainer.max_steps=${MAX_STEPS} \
	trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
	trainer.precision=bf16 \
	model.megatron_amp_O2=True \
	model.restore_from_path=/models/${MODEL_NAME} \
	model.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
	model.pipeline_model_parallel_size=${PIPELINE_MODEL_PARALLEL_SIZE} \
	model.optim.name=distributed_fused_adam \
	model.optim.lr=${LR} \
	++model.optim.bucket_cap_mb=200 \
    	++model.optim.overlap_grad_sync=False \
    	++model.optim.contiguous_grad_buffer=True \
	model.answer_only_loss=True \
	model.activations_checkpoint_granularity=selective \
	model.activations_checkpoint_method=block \
	model.activations_checkpoint_num_layers=8 \
	model.data.train_ds.max_seq_length=${MAX_SEQ_LENGTH} \
	model.data.train_ds.micro_batch_size=${MICRO_BATCH_SIZE} \
	model.data.train_ds.global_batch_size=${GLOBAL_BATCH_SIZE} \
	model.data.train_ds.file_names=${TRAIN} \
	model.data.train_ds.concat_sampling_probabilities=${CONCAT_SAMPLING_PROBS} \
	model.data.train_ds.num_workers=0 \
	model.data.validation_ds.max_seq_length=${MAX_SEQ_LENGTH} \
	model.data.validation_ds.file_names=${VALID} \
	model.data.validation_ds.names=${VALID_NAMES} \
	model.data.validation_ds.micro_batch_size=${MICRO_BATCH_SIZE} \
	model.data.validation_ds.global_batch_size=${VALID_GLOBAL_BATCH_SIZE} \
	model.data.validation_ds.write_predictions_to_file=False \
	model.data.validation_ds.output_file_path_prefix=/results/predictions \
	model.data.validation_ds.num_workers=0 \
	model.data.validation_ds.metric.name=loss \
	model.data.test_ds.max_seq_length=${MAX_SEQ_LENGTH} \
      	model.data.test_ds.file_names=${VALID} \
	model.data.test_ds.names=${VALID_NAMES} \
	model.data.test_ds.micro_batch_size=${MICRO_BATCH_SIZE} \
	model.data.test_ds.global_batch_size=${VALID_GLOBAL_BATCH_SIZE} \
	model.data.test_ds.write_predictions_to_file=False \
	model.data.test_ds.output_file_path_prefix=/results/predictions \
	model.data.test_ds.num_workers=0 \
	model.data.test_ds.metric.name=loss \
	exp_manager.create_wandb_logger=True \
	exp_manager.wandb_logger_kwargs.name=${EXPNAME} \
	exp_manager.wandb_logger_kwargs.project=${PROJECT} \
	exp_manager.explicit_log_dir=/results \
	exp_manager.resume_if_exists=False \
	exp_manager.resume_ignore_no_checkpoint=True \
	exp_manager.create_checkpoint_callback=True \
	exp_manager.checkpoint_callback_params.monitor=validation_loss \
	++exp_manager.checkpoint_callback_params.save_top_k=3 \
	exp_manager.checkpoint_callback_params.mode=min \
	++exp_manager.max_time_per_run=00:01:05:00 \
	++exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True
EOF

srun --no-container-mount-home --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
