if [[ -z $LOG_EVERY_N_STEPS ]]; then
  LOG_EVERY_N_STEPS=`expr $MAX_STEPS / 100`
fi

if [[ -z $VAL_CHECK_INTERVAL ]]; then
  VAL_CHECK_INTERVAL=`expr $MAX_STEPS / 5`
fi

if [[ -z $LIMIT_VAL_BATCHES ]]; then
  LIMIT_VAL_BATCHES=`expr $MAX_STEPS / 20`
fi

DATA_DIR=/lustre/fsw/joc/big_nlp/gpt3/prepare_dataset/the_pile/train
DATA_PREFIX=[1.0,/lustre/fsw/joc/big_nlp/gpt3/prepare_dataset/the_pile/train/my-gpt3_00_text_document]

set -o xtrace

HYDRA_FULL_ERROR=1 python3 main.py \
    +ci_test=True \
    training=${RUN_MODEL}/${RUN_MODEL_SIZE} \
    run_data_preparation=False \
    run_training=True \
    run_conversion=False \
    run_finetuning=False \
    run_evaluation=False \
    bignlp_path=${GIT_CLONE_PATH} \
    data_dir=${DATA_DIR} \
    base_results_dir=${BASE_RESULTS_DIR} \
    "container='${BUILD_IMAGE_NAME_SRUN}'" \
    cluster.partition=${SLURM_PARTITION} \
    cluster.account=${SLURM_ACCOUNT} \
    cluster.gpus_per_task=null \
    cluster.gpus_per_node=null \
    cluster.job_name_prefix="${SLURM_ACCOUNT}-bignlp_ci:" \
    training.run.name=${RUN_NAME} \
    training.run.time_limit=${RUN_TIME_LIMIT} \
    training.trainer.num_nodes=${NUM_NODES} \
    training.trainer.max_steps=${MAX_STEPS} \
    training.trainer.log_every_n_steps=${LOG_EVERY_N_STEPS} \
    training.trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
    training.trainer.limit_val_batches=${LIMIT_VAL_BATCHES} \
    training.model.tensor_model_parallel_size=${TP_SIZE} \
    training.model.pipeline_model_parallel_size=${PP_SIZE} \
    training.model.data.data_prefix=${DATA_PREFIX}
