
params=()
if [[ $MAX_STEPS -le 100 ]]; then # If greater than hundred we use defaults set in the training config file.
  LOG_EVERY_N_STEPS=`expr $MAX_STEPS / 100`
  VAL_CHECK_INTERVAL=`expr $MAX_STEPS / 5`
  LIMIT_VAL_BATCHES=`expr $MAX_STEPS / 20`
  params+=(training.trainer.log_every_n_steps=$LOG_EVERY_N_STEPS)
  params+=(training.trainer.limit_val_batches=$LIMIT_VAL_BATCHES)
  params+=(training.trainer.val_check_interval=$VAL_CHECK_INTERVAL)
fi

DATA_DIR=/lustre/fsw/joc/big_nlp/gpt3/prepare_dataset/the_pile/train
DATA_PREFIX=[1.0,/lustre/fsw/joc/big_nlp/gpt3/prepare_dataset/the_pile/train/my-gpt3_00_text_document]

set -o xtrace

#TODO : Can add additional parameters (key value pairs from gitlab-ci.yaml file)
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
    training.model.tensor_model_parallel_size=${TP_SIZE} \
    training.model.pipeline_model_parallel_size=${PP_SIZE} \
    training.model.data.data_prefix=${DATA_PREFIX} \
    "${params[@]}"
