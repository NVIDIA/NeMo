params=()
if [[ $MAX_STEPS -le 100 ]]; then # If greater than hundred we use defaults set in the training config file.
  LOG_EVERY_N_STEPS=`expr $MAX_STEPS / 100`
  VAL_CHECK_INTERVAL=`expr $MAX_STEPS / 5`
  params+=(finetuning.trainer.log_every_n_steps=$LOG_EVERY_N_STEPS)
  params+=(finetuning.trainer.val_check_interval=$VAL_CHECK_INTERVAL)
fi

set -o xtrace

MICRO_BATH_SIZE=$((16 * TP_SIZE * PP_SIZE / NUM_NODES))

HYDRA_FULL_ERROR=1 BIGNLP_CI=1 python3 main.py \
    finetuning=${RUN_MODEL}/mnli \
    stages=["fine_tuning"] \
    bignlp_path=${GIT_CLONE_PATH} \
    data_dir=${BASE_RESULTS_DIR}/data \
    base_results_dir=${BASE_RESULTS_DIR} \
    "container='${BUILD_IMAGE_NAME_SRUN}'" \
    cluster.partition=${SLURM_PARTITION} \
    cluster.account=${SLURM_ACCOUNT} \
    cluster.gpus_per_task=null \
    cluster.gpus_per_node=null \
    cluster.job_name_prefix="${SLURM_ACCOUNT}-bignlp_ci:" \
    finetuning.run.time_limit=${TIME_LIMIT} \
    finetuning.run.results_dir=${BASE_RESULTS_DIR}/${RUN_NAME} \
    finetuning.trainer.num_nodes=${NUM_NODES} \
    finetuning.trainer.max_steps=${MAX_STEPS} \
    finetuning.model.restore_from_path=${BASE_RESULTS_DIR}/convert_${RUN_MODEL}_${RUN_MODEL_SIZE}_tp${TP_SIZE}_pp${PP_SIZE}/results/megatron_t5.nemo \
    finetuning.model.tensor_model_parallel_size=${TP_SIZE} \
    finetuning.model.pipeline_model_parallel_size=${PP_SIZE} \
    finetuning.model.data.train_ds.micro_batch_size=${MICRO_BATH_SIZE} \
    finetuning.model.data.validation_ds.micro_batch_size=${MICRO_BATH_SIZE} \
    "${params[@]}"
