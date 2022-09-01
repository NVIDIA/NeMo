params=()
if [[ $MAX_STEPS -le 100 ]]; then # If greater than hundred we use defaults set in the training config file.
  LOG_EVERY_N_STEPS=`expr $MAX_STEPS / 100`
  VAL_CHECK_INTERVAL=`expr $MAX_STEPS / 5`
  params+=(fine_tuning.trainer.log_every_n_steps=$LOG_EVERY_N_STEPS)
  params+=(fine_tuning.trainer.val_check_interval=$VAL_CHECK_INTERVAL)
  UPSTREAM_RUN_NAME=convert_${RUN_MODEL}_${RUN_MODEL_SIZE}_tp${TP_SIZE}_pp${PP_SIZE}
  if [[ ! -z "$RUN_NAME_SUFFIX" ]]; then export UPSTREAM_RUN_NAME=${UPSTREAM_RUN_NAME}_${RUN_NAME_SUFFIX}; fi
  LANGUAGE_MODEL_PATH=${BASE_RESULTS_DIR}/${UPSTREAM_RUN_NAME}/results/megatron_mt5.nemo
fi
if [[ ! -z $LOCAL_NEMO_PATH ]]; then
  params+=("container_mounts=[${LOCAL_NEMO_PATH}:/opt/bignlp/NeMo]")
fi
PP_SPLIT_RANK=${PP_SPLIT_RANK:-`expr ${PP_SIZE} / 2`}
MICRO_BATH_SIZE=$((16 * TP_SIZE * PP_SIZE / NUM_NODES))

if [[ ${TEST_TASK} != "xquad" ]]; then
  params+=(++fine_tuning.model.data.test_ds.micro_batch_size=${MICRO_BATH_SIZE})
fi

set -o xtrace

HYDRA_FULL_ERROR=1 BIGNLP_CI=1 python3 main.py \
    fine_tuning=${RUN_MODEL}/${TEST_TASK} \
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
    fine_tuning.run.time_limit=${TIME_LIMIT} \
    fine_tuning.run.results_dir=${BASE_RESULTS_DIR}/${RUN_NAME} \
    fine_tuning.trainer.num_nodes=${NUM_NODES} \
    fine_tuning.trainer.max_steps=${MAX_STEPS} \
    fine_tuning.model.restore_from_path=${LANGUAGE_MODEL_PATH} \
    fine_tuning.model.tensor_model_parallel_size=${TP_SIZE} \
    fine_tuning.model.pipeline_model_parallel_size=${PP_SIZE} \
    fine_tuning.model.pipeline_model_parallel_split_rank=${PP_SPLIT_RANK} \
    fine_tuning.model.data.train_ds.micro_batch_size=${MICRO_BATH_SIZE} \
    fine_tuning.model.data.validation_ds.micro_batch_size=${MICRO_BATH_SIZE} \
    "${params[@]}" ${ADDITIONAL_PARAMS}
