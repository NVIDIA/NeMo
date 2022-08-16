params=()


if [[ "$TEST_TASK" = "squad_real" ]]; then
  # Should come in here for the test prompt_learn.gpt3.126m_tp1_pp1_1node_squad_real
  # We need container mounts and LANGUAGE MODEL PATH from the config at gitlab ci yaml file
  params+=($CONTAINER_MOUNTS)
else
  # Should come in here for the test prompt_learn.gpt3.126m_tp1_pp1_1node_100steps_squad
  LOG_EVERY_N_STEPS=`expr $MAX_STEPS / 100`
  VAL_CHECK_INTERVAL=`expr $MAX_STEPS / 5`
  params+=(prompt_learning.trainer.log_every_n_steps=$LOG_EVERY_N_STEPS)
  params+=(prompt_learning.trainer.val_check_interval=$VAL_CHECK_INTERVAL)
  params+=(prompt_learning.trainer.max_steps=${MAX_STEPS})
  UPSTREAM_RUN_NAME=convert_${RUN_MODEL}_${RUN_MODEL_SIZE}_tp${TP_SIZE}_pp${PP_SIZE}
  LANGUAGE_MODEL_PATH=${BASE_RESULTS_DIR}/${UPSTREAM_RUN_NAME}/megatron_gpt.nemo
fi

set -o xtrace

HYDRA_FULL_ERROR=1 python3 main.py \
    prompt_learning=${RUN_MODEL}/squad \
    stages=["prompt_learning"] \
    bignlp_path=${GIT_CLONE_PATH} \
    data_dir=${BASE_RESULTS_DIR}/data \
    base_results_dir=${BASE_RESULTS_DIR} \
    "container='${BUILD_IMAGE_NAME_SRUN}'" \
    cluster.partition=${SLURM_PARTITION} \
    cluster.account=${SLURM_ACCOUNT} \
    cluster.gpus_per_task=null \
    cluster.gpus_per_node=null \
    cluster.job_name_prefix="${SLURM_ACCOUNT}-bignlp_ci:" \
    prompt_learning.run.name=${RUN_NAME} \
    prompt_learning.run.time_limit=${TIME_LIMIT} \
    prompt_learning.run.results_dir=${BASE_RESULTS_DIR}/${RUN_NAME} \
    prompt_learning.run.model_train_name=${RUN_MODEL}_${RUN_MODEL_SIZE} \
    prompt_learning.trainer.num_nodes=${NUM_NODES} \
    prompt_learning.model.language_model_path=${LANGUAGE_MODEL_PATH}\
    prompt_learning.model.tensor_model_parallel_size=${TP_SIZE} \
    prompt_learning.model.pipeline_model_parallel_size=${PP_SIZE} \
     "${params[@]}"
