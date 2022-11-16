params=()

LANGUAGE_MODEL_PATH=${BASE_RESULTS_DIR}/${FINETUNE_JOB_DIR}/results/checkpoints/megatron_t5_squad.nemo
if [[ "$TEST_TASK" = "squad_real" ]]; then
  # Should come in here for the test prompt_learn.gpt3.126m_tp1_pp1_1node_squad_real
  # We need container mounts and LANGUAGE MODEL PATH from the config at gitlab ci yaml file
  params+=("container_mounts=[${CONTAINER_MOUNTS}]")
elif [[ "$TEST_TASK" = "squad_ckpt" ]]; then
    LANGUAGE_MODEL_PATH=null
    FINETUNE_DIR=${BASE_RESULTS_DIR}/${FINETUNE_JOB_DIR}
    CHECKPOINT_DIR=${FINETUNE_DIR}/results/checkpoints
    CHECKPOINT_NAME=$(ls -tr ${CHECKPOINT_DIR}/* | tail -1)
    HPARAMS_FILE=${FINETUNE_DIR}/results/hparams.yaml
    params+=(evaluation.model.pretrained_checkpoint.checkpoint_dir=$CHECKPOINT_DIR)
    params+=(evaluation.model.pretrained_checkpoint.checkpoint_name=${CHECKPOINT_NAME})
    params+=(evaluation.model.pretrained_checkpoint.hparams_file=$HPARAMS_FILE)
  fi
  if [[ ! -z $LOCAL_NEMO_PATH ]]; then
    params+=("container_mounts=[${LOCAL_NEMO_PATH}:/opt/bignlp/NeMo]")
  fi
else
  if [[ ! -z $LOCAL_NEMO_PATH ]]; then
    params+=("container_mounts=[${LOCAL_NEMO_PATH}:/opt/bignlp/NeMo]")
  fi
fi
set -o xtrace
PP_SPLIT_RANK=${PP_SPLIT_RANK:-`expr ${PP_SIZE} / 2`}

HYDRA_FULL_ERROR=1 BIGNLP_CI=1 python3 main.py \
    evaluation=${RUN_MODEL}/squad \
    stages=["evaluation"] \
    bignlp_path=${GIT_CLONE_PATH} \
    data_dir=${BASE_RESULTS_DIR}/data \
    base_results_dir=${BASE_RESULTS_DIR} \
    "container='${BUILD_IMAGE_NAME_SRUN}'" \
    cluster.partition=${SLURM_PARTITION} \
    cluster.account=${SLURM_ACCOUNT} \
    cluster.gpus_per_task=null \
    cluster.gpus_per_node=null \
    cluster.job_name_prefix="${SLURM_ACCOUNT}-bignlp_ci:" \
    evaluation.run.time_limit=${TIME_LIMIT} \
    evaluation.run.results_dir=${BASE_RESULTS_DIR}/${RUN_NAME} \
    evaluation.model.tensor_model_parallel_size=${TP_SIZE} \
    evaluation.model.pipeline_model_parallel_size=${PP_SIZE} \
    evaluation.model.pipeline_model_parallel_split_rank=${PP_SPLIT_RANK} \
    evaluation.model.restore_from_path=${LANGUAGE_MODEL_PATH} \
    "${params[@]}"
