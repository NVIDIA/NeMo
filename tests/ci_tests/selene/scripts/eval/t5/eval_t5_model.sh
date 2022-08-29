params=()
if [[ ! -z $LOCAL_NEMO_PATH ]]; then
  params+=("container_mounts=[${LOCAL_NEMO_PATH}:/opt/bignlp/NeMo]")
fi
set -o xtrace

MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}

HYDRA_FULL_ERROR=1 BIGNLP_CI=1 python3 main.py \
    evaluation=${RUN_MODEL}/${TEST_TASK} \
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
    evaluation.trainer.num_nodes=${NUM_NODES} \
    evaluation.model.restore_from_path=${BASE_RESULTS_DIR}/${FINETUNE_JOB_DIR}/results/checkpoints/megatron_t5_glue.nemo \
    evaluation.model.data.validation_ds.micro_batch_size=${MICRO_BATCH_SIZE} \
    "${params[@]}"
