params=()
if [[ ! -z $LOCAL_NEMO_PATH ]]; then
  params+=("container_mounts=[${LOCAL_NEMO_PATH}:/opt/bignlp/NeMo]")
fi
DATA_DIR=/lustre/fsw/joc/big_nlp/mt5/dataset/ci_data

set -o xtrace

HYDRA_FULL_ERROR=1 BIGNLP_CI=1 python3 main.py \
    conversion=${RUN_MODEL}/convert_${RUN_MODEL} \
    stages=["conversion"] \
    bignlp_path=${GIT_CLONE_PATH} \
    data_dir=${DATA_DIR} \
    base_results_dir=${BASE_RESULTS_DIR} \
    "container='${BUILD_IMAGE_NAME_SRUN}'" \
    cluster.partition=${SLURM_PARTITION} \
    cluster.account=${SLURM_ACCOUNT} \
    cluster.gpus_per_task=null \
    cluster.gpus_per_node=null \
    cluster.job_name_prefix="${SLURM_ACCOUNT}-bignlp_ci:" \
    conversion.run.name=${RUN_NAME} \
    conversion.run.time_limit=${TIME_LIMIT} \
    conversion.run.nodes=1 \
    conversion.run.model_train_name=${RUN_NAME} \
    conversion.run.train_dir=${BASE_RESULTS_DIR}/${UPSTREAM_RUN_NAME} \
    conversion.run.results_dir=${BASE_RESULTS_DIR}/${RUN_NAME} \
    conversion.model.tensor_model_parallel_size=${TP_SIZE} \
    conversion.model.pipeline_model_parallel_size=${PP_SIZE} \
    "${params[@]}"
