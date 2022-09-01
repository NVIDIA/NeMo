HYDRA_FULL_ERROR=1 BIGNLP_CI=1 python3 main.py \
    search_config=t5/2.8b \
    run_training_hp_search=True \
    run_inference_hp_search=False \
    bignlp_hp_tool_path=${GIT_CLONE_PATH} \
    data_dir=/lustre/fsw/joc/big_nlp/t5/dataset/Pile \
    base_results_dir=${BASE_RESULTS_DIR} \
    "training_container='${BUILD_IMAGE_TRAINING_NAME_SRUN}'" \
    cluster.partition=${SLURM_PARTITION} \
    cluster.account=${SLURM_ACCOUNT} \
    cluster.job_name_prefix="${SLURM_ACCOUNT}-bignlp_hp_tool:" \
    cluster.gpus_per_task=null \
    cluster.gpus_per_node=null \
    search_config.train_settings.gpu_memory_gb=80 \
    search_config.train_settings.limit_search_runs=3