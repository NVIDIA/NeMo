HYDRA_FULL_ERROR=1 python3 main.py \
    +ci_test=True \
    training=gpt3/126m \
    run_data_preparation=False \
    run_training=True \
    run_conversion=False \
    run_finetuning=False \
    run_evaluation=False \
    bignlp_path=${GIT_CLONE_PATH} \
    data_dir=/lustre/fsw/joc/yuya/bignlp/bignlp-scripts_ci/data \
    base_results_dir=/lustre/fsw/joc/big_nlp/bignlp_ci/results \
    "container='${BUILD_IMAGE_NAME_SRUN}'" \
    cluster.partition=luna \
    cluster.account=joc \
    cluster.gpus_per_task=null \
    cluster.job_name_prefix="joc-bignlp_ci:" \
    training.run.name=ci_gpt3_126m_deterministic \
    training.run.time_limit="30:00" \
    training.trainer.num_nodes=${NUM_NODES} \
    training.trainer.max_steps=50 \
    training.model.data.data_prefix=[1.0,/lustre/fsw/joc/yuya/bignlp/bignlp-scripts_ci/data/my-gpt3_00_text_document] 

srun --nodes=1 --ntasks-per-node=1 --account=${SLURM_ACCOUNT} --partition=${SLURM_PARTITION} --job-name="${SLURM_ACCOUNT}-bignlp_ci:pytest_gpt3_126m" --time=20:00 --dependency=singleton --exclusive --mem=0 --overcommit --container-image=${BUILD_IMAGE_NAME_SRUN} --container-mounts=.:/workspace/bignlp-scripts,/lustre/fsw/joc/big_nlp/bignlp_ci/results:/workspace/bignlp-scripts/results sh -c "cd /workspace/bignlp-scripts; pytest tests/ci_tests"
