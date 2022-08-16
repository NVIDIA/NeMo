set -o xtrace

DATA_DIR=/lustre/fsw/joc/big_nlp/gpt3/prepare_dataset/the_pile/train
# cut the last two underscores gpt3_126m_tp2_pp2_2node_100steps should become gpt_126m_tp2_pp2
EVAL_MODEL_NAME=${TRAIN_JOB_NAME}
# can WE JUST EQUAL THIS TO THE RUN NAME which will be eval_gpt3_126m_tp2_pp2_lambda
HYDRA_FULL_ERROR=1 python3 main.py \
    +ci_test=True \
    evaluation=${RUN_MODEL}/evaluate_${TEST_TASK} \
    run_data_preparation=False \
    run_training=False \
    run_conversion=False \
    run_finetuning=False \
    run_evaluation=True \
    bignlp_path=${GIT_CLONE_PATH} \
    data_dir=${DATA_DIR} \
    base_results_dir=${BASE_RESULTS_DIR} \
    "container='${BUILD_IMAGE_NAME_SRUN}'" \
    cluster.partition=${SLURM_PARTITION} \
    cluster.account=${SLURM_ACCOUNT} \
    cluster.gpus_per_task=null \
    cluster.gpus_per_node=null \
    cluster.job_name_prefix="${SLURM_ACCOUNT}-bignlp_ci:" \
    evaluation.run.name=${RUN_NAME} \
    evaluation.run.time_limit=${TIME_LIMIT} \
    evaluation.run.nodes=1 \
    evaluation.run.model_train_name=${EVAL_MODEL_NAME} \
    evaluation.run.train_dir=${BASE_RESULTS_DIR}/${TRAIN_JOB_NAME} \
    evaluation.run.results_dir=${BASE_RESULTS_DIR}/${RUN_NAME} \
    evaluation.model.eval_batch_size=16 \
    evaluation.model.tensor_model_parallel_size=${TP_SIZE} \
    evaluation.model.pipeline_model_parallel_size=${PP_SIZE}
