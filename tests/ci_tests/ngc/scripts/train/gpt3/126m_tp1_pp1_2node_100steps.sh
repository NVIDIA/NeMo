RUN_CMD=" \
 python3 main.py \
    training=gpt3/126m \
    run_training=True \
    run_data_preparation=False \
    run_conversion=False \
    run_finetuning=False \
    run_evaluation=False \
    bignlp_path=/mount/results/${NGC_PROJECT_DIR} \
    data_dir=/mount/data/the_pile_gpt3 \
    base_results_dir=/mount/results/${BASE_RESULTS_DIR} \
    cluster_type=bcp \
    training.trainer.num_nodes=2 \
    training.model.tokenizer.vocab_file=/mount/data/bpe/vocab.json \
    training.model.tokenizer.merge_file=/mount/data/bpe/merges.txt \
    training.run.name=${RUN_NAME} \
    training.trainer.max_steps=100 \
    training.trainer.log_every_n_steps=1 \
    training.trainer.val_check_interval=20 \
    training.trainer.limit_val_batches=5 \
    training.model.tensor_model_parallel_size=1 \
    training.model.pipeline_model_parallel_size=1 \
"
ngc batch run \
  --name "${RUN_NAME}-${CI_PIPELINE_ID}" \
  --preempt RUNONCE \
  --total-runtime 1800s \
  --ace nv-eagle \
  --instance dgxa100.40g.8.norm \
  --result /results \
  --array-type "PYTORCH" \
  --replicas "2" \
  --image "$NVCR_IMAGE_NAME" \
  --workspace l6md0daQSkmS-aFpe7ZpnQ:/mount/data:RW \
  --workspace ${NGC_CI_RESULT_WORKSPACE}:/mount/results:RW \
  --commandline " \
  set -x; \
  export RESULTS_DIR=/mount/results/${RESULTS_DIR}; \
  export RUN_TASK=${RUN_TASK}; \
  cd /mount/results/${NGC_PROJECT_DIR}; \
  ${RUN_CMD}; \
  python3 tests/ci_tests/utils/convert_ci_metric_to_json.py tests/ci_tests/ngc/pytest/${RUN_TASK}/${RUN_MODEL}/test_${RUN_JOB_NAME}.py; \
  pytest tests/ci_tests/ngc/pytest/${RUN_TASK}/${RUN_MODEL}/test_${RUN_JOB_NAME}.py; \
"
