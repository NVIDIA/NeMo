coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/evaluation/test_evaluation.py \
    --nemo2_ckpt_path=/home/TestData/nemo2_ckpt/llama3-1b-lingua \
    --max_batch_size=4 \
    --trtllm_dir='/tmp/trtllm_dir' \
    --eval_type='arc_challenge' \
    --limit=1
