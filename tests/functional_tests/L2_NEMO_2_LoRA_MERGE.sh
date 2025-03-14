coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/peft/lora_merge.py \
    --lora_checkpoint_path=/home/TestData/nemo2_ckpt/llama_lora_ci_checkpoint_v2/ \
    --output_path=/tmp/nemo2_lora_merge/$RUN_ID \
    --legacy_ckpt
