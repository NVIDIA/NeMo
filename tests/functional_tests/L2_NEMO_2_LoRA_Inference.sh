coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo scripts/llm/generate.py \
  --model_path /home/TestData/nemo2_ckpt/llama_lora_ci_checkpoint_v2/ \
  --tp 1 \
  --pp 1 \
  --devices 1 \
  --top_p 0.0 \
  --top_k 1 \
  --num_tokens_to_generate 3 \
  --legacy_ckpt
