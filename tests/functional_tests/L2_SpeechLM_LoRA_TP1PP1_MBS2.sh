coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/speechlm/speech_to_text_llm_train.py \
  --train_manifest /home/TestData/speechlm/speechlm_data/speech_to_text_debug2/debug_2.json \
  --val_manifest /home/TestData/speechlm/speechlm_data/speech_to_text_debug2/debug_2.json \
  --restore_path /home/TestData/nemo2_ckpt/llama_68M \
  --devices 2 \
  --max_steps 500 \
  --experiment_dir /tmp/nemo2_speechlm_lora/$RUN_ID \
  --peft lora \
  --tp_size 1 \
  --pp_size 1 \
  --mbs 2

coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/speechlm/speech_to_text_llm_train.py \
  --train_manifest /home/TestData/speechlm/speechlm_data/speech_to_text_debug2/debug_2.json \
  --val_manifest /home/TestData/speechlm/speechlm_data/speech_to_text_debug2/debug_2.json \
  --restore_path /home/TestData/nemo2_ckpt/llama_68M \
  --devices 2 \
  --max_steps 600 \
  --experiment_dir /tmp/nemo2_speechlm_lora/$RUN_ID \
  --peft lora \
  --tp_size 1 \
  --pp_size 1 \
  --mbs 2
