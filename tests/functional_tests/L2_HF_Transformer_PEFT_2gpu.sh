TRANSFORMERS_OFFLINE=1 coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/hf/peft_hf.py \
  --model /home/TestData/akoumparouli/hf_mixtral_2l/ \
  --max-steps 3 \
  --devices 2 \
  --strategy ddp \
  --ckpt-folder /tmp/hf_peft_ckpt_ddp

TRANSFORMERS_OFFLINE=1 coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/hf/peft_hf.py \
  --model /home/TestData/akoumparouli/hf_mixtral_2l/ \
  --max-steps 3 \
  --devices 2 \
  --strategy ddp \
  --ckpt-folder /tmp/hf_peft_ckpt_ddp --auto-resume
