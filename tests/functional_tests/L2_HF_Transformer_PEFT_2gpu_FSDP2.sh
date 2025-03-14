TRANSFORMERS_OFFLINE=1 coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/hf/peft_hf.py \
  --model /home/TestData/akoumparouli/hf_mixtral_2l/ \
  --max-steps 3 \
  --devices 2 \
  --strategy fsdp2 \
  --ckpt-folder /tmp/hf_peft_ckpt_fsdp2

TRANSFORMERS_OFFLINE=1 coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/hf/peft_hf.py \
  --model /home/TestData/akoumparouli/hf_mixtral_2l/ \
  --max-steps 3 \
  --devices 2 \
  --strategy fsdp2 \
  --ckpt-folder /tmp/hf_peft_ckpt_fsdp2 --auto-resume

TRANSFORMERS_OFFLINE=1 HF_HOME=/home/TestData/automodel/hf_home coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/llm/peft/automodel.py \
  --model /home/TestData/akoumparouli/hf_mixtral_2l/ \
  --max-steps 3 \
  --devices 2 \
  --strategy fsdp2
