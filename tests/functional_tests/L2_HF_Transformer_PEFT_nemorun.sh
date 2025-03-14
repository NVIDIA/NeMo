TRANSFORMERS_OFFLINE=1 coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/hf/peft_nemorun.py \
    --model /home/TestData/akoumparouli/hf_mixtral_2l/ --max-steps 3
