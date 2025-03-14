coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/lora_mistralai.py \
    --max-steps 3 \
    --tp 2 \
    --mbs 1 \
    --model mistral \
    --dist-opt
