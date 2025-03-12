TRANSFORMERS_OFFLINE=1 \
      python tests/collections/llm/hf/sft_fsdp2.py --model /home/TestData/nlp/hf_gemma/hf_gemma_2b --max-steps 10 --devices 2
TRANSFORMERS_OFFLINE=1 HF_HOME=/home/TestData/automodel/hf_home python examples/llm/sft/automodel.py --model /home/TestData/nlp/hf_gemma/hf_gemma_2b --max-steps 10 --devices 2 --strategy fsdp2
