TRANSFORMERS_OFFLINE=1 python tests/collections/llm/hf/sft.py --model /home/TestData/akoumparouli/hf_mixtral_2l/ --max-steps 3 --devices 2 --strategy ddp
TRANSFORMERS_OFFLINE=1 python tests/collections/llm/hf/sft.py --model /home/TestData/akoumparouli/hf_mixtral_2l/ --max-steps 3 --devices 2 --strategy ddp --auto-resume
