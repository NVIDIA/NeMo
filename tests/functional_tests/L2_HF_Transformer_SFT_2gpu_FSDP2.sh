TRANSFORMERS_OFFLINE=1 torchrun --nproc-per-node=2 tests/collections/llm/hf/sft.py --model /home/TestData/akoumparouli/hf_mixtral_2l/ --max-steps 3 --devices 2 --strategy fsdp2
TRANSFORMERS_OFFLINE=1 torchrun --nproc-per-node=2 tests/collections/llm/hf/sft.py --model /home/TestData/akoumparouli/hf_mixtral_2l/ --max-steps 3 --devices 2 --strategy fsdp2 --auto-resume
