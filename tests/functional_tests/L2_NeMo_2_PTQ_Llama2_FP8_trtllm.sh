python tests/collections/llm/test_hf_import.py --hf_model /home/TestData/nlp/megatron_llama/llama-ci-hf --output_path /tmp/nemo2_ckpt

python scripts/llm/ptq.py -nc /tmp/nemo2_ckpt -algo fp8 -out /tmp/nemo2_ptq_engine --export_format trtllm --legacy_ckpt
