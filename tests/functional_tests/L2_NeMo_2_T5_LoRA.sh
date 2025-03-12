python tests/collections/llm/megatron_t5_finetuning.py \
    --devices=2 \
    --max-steps=250 \
    --peft=lora \
    --experiment-dir=tests/collections/llm/t5_peft_results/$RUN_ID \
    --checkpoint-path=/home/TestData/nlp/megatron_t5/220m/nemo2.0_t5_220m_padding_attnmasktype_150steps
