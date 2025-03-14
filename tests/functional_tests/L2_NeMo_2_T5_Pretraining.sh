coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/megatron_t5_pretraining.py \
    --devices=2 \
    --max-steps=3 \
    --experiment-dir=tests/collections/llm/t5_pretrain_results/$RUN_ID \
    --data-path=/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document \
    --index-mapping-dir=tests/collections/llm/t5_index_mappings/$RUN_ID

coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/megatron_t5_pretraining.py \
    --devices=2 \
    --max-steps=6 \
    --experiment-dir=tests/collections/llm/t5_pretrain_results/$RUN_ID \
    --data-path=/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document \
    --index-mapping-dir=tests/collections/llm/t5_index_mappings/$RUN_ID
