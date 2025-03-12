python tests/collections/llm/megatron_t5_pretraining.py \
--devices=2 \
--max-steps=3 \
--experiment-dir=tests/collections/llm/t5_pretrain_results/${{ github.run_id }} \
--data-path=/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document \
--index-mapping-dir=tests/collections/llm/t5_index_mappings/${{ github.run_id }}

python tests/collections/llm/megatron_t5_pretraining.py \
--devices=2 \
--max-steps=6 \
--experiment-dir=tests/collections/llm/t5_pretrain_results/${{ github.run_id }} \
--data-path=/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document \
--index-mapping-dir=tests/collections/llm/t5_index_mappings/${{ github.run_id }}
