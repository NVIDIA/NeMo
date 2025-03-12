
python tests/collections/llm/llama3_pretraining.py \
--seq-length 1024 \
--devices=2 \
--max-steps=6 \
--early-stop=3 \
--experiment-dir=/tmp/llm_tests/llama_pretrain_results \
--data-path=/home/TestData/nlp/megatron_llama/data/rp2_sample_sentencepiece_preproc_text_document \
--tokenizer-path=/home/TestData/nlp/megatron_llama/tokenizer.model \
--index-mapping-dir=/tmp/llm_tests/llama_index_mappings \

python tests/collections/llm/llama3_pretraining.py \
--seq-length 1024 \
--devices=2 \
--max-steps=6 \
--experiment-dir=/tmp/llm_tests/llama_pretrain_results \
--data-path=/home/TestData/nlp/megatron_llama/data/rp2_sample_sentencepiece_preproc_text_document \
--tokenizer-path=/home/TestData/nlp/megatron_llama/tokenizer.model \
--index-mapping-dir=/tmp/llm_tests/llama_index_mappings \
--cp 1 --tp 2 --sp 1
