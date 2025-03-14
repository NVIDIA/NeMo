pip uninstall -y apex ## TODO: remove when apex is no longer a dependency
pip uninstall -y transformer_engine

coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/megatron_gpt_pretraining.py \
    --devices=2 \
    --max-steps=3 \
    --experiment-dir=tests/collections/llm/gpt_pretrain_results \
    --vocab-path=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
    --merges-path=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
    --data-path=/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document \
    --index-mapping-dir=tests/collections/llm/gpt_index_mappings \
    --no-masked-softmax-fusion

coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/megatron_gpt_pretraining.py \
    --devices=2 \
    --max-steps=6 \
    --experiment-dir=tests/collections/llm/gpt_pretrain_results \
    --vocab-path=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
    --merges-path=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
    --data-path=/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document \
    --index-mapping-dir=tests/collections/llm/gpt_index_mappings \
    --no-masked-softmax-fusion
