TORCHDYNAMO_DISABLE=1 coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/lightning/test_ddp_parity_checker.py \
    --vocab-path=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
    --merges-path=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
    --data-path=/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document
