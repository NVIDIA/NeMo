coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/test_hf_import.py --hf_model /home/TestData/nlp/megatron_llama/llama-ci-hf --output_path /tmp/nemo2_ckpt

coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo scripts/llm/gpt_prune.py \
  --restore_path /tmp/nemo2_ckpt \
  --legacy_ckpt \
  --tokenizer gpt2 \
  --tp_size 1 \
  --pp_size 2 \
  --devices 2 \
  --seq_length 128 \
  --mbs 2 \
  --num_train_samples 32 \
  --data_paths 1.0 /home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document \
  --index_mapping_dir examples/nlp/language_modeling/gpt_index_mappings \
  --prune_hidden_size 64 \
  --prune_ffn_hidden_size 128 \
  --prune_num_attention_heads 4 \
  --prune_num_query_groups 4 \
  --prune_num_layers 2 \
  --save_path /tmp/pruned-llama
