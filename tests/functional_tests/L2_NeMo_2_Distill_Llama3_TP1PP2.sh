python tests/collections/llm/test_hf_import.py --hf_model /home/TestData/nlp/megatron_llama/llama-ci-hf --output_path /tmp/nemo2_ckpt

python scripts/llm/gpt_distillation.py \
  --name nemo2_llama_distill \
  --teacher_path /tmp/nemo2_ckpt \
  --student_path /tmp/nemo2_ckpt \
  --tokenizer gpt2 \
  --tp_size 1 \
  --cp_size 1 \
  --pp_size 2 \
  --devices 2 \
  --log_dir /tmp/distill_logs \
  --max_steps 5 \
  --gbs 4 \
  --mbs 1 \
  --data_paths 1.0 /home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document \
  --index_mapping_dir examples/nlp/language_modeling/gpt_index_mappings \
  --seq_length 2048 \
  --warmup_steps 1 \
  --val_check_interval 5 \
  --log_interval 5 \
  --limit_val_batches 2 \
  --legacy_ckpt
