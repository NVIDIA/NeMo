
mkdir -p /tmp/llm_tests/llama_pretrain_results \
export FAULT_TOL_CFG_PATH="/tmp/llm_tests/llama_pretrain_results/sample_job_ft_cfg.yml"; \
export FAULT_TOL_FINISHED_FLAG_FILE="/tmp/llm_tests/llama_pretrain_results/sample_job_finished_flag"; \
python tests/collections/llm/test_fault_nvrx.py \
--devices=2 \
--check-report=True \
--experiment-dir=/tmp/llm_tests/llama_pretrain_results \
--data-path=/home/TestData/nlp/megatron_llama/data/rp2_sample_sentencepiece_preproc_text_document \
--tokenizer-path=/home/TestData/nlp/megatron_llama/tokenizer.model \
--index-mapping-dir=/tmp/llm_tests/llama_index_mappings \
2>&1 | tee /tmp/llm_tests/llama_pretrain_results/run.log \
