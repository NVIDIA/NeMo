
python tests/collections/llm/gpt/model/megatron_ssm_finetuning.py \
--devices 1 \
--max-steps 10 \
--experiment-dir /tmp/nlp_megatron_mamba_nemo-ux-mamba_cicd_test_sft/${{ github.run_id }} \
--model-path /home/TestData/nlp/megatron_mamba/model_optim_rng.pt \
--ckpt_load_strictness log_all
