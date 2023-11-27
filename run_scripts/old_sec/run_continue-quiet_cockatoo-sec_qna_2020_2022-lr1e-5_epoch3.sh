echo "****** STARTING ******" \
; echo "------------------" \
; wandb login 4935993d4748a2ef5a2349cf4f70ff3b9722d0e5 \
; WANDB_PROJECT_NAME="sec_continue_pretrain" \
; WANDB_EXP_NAME="qna-extr-2020-2022-qui_coc_lr1e-5_epoch3" \
; DATA_DIR="/home/hshin/projects/llmservice_modelalignment_ptune/datasets/sec_qna_jsonls" \
; cd /home/hshin/workspace/NeMo \
; export PYTHONPATH="/home/hshin/workspace/NeMo/.:${PYTHONPATH}" \
; python examples/nlp/language_modeling/megatron_gpt_continue_training.py \
				 --config-path=conf \
				 --config-name=megatron_gpt_config.yaml \
				 trainer.num_nodes=4 \
				 trainer.devices=8 \
				 trainer.max_steps=310 \
				 trainer.precision=bf16 \
				 restore_from_path=/home/hshin/projects/llmservice_modelalignment_ptune/checkpoints/8B/megatron_gpt_sft_quiet_cockatoo--validation_loss-1.325-step-1000-consumed_samples-127872.0_tp4.nemo \
				 model.micro_batch_size=2 \
				 model.global_batch_size=128 \
				 model.tensor_model_parallel_size=4 \
				 model.encoder_seq_length=4096 \
				 model.use_flash_attention=True \
				 model.optim.sched.name=CosineAnnealing \
				 model.optim.lr=1e-5 \
				 model.optim.sched.warmup_steps=100 \
				 model.optim.sched.min_lr=9e-6 \
				 exp_manager.create_wandb_logger=True \
				 exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT_NAME} \
         exp_manager.wandb_logger_kwargs.name=${WANDB_EXP_NAME} \
         exp_manager.explicit_log_dir=/home/hshin/results/${WANDB_PROJECT_NAME}/${WANDB_EXP_NAME} \
				 exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
				 model.data.data_prefix="{train:[1.0,${DATA_DIR}/sec_qna_2020-2022_num-200-shuffle_text-unfiltered-metaremoved_train-ln100_text_document], validation:[1.0,${DATA_DIR}/sec_qna_2023_Q1_num-100_text-unfiltered_txts-metaremoved_train-ln40_text_document], test:[1.0,${DATA_DIR}/sec_qna_2023_Q1_num-100_text-unfiltered_txts-metaremoved_val-ln10_text_document]}" \
				 model.data.splits_string=null \
				 model.data.index_mapping_dir=/home/hshin/outputs/index_mappings/${WANDB_PROJECT_NAME}/${WANDB_EXP_NAME}
# num_tokens = 36184864
# 1 epoch = 70 = 36184864 / (4096 * 2 * (8/4)*4 * (128/(2*(8/4)*4)))
