echo "****** STARTING ******" \
; echo "------------------" \
; export HYDRA_FULL_ERROR=1 \
; wandb login 4935993d4748a2ef5a2349cf4f70ff3b9722d0e5 \
; WANDB_PROJECT_NAME="sec_sft" \
; WANDB_EXP_NAME="all-2020-2022-8b_qui_coc-lora_lr1e-5_epoch3--sft-lora--qna_2020-2022-x12_sampl-1.0_lr5e-6_epoch1000" \
; DATA_DIR_QNA="/home/hshin/projects/llmservice_modelalignment_ptune/datasets/sec_qna_jsonls" \
; DATA_DIR_QUIET_COCKATOO="/home/hshin/datasets/quiet-cockatoo" \
; cd /home/hshin/workspace/NeMo \
; export PYTHONPATH="/home/hshin/workspace/NeMo/.:${PYTHONPATH}" \
; python examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py \
   --config-path=conf \
   --config-name=megatron_gpt_peft_tuning_config \
	 ++model.use_flash_attention=True \
	 model.answer_only_loss=True \
   model.data.train_ds.num_workers=0 \
   model.data.train_ds.prompt_template="\{input\} \{output\}" \
   model.data.train_ds.label_key="output" \
   model.data.train_ds.truncation_field="input" \
   model.data.train_ds.file_names=["${DATA_DIR_QNA}/sec_qna_2020-2022_train_num-200-shuffle_train_input-outupt_aug12x.jsonl"] \
   model.data.train_ds.max_seq_length=2048 \
   model.data.train_ds.concat_sampling_probabilities=[1.0] \
   model.data.validation_ds.file_names=["${DATA_DIR_QNA}/sec_qna_2020-2022_train_num-200-shuffle_val_input-outupt_aug12x_quiet_cockatoo_validation.jsonl"] \
   model.data.validation_ds.max_seq_length=2048 \
   model.data.validation_ds.names=["quiet_cockatoo_sec_qna_2020-2022"] \
   model.data.validation_ds.num_workers=0 \
   model.micro_batch_size=6 \
   model.global_batch_size=192 \
   model.tensor_model_parallel_size=4 \
	 model.restore_from_path=/home/hshin/results/sec_continue_pretrain/all-2020-2022-qui_coc_lr1e-5_epoch3/checkpoints/megatron_gpt.nemo \
	 model.optim.sched.name=CosineAnnealing \
	 model.optim.lr=5e-6 \
	 model.optim.sched.warmup_steps=50 \
   model.optim.sched.min_lr=9e-7 \
	 model.peft.peft_scheme=lora \
	 name=${WANDB_EXP_NAME} \
   trainer.devices=8 \
   trainer.num_nodes=4 \
	 trainer.max_steps=500 \
   trainer.precision=bf16 \
   trainer.val_check_interval=100 \
	 exp_manager.early_stopping_callback_params.monitor="reduced_train_loss" \
   exp_manager.exp_dir=/home/hshin/results/${WANDB_PROJECT_NAME}/${WANDB_EXP_NAME} \
   exp_manager.create_wandb_logger=True \
   exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT_NAME} \
   exp_manager.wandb_logger_kwargs.name=${WANDB_EXP_NAME} \
   exp_manager.explicit_log_dir=/home/hshin/results/${WANDB_PROJECT_NAME}/${WANDB_EXP_NAME} \
   exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
   exp_manager.checkpoint_callback_params.save_top_k=5 \
	 exp_manager.checkpoint_callback_params.mode=min
# num_tokens: sec_qna_2020-2022 = 38649
## num_tokens: quiet_cockatoo = 89311852
## num_tokens: sum = 89350501
# 1 epoch ~= 0.5 = 38649 / (2048 * 6 * (8/4)*4)

