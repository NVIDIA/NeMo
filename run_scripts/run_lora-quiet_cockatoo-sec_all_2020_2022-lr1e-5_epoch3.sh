echo "****** STARTING ******" \
; echo "------------------" \
; wandb login 4935993d4748a2ef5a2349cf4f70ff3b9722d0e5 \
; WANDB_PROJECT_NAME="sec_continue_pretrain" \
; WANDB_EXP_NAME="all-2020-2022-qui_coc-lora_lr1e-5_epoch3" \
; DATA_DIR="/home/hshin/projects/llmservice_modelalignment_ptune/datasets/sec_parsed" \
; cd /home/hshin/workspace/NeMo \
; export PYTHONPATH="/home/hshin/workspace/NeMo/.:${PYTHONPATH}" \
; python examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py \
   --config-path=conf \
   --config-name=megatron_gpt_peft_tuning_config \
   ++model.data.continue_pretrain=True \
   +model.data.data_prefix="{train:[0.33,${DATA_DIR}/2020/sec-2020_text_document,0.33,${DATA_DIR}/2021/sec-2021_text_document,0.33,${DATA_DIR}/2022/sec-2022_text_document], validation:[1.0,${DATA_DIR}/2018/sec-2018_text_document], test:[1.0,${DATA_DIR}/2017/sec-2017_text_document]}" \
   +model.data.index_mapping_dir=/home/hshin/outputs/index_mappings/${WANDB_PROJECT_NAME}/${WANDB_EXP_NAME} \
   +model.data.data_impl=mmap \
   +model.data.splits_string=null \
   +model.data.seq_length=4097 \
   model.data.train_ds.file_names=[${DATA_DIR}/dummy.jsonl] \
   model.data.validation_ds.file_names=[${DATA_DIR}/dummy.jsonl] \
   model.data.train_ds.concat_sampling_probabilities=[1.0] \
   +model.data.splits_string=null \
   model.micro_batch_size=2 \
   model.global_batch_size=128 \
   model.tensor_model_parallel_size=4 \
   model.restore_from_path=/home/hshin/projects/llmservice_modelalignment_ptune/checkpoints/8B/megatron_gpt_sft_quiet_cockatoo--validation_loss-1.325-step-1000-consumed_samples-127872.0_tp4.nemo \
   model.optim.sched.name=CosineAnnealing \
   model.optim.lr=1e-4 \
   model.optim.sched.warmup_steps=500 \
   model.optim.sched.min_lr=9e-6 \
   model.peft.peft_scheme=lora \
   name=${WANDB_EXP_NAME} \
   trainer.devices=8 \
   trainer.num_nodes=16 \
   trainer.max_steps=29300 \
   trainer.precision=bf16 \
   trainer.val_check_interval=80 \
   exp_manager.exp_dir=/home/hshin/results/${WANDB_PROJECT_NAME}/${WANDB_EXP_NAME} \
   exp_manager.create_wandb_logger=True \
   exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT_NAME} \
   exp_manager.wandb_logger_kwargs.name=${WANDB_EXP_NAME} \
   exp_manager.explicit_log_dir=/home/hshin/results/${WANDB_PROJECT_NAME}/${WANDB_EXP_NAME} \
   exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
   exp_manager.checkpoint_callback_params.save_top_k=5 \
	 exp_manager.checkpoint_callback_params.mode=min
# num_tokens = 5008421041
# 1 epoch ~= 9600 = 5008421041 / (4096 * 2 * (8/4)*16 * (128/(2*(8/4)*16)))
