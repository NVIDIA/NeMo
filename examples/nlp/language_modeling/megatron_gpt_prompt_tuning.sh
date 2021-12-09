#/bin/bash

GPUS=2
MAX_STEPS=3000
PROMPT_LENGTH=10
RESTORE_PATH='megatron_gpt.nemo'

echo "Prompt tuning starting"
python megatron_gpt_prompt_tuning.py \
	--config-name=megatron_gpt_config \
	trainer.gpus=$GPUS \
	trainer.max_steps=$MAX_STEPS \
	restore_from_path=$RESTORE_PATH \
	+model.use_soft_prompts=True \
	+model.prompt_length=$PROMPT_LENGTH \
	+model.new_prompt_tags=['NER-Yes-No, NER-Complete'] \
	+model.new_prompt_init_text=['named entities yes no, None'] \
	+model.new_prompt_init_methods=['text, random'] \
	model.data.data_prefix=None \
	+model.data.train_ds='prompt_tuning_ner_train.json' \
	+model.data.valid_ds='prompt_tuning_ner_val.json' \
	+model.data.test_ds='prompt_tuning_ner_test.json' \
	+model.data.batch_size=32 \
	model.optim.lr=2e-3 \
	model.optim.sched.min_lr=2e-6 \
	model.optim.sched.warmup_steps=200 \
	model.optim.sched.constant_steps=1000 \
	model.encoder_seq_length=2048


