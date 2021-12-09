#/bin/bash

GPUS=1
MAX_STEPS=2000
PROMPT_LENGTH=10
RESTORE_PATH='megatron_gpt.nemo'

echo "Prompt tuning starting"
python megatron_gpt_prompt_tuning.py \
	trainer.gpus=$GPUS \
	cfg.trainer.max_steps=$MAX_STEPS \
	cfg.restore_from_path=$RESTORE_PATH \
	cfg.model.use_soft_prompts=True \
	cfg.model.prompt_length=$PROMPT_LENGTH \
	cfg.model.data.data_prefix=None \
	cfg.model.data.train_ds='prompt_tuning_ner_train.json' \
	cfg.model.data.valid_ds='prompt_tuning_ner_val.json' \
	cfg.model.data.test_ds='prompt_tuning_ner_test.json' \
	cfg.model.data.batch_size=32 \
	cfg.model.optim.lr=2e-4 \
	cfg.model.optim.sched.min_lr=2e-6 \
	cfg.model.optim.sched.warmup_steps=100 \
	cfg.model.optim.sched.constant_steps=1000 \


