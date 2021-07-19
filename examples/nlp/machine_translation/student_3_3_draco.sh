#!/bin/bash
WANDB_LOGIN=1589819cfa34108320cd27634a3f764a29b211d8
wandb login ${WANDB_LOGIN}

# Hyperparams
TOKENS_IN_BATCH=8000
LEARNING_RATE=4e-4
STEPS=300000
WARMUP_STEPS=30000
DISTILLATION_LOSS_WEIGHT=1.0
STUDENT_LOSS_WEIGHT=0.0
TEMPERATURE=2

# logging
PROJECT=nemo_nmt_enc_dec
EXPNAME=de_en_teacher_24x6_student_12x6_dl_${DISTILLATION_LOSS_WEIGHT}_sl_${STUDENT_LOSS_WEIGHT}_temp_${TEMPERATURE}


CUDA_VISIBLE_DEVICES=0,1,2,3 python enc_dec_nmt_distill.py \
	--config-path=conf \
	--config-name=aayn_base_distill \
	do_training=true \
	trainer.gpus=4 \
	+trainer.max_steps=${STEPS} \
	+trainer.val_check_interval=10000 \
	model.beam_size=4 \
	model.max_generation_delta=6 \
	model.label_smoothing=0.1 \
	model.src_language='de' \
	model.tgt_language='en' \
	model.preproc_out_dir=/raid/data/wmt21_de_en_yttm_tokens_${TOKENS_IN_BATCH} \
	model.encoder.hidden_size=1024 \
	model.encoder.inner_size=4096 \
	model.encoder.num_attention_heads=16 \
	model.encoder.num_layers=12 \
	model.encoder.ffn_dropout=0.1 \
	model.encoder.pre_ln=true \
	model.encoder_tokenizer.vocab_size=32000 \
	model.decoder_tokenizer.vocab_size=32000 \
	model.decoder.hidden_size=1024 \
	model.decoder.inner_size=4096 \
	model.decoder.num_attention_heads=16 \
	model.decoder.num_layers=6 \
	model.decoder.attn_layer_dropout=0.1 \
	model.decoder.ffn_dropout=0.1 \
	model.train_ds.use_tarred_dataset=true \
	model.train_ds.src_file_name=/raid/data/wmt21/train.dedup.de \
	model.train_ds.tgt_file_name=/raid/data/wmt21/train.dedup.en \
	model.train_ds.tokens_in_batch=${TOKENS_IN_BATCH} \
	model.validation_ds.src_file_name=[/raid/data/wmt21/val/newstest2020-en-de.clean.tok.ref,/raid/data/wmt21/val/newstest2019-en-de.clean.tok.ref,/raid/data/wmt21/val/newstest2018-en-de.clean.tok.ref,/raid/data/wmt21/val/newstest2014-en-de.clean.tok.ref,/raid/data/wmt21/val/newstest2013-en-de.clean.tok.ref] \
	model.validation_ds.tgt_file_name=[/raid/data/wmt21/val/newstest2020-en-de.clean.tok.src,/raid/data/wmt21/val/newstest2019-en-de.clean.tok.src,/raid/data/wmt21/val/newstest2018-en-de.clean.tok.src,/raid/data/wmt21/val/newstest2014-en-de.clean.tok.src,/raid/data/wmt21/val/newstest2013-en-de.clean.tok.src] \
	~model.test_ds \
	model.optim.lr=${LEARNING_RATE} \
	+model.optim.sched.warmup_steps=$WARMUP_STEPS \
  	~model.optim.sched.warmup_ratio \
	model.distillation.model_path=/raid/nemo_models/en_de_24x6.nemo \
	model.distillation.distillation_loss_weight=${DISTILLATION_LOSS_WEIGHT} \
	model.distillation.student_train_loss_weight=${STUDENT_LOSS_WEIGHT} \
	model.distillation.temperature=${TEMPERATURE} \
	+exp_manager.explicit_log_dir=/raid/nemo_experiments/${EXPNAME} \
	+exp_manager.create_wandb_logger=true \
	+exp_manager.wandb_logger_kwargs.name=${EXPNAME} \
	+exp_manager.wandb_logger_kwargs.project=${PROJECT} \
	+exp_manager.create_checkpoint_callback=True \
	+exp_manager.resume_if_exists=True \
	+exp_manager.resume_ignore_no_checkpoint=True \
	+exp_manager.checkpoint_callback_params.monitor=val_sacreBLEU \
	+exp_manager.checkpoint_callback_params.save_top_k=1 \
	+exp_manager.checkpoint_callback_params.mode=max 
