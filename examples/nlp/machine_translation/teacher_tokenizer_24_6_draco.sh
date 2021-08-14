#!/bin/bash
# Hyperparams
TOKENS_IN_BATCH=2000
LEARNING_RATE=4e-4
STEPS=300000
WARMUP_STEPS=30000

SLURM_GPUS_PER_NODE=2

python enc_dec_nmt.py \
	--config-path=conf \
	--config-name=aayn_base \
	do_training=false \
	trainer.gpus=${SLURM_GPUS_PER_NODE} \
	~trainer.max_epochs \
	+trainer.max_steps=${STEPS} \
	+trainer.val_check_interval=1000 \
	+trainer.accumulate_grad_batches=1 \
	model.src_language=de \
	model.tgt_language=en \
	model.beam_size=4 \
	model.max_generation_delta=6 \
	model.label_smoothing=0.1 \
	model.preproc_out_dir=/raid/preproc_data/wmt21_de_en_yttm_tokens_${TOKENS_IN_BATCH} \
	model.encoder.hidden_size=1024 \
	model.encoder.hidden_size=1536 \
	model.encoder.inner_size=6144 \
	model.encoder.num_attention_heads=24 \
	model.encoder.num_layers=24 \
	model.encoder.ffn_dropout=0.1 \
	model.encoder.pre_ln=true \
	model.encoder_tokenizer.vocab_size=32000 \
	model.decoder_tokenizer.vocab_size=32000 \
	model.decoder.hidden_size=1536 \
	model.decoder.inner_size=6144 \
	model.decoder.num_attention_heads=24 \
	model.decoder.num_layers=6 \
	model.decoder.attn_layer_dropout=0.1 \
	model.decoder.ffn_dropout=0.1 \
	model.train_ds.use_tarred_dataset=true \
	model.train_ds.shard_strategy=scatter \
	model.train_ds.src_file_name=/raid/wmt21/train.dedup.de \
	model.train_ds.tgt_file_name=/raid/wmt21/train.dedup.en \
	model.train_ds.tokens_in_batch=${TOKENS_IN_BATCH} \
	model.validation_ds.src_file_name=[/raid/wmt21/val/newstest2020-en-de.clean.tok.ref,/raid/wmt21/val/newstest2019-en-de.clean.tok.ref,/raid/wmt21/val/newstest2018-en-de.clean.tok.ref,/raid/wmt21/val/newstest2014-en-de.clean.tok.ref,/raid/wmt21/val/newstest2013-en-de.clean.tok.ref] \
	model.validation_ds.tgt_file_name=[/raid/wmt21/val/newstest2020-en-de.clean.tok.src,/raid/wmt21/val/newstest2019-en-de.clean.tok.src,/raid/wmt21/val/newstest2018-en-de.clean.tok.src,/raid/wmt21/val/newstest2014-en-de.clean.tok.src,/raid/wmt21/val/newstest2013-en-de.clean.tok.src] \
	~model.test_ds \
	model.optim.lr=$LEARNING_RATE \
	+model.optim.sched.warmup_steps=$WARMUP_STEPS \
  	~model.optim.sched.warmup_ratio \
	+exp_manager.explicit_log_dir=/raid/results/wmt21_de_en_yttm_tokens_${TOKENS_IN_BATCH} \
	+exp_manager.resume_if_exists=True \
	+exp_manager.resume_ignore_no_checkpoint=True \
	+exp_manager.create_checkpoint_callback=True \
	+exp_manager.checkpoint_callback_params.monitor=val_sacreBLEU \
	+exp_manager.checkpoint_callback_params.save_top_k=1 \
	+exp_manager.checkpoint_callback_params.mode=max
