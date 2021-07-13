#!/bin/bash
BATCH_SIZE=12500
lr=1e-3
ws=7500
VOCAB_SIZE=32000
STEPS=100000
EXPNAME=STUDENT_DISTILLED_LOCAL

# yttm bpe --data /raid/wmt_16/train.clean.en-de.shuffled.common --model /raid/wmt_16/tokenizer.BPE.${VOCAB_SIZE}.model --vocab_size $VOCAB_SIZE && \
# yttm bpe --data /raid/wmt_16/train.clean.en-de.shuffled.common --model /raid/wmt_16/tokenizer.BPE.8192.model --vocab_size 8192 --coverage 0.999

python enc_dec_nmt_distill.py \
--config-path=conf \
--config-name=aayn_base_distill \
trainer.gpus=2 \
+trainer.val_check_interval=1000 \
+trainer.max_steps=${STEPS} \
model.beam_size=4 \
model.max_generation_delta=5 \
model.label_smoothing=0.1 \
do_training=False \
do_testing=True \
model.encoder_tokenizer.tokenizer_model=/raid/wmt_16/tokenizer.BPE.${VOCAB_SIZE}.model \
model.decoder_tokenizer.tokenizer_model=/raid/wmt_16/tokenizer.BPE.${VOCAB_SIZE}.model \
model.encoder.num_layers=6 \
model.encoder.hidden_size=1024 \
model.encoder.inner_size=4096 \
model.encoder.num_attention_heads=16 \
model.encoder.ffn_dropout=0.3 \
model.encoder.pre_ln=True \
model.encoder.attn_score_dropout=0.1 \
model.encoder.attn_layer_dropout=0.3 \
model.encoder.hidden_act=relu \
model.encoder.mask_future=False \
model.decoder.pre_ln=True \
model.decoder.num_layers=2 \
model.decoder.hidden_size=1024 \
model.decoder.inner_size=4096 \
model.decoder.num_attention_heads=16 \
model.decoder.ffn_dropout=0.3 \
model.decoder.attn_score_dropout=0.1 \
model.decoder.attn_layer_dropout=0.3 \
model.decoder.hidden_act=relu \
model.train_ds.src_file_name=/raid/wmt_16/train.clean.de.shuffled \
model.train_ds.tgt_file_name=/raid/wmt_16/train.clean.en.shuffled \
model.train_ds.tokens_in_batch=${BATCH_SIZE} \
model.validation_ds.src_file_name=[/raid/wmt_16/wmt13-en-de.ref,/raid/wmt_16/wmt14-en-de.ref] \
model.validation_ds.tgt_file_name=[/raid/wmt_16/wmt13-en-de.src,/raid/wmt_16/wmt14-en-de.src] \
model.validation_ds.tokens_in_batch=8192 \
model.test_ds.src_file_name=/raid/wmt_16/wmt14-en-de.ref \
model.test_ds.tgt_file_name=/raid/wmt_16/wmt14-en-de.src \
model.optim.lr=$lr  \
~model.optim.sched.warmup_ratio \
+model.optim.sched.warmup_steps=$ws \
+exp_manager.create_wandb_logger=True \
+exp_manager.wandb_logger_kwargs.name=${EXPNAME} \
+exp_manager.wandb_logger_kwargs.project=nmt-de-en_distill \
+exp_manager.create_checkpoint_callback=True \
+exp_manager.checkpoint_callback_params.monitor=val_sacreBLEU \
+exp_manager.exp_dir=/raid/exp_results_distill \
+exp_manager.checkpoint_callback_params.mode=max \
+exp_manager.checkpoint_callback_params.always_save_nemo=True