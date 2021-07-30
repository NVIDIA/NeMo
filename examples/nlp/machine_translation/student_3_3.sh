#!/bin/bash
PROJECT=nmt-de-en
STEPS=100000
WANDBLOGIN=1589819cfa34108320cd27634a3f764a29b211d8
DISTILLATION_LOSS_WEIGHT=0.5
STUDENT_TRAIN_LOSS_WEIGHT=$(bc <<< "1.0 - $DISTILLATION_LOSS_WEIGHT")
TEMPERATURE=2.0
EXPNAME=STUDENT_3_3_NMT_DE_EN_DL_${DISTILLATION_LOSS_WEIGHT}_SL_${STUDENT_TRAIN_LOSS_WEIGHT}_TEMP_${TEMPERATURE}

# model.encoder_tokenizer.tokenizer_model=/raid/wmt_16/tokenizer.BPE.8192.model \
# model.decoder_tokenizer.tokenizer_model=/raid/wmt_16/tokenizer.BPE.8192.model \
# encoder: abb8f67c1d0f47bc8ce88cac13dfc868_tokenizer.BPE.8192.model 
# decoder: 600b4abaf3194178bc521cbc7f2c5be9_tokenizer.BPE.8192.model

python enc_dec_nmt_distill.py \
--config-path=conf \
--config-name=aayn_base_distill_local \
trainer.gpus=2 \
~trainer.max_epochs \
+trainer.max_steps=100000 \
+trainer.val_check_interval=1000 \
model.beam_size=4 \
model.max_generation_delta=5 \
model.label_smoothing=0.1 \
model.encoder.num_layers=3 \
model.encoder.hidden_size=512 \
model.encoder.inner_size=2048 \
model.encoder.num_attention_heads=8 \
model.encoder.ffn_dropout=0.1 \
model.decoder.pre_ln=True \
model.decoder.num_layers=3 \
model.decoder.hidden_size=512 \
model.decoder.inner_size=2048 \
model.decoder.num_attention_heads=8 \
model.decoder.ffn_dropout=0.1 \
model.encoder_tokenizer.vocab_size=8192 \
model.decoder_tokenizer.vocab_size=8192 \
model.train_ds.src_file_name=/raid/wmt_16/train.clean.de.shuffled \
model.train_ds.tgt_file_name=/raid/wmt_16/train.clean.en.shuffled \
model.train_ds.tokens_in_batch=12500 \
model.validation_ds.src_file_name=/raid/wmt_16/wmt14-en-de.ref \
model.validation_ds.tgt_file_name=/raid/wmt_16/wmt14-en-de.src \
model.validation_ds.tokens_in_batch=8192 \
model.test_ds.src_file_name=/raid/wmt_16/wmt14-en-de.ref \
model.test_ds.tgt_file_name=/raid/wmt_16/wmt14-en-de.src \
model.optim.lr=1e-3 \
~model.optim.sched.warmup_ratio \
+model.optim.sched.warmup_steps=7500 \
model.distillation.distillation_loss_weight=${DISTILLATION_LOSS_WEIGHT} \
model.distillation.student_train_loss_weight=${STUDENT_TRAIN_LOSS_WEIGHT} \
model.distillation.temperature=${TEMPERATURE} \
+exp_manager.create_wandb_logger=True \
+exp_manager.wandb_logger_kwargs.name=${EXPNAME} \
+exp_manager.wandb_logger_kwargs.project=nmt-de-en \
+exp_manager.create_checkpoint_callback=True \
+exp_manager.checkpoint_callback_params.monitor=val_sacreBLEU \
+exp_manager.exp_dir=/raid/exp_results_student \
+exp_manager.checkpoint_callback_params.mode=max \
+exp_manager.checkpoint_callback_params.always_save_nemo=True