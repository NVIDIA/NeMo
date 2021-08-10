#!/bin/bash
PROJECT=nmt-de-en
WANDBLOGIN=1589819cfa34108320cd27634a3f764a29b211d8
NUM_GPUS=1

# Hyperparams
TOKENS_IN_BATCH=8000 \
LEARNING_RATE=4e-4
STEPS=100000
WARMUP_STEPS=30000

# Distillation
DISTILLATION_LOSS_WEIGHT=10.0
STUDENT_TRAIN_LOSS_WEIGHT=1.0
COSINE_LOSS_WEIGHT=1.0
TEMPERATURE=2.0

EXPNAME=NMT_STUDENT_3_3_NMT_DE_EN_DL_${DISTILLATION_LOSS_WEIGHT}_SL_${STUDENT_TRAIN_LOSS_WEIGHT}_CL_${COSINE_LOSS_WEIGHT}_TEMP_${TEMPERATURE}

python enc_dec_nmt_distill.py \
--config-path=conf \
--config-name=aayn_base_distill \
do_training=true \
trainer.gpus=${NUM_GPUS} \
~trainer.max_epochs \
+trainer.max_steps=${STEPS} \
+trainer.val_check_interval=1000 \
+trainer.accumulate_grad_batches=1 \
model.src_language=de \
model.tgt_language=en \
model.beam_size=4 \
model.max_generation_delta=6 \
model.label_smoothing=0.1 \
model.preproc_out_dir=/raid/wmt21_en_de_yttm_tokens_${TOKENS_IN_BATCH} \
model.encoder.hidden_size=1024 \
model.encoder.inner_size=4096 \
model.encoder.num_attention_heads=16 \
model.encoder.num_layers=3 \
model.encoder.ffn_dropout=0.1 \
model.encoder.pre_ln=true \
model.encoder_tokenizer.vocab_size=32000 \
model.decoder_tokenizer.vocab_size=32000 \
model.decoder.pre_ln=true \
model.decoder.num_layers=3 \
model.decoder.hidden_size=1024 \
model.decoder.inner_size=4096 \
model.decoder.num_attention_heads=16 \
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
model.distillation.model_path='/raid/nemo_models/teacher_12_6_de_en/AAYNBase.nemo' \
model.distillation.distillation_loss_weight=${DISTILLATION_LOSS_WEIGHT} \
model.distillation.student_train_loss_weight=${STUDENT_TRAIN_LOSS_WEIGHT} \
model.distillation.cosine_loss_weight=${COSINE_LOSS_WEIGHT} \
model.distillation.temperature=${TEMPERATURE} \
model.distillation.distill_encoder=True \
+exp_manager.create_wandb_logger=True \
+exp_manager.wandb_logger_kwargs.name=${EXPNAME} \
+exp_manager.wandb_logger_kwargs.project=${PROJECT} \
+exp_manager.explicit_log_dir=/raid/results \
+exp_manager.resume_if_exists=False \
+exp_manager.resume_ignore_no_checkpoint=True \
+exp_manager.create_checkpoint_callback=True \
+exp_manager.checkpoint_callback_params.monitor=val_sacreBLEU \
+exp_manager.checkpoint_callback_params.save_top_k=1 \
+exp_manager.checkpoint_callback_params.mode=max \
+exp_manager.checkpoint_callback_params.always_save_nemo=True
