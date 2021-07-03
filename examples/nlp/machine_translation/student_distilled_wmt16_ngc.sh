#!/bin/bash

INSTANCE=dgx1v.32g.8.norm
PROJECT=nmt-de-en-ngc
DATAID=68792
STEPS=100000
WANDBLOGIN=1589819cfa34108320cd27634a3f764a29b211d8

for BATCH_SIZE in 12500
do
  for lr in 1e-3
  do
    for ws in 7500
    do
      for VOCAB_SIZE in 32000
      do
          EXPNAME=STUDENT_NMT_DE_EN_DISTILL_NGC
          ngc batch run --name ${EXPNAME} --preempt RUNONCE \
                --image "nvcr.io/nvidia/pytorch:21.05-py3" \
                --ace nv-us-west-2 \
                --instance $INSTANCE \
                --commandline "export DEBIAN_FRONTEND=noninteractive && nvidia-smi && apt-get update && apt-get install -y libsndfile1 ffmpeg && \
                pip install wandb==0.10.21 && pip install Cython && wandb login ${WANDBLOGIN} && \
                git clone https://github.com/NVIDIA/NeMo.git && cd NeMo && \
                git checkout main && ./reinstall.sh && \
                cp -R /data/* /raid/ && \
                yttm bpe --data /raid/train.clean.en-de.shuffled.common --model /results/tokenizer.BPE.${VOCAB_SIZE}.model --vocab_size $VOCAB_SIZE && \
                python enc_dec_nmt_distill.py \
                --config-path=conf \
                --config-name=aayn_base_distill \
                trainer.gpus=8 \
                +trainer.val_check_interval=100 \
                +trainer.max_steps=${STEPS} \
                model.beam_size=4 \
                model.max_generation_delta=5 \
                model.label_smoothing=0.1 \
                model.encoder_tokenizer.tokenizer_model=/results/tokenizer.BPE.${VOCAB_SIZE}.model  \
                model.decoder_tokenizer.tokenizer_model=/results/tokenizer.BPE.${VOCAB_SIZE}.model  \
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
                model.train_ds.src_file_name=/raid/train.clean.de.shuffled \
                model.train_ds.tgt_file_name=/raid/train.clean.en.shuffled \
                model.train_ds.tokens_in_batch=${BATCH_SIZE} \
                model.validation_ds.src_file_name=[/raid/wmt13-en-de.ref,/raid/wmt14-en-de.ref] \
                model.validation_ds.tgt_file_name=[/raid/wmt13-en-de.src,/raid/wmt14-en-de.src] \
                model.validation_ds.tokens_in_batch=8192 \
                model.test_ds.src_file_name=/raid/wmt14-en-de.ref \
                model.test_ds.tgt_file_name=/raid/wmt14-en-de.src \
                model.optim.lr=$lr  \
                ~model.optim.sched.warmup_ratio \
                +model.optim.sched.warmup_steps=$ws \
                +exp_manager.create_wandb_logger=True \
                +exp_manager.wandb_logger_kwargs.name=${EXPNAME} \
                +exp_manager.wandb_logger_kwargs.project=${PROJECT} \
                +exp_manager.create_checkpoint_callback=True \
                +exp_manager.checkpoint_callback_params.monitor=val_sacreBLEU \
                +exp_manager.exp_dir=/results \
                +exp_manager.checkpoint_callback_params.mode=max \
                +exp_manager.checkpoint_callback_params.always_save_nemo=True" \
                --result /results/ \
                --org nvidian \
                --team ac-aiapps \
                --datasetid $DATAID:/data/
      done
    done
  done
done