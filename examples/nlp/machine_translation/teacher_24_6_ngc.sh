#!/bin/bash
INSTANCE=dgx1v.32g.8.norm
PROJECT=nmt-en-de-ngc
DATAID=68792
STEPS=200000
WANDBLOGIN=1589819cfa34108320cd27634a3f764a29b211d8
SLURM_GPUS_PER_NODE=8

for BATCH_SIZE in 12500
do
  for lr in 1e-3
  do
    for ws in 7500
    do
      for VOCAB_SIZE in 32000
      do
          EXPNAME=NMT_TEACHER_DE_EN_NGC
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
                python /code/examples/nlp/machine_translation/enc_dec_nmt.py \
                --config-path=conf \
                --config-name=aayn_base \
                do_training=true \
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
                model.preproc_out_dir=/preproc_data \
                model.encoder.hidden_size=1024 \
                model.encoder.inner_size=4096 \
                model.encoder.num_attention_heads=16 \
                model.encoder.num_layers=24 \
                model.encoder.ffn_dropout=0.1 \
                model.encoder.pre_ln=true \
                model.encoder_tokenizer.vocab_size=32000 \
                model.decoder_tokenizer.vocab_size=32000 \
                model.decoder.pre_ln=true \
                model.decoder.num_layers=6 \
                model.decoder.hidden_size=1024 \
                model.decoder.inner_size=4096 \
                model.decoder.num_attention_heads=16 \
                model.decoder.ffn_dropout=0.1 \
                model.train_ds.use_tarred_dataset=true \
                model.train_ds.shard_strategy=scatter \
                model.train_ds.src_file_name=/raid/train.clean.de.shuffled \
                model.train_ds.tgt_file_name=/raid/train.clean.en.shuffled \
                model.train_ds.tokens_in_batch=${BATCH_SIZE} \
                model.validation_ds.src_file_name=[/raid/wmt13-en-de.ref,/raid/wmt14-en-de.ref] \
                model.validation_ds.tgt_file_name=[/raid/wmt13-en-de.src,/raid/wmt14-en-de.src] \
                ~model.test_ds \
                model.optim.lr=$LEARNING_RATE \
                +model.optim.sched.warmup_steps=$WARMUP_STEPS \
                ~model.optim.sched.warmup_ratio \
                +exp_manager.create_wandb_logger=True \
                +exp_manager.wandb_logger_kwargs.name=${EXPNAME} \
                +exp_manager.wandb_logger_kwargs.project=${PROJECT} \
                +exp_manager.explicit_log_dir=/results \
                +exp_manager.resume_if_exists=True \
                +exp_manager.resume_ignore_no_checkpoint=True \
                +exp_manager.create_checkpoint_callback=True \
                +exp_manager.checkpoint_callback_params.monitor=val_sacreBLEU \
                +exp_manager.checkpoint_callback_params.save_top_k=2 \
                +exp_manager.checkpoint_callback_params.mode=max" \
                --result /results/ \
                --org nvidian \
                --team ac-aiapps \
                --datasetid $DATAID:/data/
      done
    done
  done
done