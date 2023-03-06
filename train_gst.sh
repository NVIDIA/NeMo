#!/bin/bash

SPK=11084
TRAIN_DS="/data/speech/combined_datasets/local/train_libritts360-hifitts_titanet_subset.json"
EVAL_DS="/data/speech/combined_datasets/local/train_libritts360-hifitts_titanet_subset.json"
fastpitch_sup_data_path="fastpitch_sup_data_folder"
OPTIM="adamw"
project_name="finetune-${SPK}-${OPTIM}-FrzSpkProj"
BS=24
LR=1e-4

## speaker 11084
pitch_mean=236.22108459472656
pitch_std=175.60159301757812
pitch_min=65.4063949584961
pitch_max=2093.004638671875

SPKRS=`cat /data/speech/combined_datasets/local/train_speakers.txt`


 # model.n_speakers=12800 \
 # +init_from_pretrained_model="tts_en_fastpitch_multispeaker" \
 # sup_data_types="['align_prior_matrix','pitch','speaker_id','gst_ref_audio', 'speaker_embedding']" \
 # --config-name=fastpitch_speaker_adaptation.yaml \
### ~model.optim.sched \
wandb login 439d5b4a58e063dfc8b5aba424f545e537f9ec10 \
&& HYDRA_FULL_ERROR=1 python examples/tts/fastpitch_finetune.py \
    --config-name=fastpitch_speaker_adaptation \
    sample_rate=44100 \
    train_dataset=${TRAIN_DS} \
    validation_datasets=${EVAL_DS} \
    sup_data_types="['align_prior_matrix','pitch','speaker_id','gst_ref_audio','speaker_embedding']" \
    sup_data_path=${fastpitch_sup_data_path} \
    +init_from_pretrained_model="tts_en_fastpitch_multispeaker" \
    pitch_mean=${pitch_mean} \
    pitch_std=${pitch_std} \
    pitch_fmin=${pitch_min} \
    pitch_fmax=${pitch_max} \
    model.n_speakers=12800 \
    model.train_ds.dataloader_params.batch_size=${BS} \
    model.validation_ds.dataloader_params.batch_size=${BS} \
    model.train_ds.dataloader_params.num_workers=8 \
    model.validation_ds.dataloader_params.num_workers=8 \
    model.speaker_emb_condition_prosody=True \
    model.speaker_emb_condition_decoder=True \
    model.speaker_emb_condition_aligner=True \
    +model.speaker_encoder.add_weight_speaker_list="${SPKRS}" \
    +model.speaker_encoder.add_weight_speaker=True \
    +model.text_tokenizer.add_blank_at=True \
    +model.train_ds.dataset.speaker_embedding_path=/data/speech/combined_datasets/speaker_embeddings_44KHz/train_libritts360-hifitts_titanet.npy \
    +model.validation_ds.dataset.speaker_embedding_path=/data/speech/combined_datasets/speaker_embeddings_44KHz/train_libritts360-hifitts_titanet.npy \
    model.optim.lr=${LR} \
    model.optim.name=${OPTIM} \
    model.optim.weight_decay=0.0 \
    trainer.check_val_every_n_epoch=1 \
    trainer.max_epochs=200 \
    trainer.log_every_n_steps=1 \
    trainer.devices=1 \
    trainer.precision=32 \
    exp_manager.exp_dir="nemo_experiments" \
    +exp_manager.create_wandb_logger=False \
    +exp_manager.wandb_logger_kwargs.name=${project_name} \
    +exp_manager.wandb_logger_kwargs.project="Adapters_TitaNetEmb"
    
#     +model.text_tokenizer.add_blank_at=True \
    
