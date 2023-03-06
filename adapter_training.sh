#!/bin/bash

SPEAKERS="5789 5872 10220 11084 11484 15664 16741 16955 16999"
SPEAKERS="SALLY"
wandb login 439d5b4a58e063dfc8b5aba424f545e537f9ec10

for SPK in $SPEAKERS; do
    
    echo "Starting ${SPK}"
#     TRAIN_DS="/data/speech/HiFiTTS/hifi_tts_short_v0/${SPK}_train_process.json"
    TRAIN_DS="/data/speech/RIVA/sally_train_30m_local.json"
    TRAIN_SPK_EMB="/data/speech/RIVA/sally_train_30m_local_titanet.npy"
#     EVAL_DS="/data/speech/HiFiTTS/hifi_tts_short_v0/${SPK}_dev_process.json"
    EVAL_DS="/data/speech/RIVA/sally_audio_normalized_randomized_sorted_manifest_test_local.json"
    EVAL_SPK_EMB="/data/speech/RIVA/sally_audio_normalized_randomized_sorted_manifest_test_local_titanet.npy"
    
    fastpitch_sup_data_path="fastpitch_sup_data_folder"
    EXP_DIR="${SPK}_adapter_experiment_test"
    EXP_DIR="SALLY_adapter_experiment_test_RIVA"
    WEIGHTED_SPEAKER=true
    CONDITION_SPEAKER=true
    OPTIM="adam"
    project_name="finetune-RIVA-GSTTitanet-${OPTIM}"
    BS=24
    LR=2e-4

    STATS=`cat /data/speech/HiFiTTS/hifi_tts_short_v0/${SPK}_pitch_stats.txt`
    STATSARRAY=($STATS)

    pitch_mean=173.4511260986328
    ### ${STATSARRAY[0]}
    pitch_std=27.54319190979004
    ### ${STATSARRAY[1]}
    pitch_min=65.4063949584961
    ### ${STATSARRAY[2]}
    pitch_max=1147.9888305664062
    ### ${STATSARRAY[3]}
    SPKRS=`cat /data/speech/combined_datasets/local/train_speakers.txt`
#     SPKRS="[0,1]"
    echo "Training ${SPK}"
    ### ~model.optim.sched \
    ### ~model.speaker_encoder.sv_projection_module \


    HYDRA_FULL_ERROR=1 python examples/tts/fastpitch_finetune_adapters.py \
        --config-name=fastpitch_speaker_adaptation.yaml \
        sample_rate=44100 \
        train_dataset=${TRAIN_DS} \
        validation_datasets=${EVAL_DS} \
        sup_data_types="['align_prior_matrix','pitch','speaker_id','gst_ref_audio','speaker_embedding']" \
        sup_data_path=${fastpitch_sup_data_path} \
        +init_from_ptl_ckpt="FastPitch-GSTTitanetFinetune-HFLiTTS--val_loss\=1.0590-epoch\=269-last.ckpt" \
        pitch_mean=${pitch_mean} \
        pitch_std=${pitch_std} \
        pitch_fmin=${pitch_min} \
        pitch_fmax=${pitch_max} \
        model.n_speakers=12800 \
        phoneme_dict_path=scripts/tts_dataset_files/cmudict-0.7b_nv22.08 \
        heteronyms_path=scripts/tts_dataset_files/heteronyms-052722 \
        whitelist_path=nemo_text_processing/text_normalization/en/data/whitelist/lj_speech.tsv \
        model.adapter.add_weight_speaker=${WEIGHTED_SPEAKER} \
        model.speaker_emb_condition_prosody="${CONDITION_SPEAKER}" \
        model.speaker_emb_condition_decoder="${CONDITION_SPEAKER}" \
        model.speaker_emb_condition_aligner="${CONDITION_SPEAKER}" \
        +model.adapter.add_weight_speaker_list="${SPKRS}" \
        +model.train_ds.dataset.speaker_embedding_path=${TRAIN_SPK_EMB} \
        +model.validation_ds.dataset.speaker_embedding_path=${EVAL_SPK_EMB} \
        model.train_ds.dataloader_params.batch_size=${BS} \
        model.validation_ds.dataloader_params.batch_size=${BS} \
        model.train_ds.dataloader_params.num_workers=8 \
        model.validation_ds.dataloader_params.num_workers=8 \
        model.optim.lr=${LR} \
        ~model.optim.sched \
        model.optim.name=${OPTIM} \
        model.optim.weight_decay=0.0 \
        +model.text_tokenizer.add_blank_at=True \
        trainer.check_val_every_n_epoch=30 \
        trainer.max_epochs=500 \
        trainer.log_every_n_steps=5 \
        trainer.devices=1 \
        trainer.precision=32 \
        exp_manager.exp_dir=${EXP_DIR} \
        +exp_manager.create_wandb_logger=True \
        +exp_manager.wandb_logger_kwargs.name=${project_name} \
        +exp_manager.wandb_logger_kwargs.project="Adapters_GST"

done
