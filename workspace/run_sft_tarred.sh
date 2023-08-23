NEMO_DIR=/home/heh/codes/nemo-zhehuai
export PYTHONPATH=$NEMO_DIR:$PYTHONPATH

MEGATRON_CKPT=/media/data3/pretrained_models/megatron_gpt/gpt_pretrain_220m_len_4096_pos_alibi_step_595508_gbs256.nemo
ASR_MODEL="stt_en_fastconformer_transducer_large"

PROJECT_NAME=audio-text-llm-debug
EXP_NAME=AudioGPT-tarred-full-LS-debug-frz-audio-tmp8

# 249 batch, 162
GLOBAL_BATCH=64
MICRO_BATCH=32

NUM_WORKERS=0
# TRAIN_MANIFESTS="/media/data3/librispeech_tarred/tarred_audio_manifest.json"
# TRAIN_FILEPATHS="/media/data3/librispeech_tarred/audio__OP_0..511_CL_.tar"


TRAIN_MANIFESTS="/media/data3/librispeech_test_tarred/tarred_audio_manifest.json"
TRAIN_FILEPATHS="/media/data3/librispeech_test_tarred/audio__OP_0..7_CL_.tar"
VAL_MANIFESTS="[/media/data/datasets/LibriSpeech/dev_clean.json,/media/data/datasets/LibriSpeech/dev_other.json]"
VAL_NAMES="[dev-clean,dev-other]"

# VAL_MANIFESTS="[/media/data/datasets/LibriSpeech/dev_small.json]"


python run_sft_audio_lm.py --config-path="../examples/multimodel/conf/speechllm/" --config-name "modularized_speech_gpt_config" \
    name=$EXP_NAME \
    ++exp_manager.create_wandb_logger=false \
    ++exp_manager.name=$EXP_NAME \
    ++exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    ++exp_manager.wandb_logger_kwargs.project=${PROJECT_NAME} \
    ++exp_manager.wandb_logger_kwargs.resume=false \
    trainer.devices=-1 \
    model.freeze_audio_encoder=True \
    model.global_batch_size=$GLOBAL_BATCH \
    model.micro_batch_size=$MICRO_BATCH \
    model.pretrained_audio_model=$ASR_MODEL \
    model.restore_from_path=$MEGATRON_CKPT \
    model.data.train_ds.is_tarred=True \
    model.data.train_ds.manifest_filepath=$TRAIN_MANIFESTS \
    model.data.train_ds.tarred_audio_filepaths=${TRAIN_FILEPATHS} \
    model.data.train_ds.num_workers=$NUM_WORKERS \
    model.data.validation_ds.manifest_filepath=$VAL_MANIFESTS \
    model.data.validation_ds.num_workers=$NUM_WORKERS

