#!/usr/bin/env bash

### bash scripts/speech_recognition/k2/setup.sh


train_script="examples/tts/tts_transducer.py"
config_name="conformer_transducer_bpe_as_tts.yaml"
config_path="examples/tts/conf/"
# tokenizer_dir=code_dir / "tokenizer_models/libritts_r_v1/tokenizer_spe_unigram_v1024"
TRAIN_AUDIO="/data/LibriTTS-R/dev_all_tar/audio__OP_0..3_CL_.tar"
TRAIN_MANIFEST="/data/LibriTTS-R/manifests/dev.json"
VAL_AUDIO="/data/LibriTTS-R/dev_all_tar/audio__OP_0..3_CL_.tar"
VAL_MANIFEST="/data/LibriTTS-R/manifests/dev.json"
num_workers=8
batch_size=32
num_epochs=10
grad_accumulation=1
precision=16
sup_data_types="['speaker_id','semantic_code']" # 'align_prior_matrix','speaker_id'
sup_data_path="/data/LibriTTS-R/dev_ssup"

# model.tokenizer.dir=${tokenizer_dir} \
# model.tokenizer.type="bpe" \
# --config-path=${config_path} \

HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python ${train_script} \
--config-name=${config_name} \
sup_data_types=${sup_data_types} \
sup_data_path=${sup_data_path} \
train_dataset=${TRAIN_MANIFEST} \
train_audio=${TRAIN_AUDIO} \
validation_datasets=${VAL_MANIFEST} \
validation_audio=${VAL_AUDIO} \
model.encoder.conv_norm_type="layer_norm" \
model.train_ds.dataloader_params.num_workers=${num_workers} \
model.validation_ds.dataloader_params.num_workers=${num_workers} \
model.train_ds.dataloader_params.batch_size=${batch_size} \
model.validation_ds.dataloader_params.batch_size=${batch_size} \
model.train_ds.dataset.max_duration=20 \
model.validation_ds.dataset.max_duration=20 \
model.validation_ds.dataset.min_duration=0.2 \
trainer.max_epochs=${num_epochs} \
trainer.accumulate_grad_batches=${grad_accumulation} \
trainer.log_every_n_steps=100 \
++trainer.reload_dataloaders_every_n_epochs=0 \
++trainer.precision=${precision} > debug.log