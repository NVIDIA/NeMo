#!/bin/bash

SPKRS=`cat /data/speech/combined_datasets/local/train_speakers.txt`
# SPKRS="[0,1]"

### ~model.optim.sched \
wandb login 439d5b4a58e063dfc8b5aba424f545e537f9ec10 \
&& HYDRA_FULL_ERROR=1 python generate_inference_audio.py \
    --config-name=fastpitch_speaker_adaptation.yaml \
    ~model.speaker_encoder.sv_projection_module \
    +model.adapter.add_weight_speaker_list="${SPKRS}" \
    