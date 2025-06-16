!/usr/bin/env bash

DOCKER_EXP_DIR="/checkpoints/results"

### Good
# CKPT="/checkpoints/streaming/magpie/jason/magpieTTS--val_loss=5.1255-epoch=89-last.ckpt"
# HPARAM="/checkpoints/streaming/magpie/jason/magpietts_en_jason_inference.yaml"

# CKPT="/TB/magpie2503_DC_CE_CA_BIN_F5P5E0_DECF1.0P1.0_C21FPS_Causal_8cb_HRLLM_lr1e-4_bs12_precbf16_wait1_strictwindowTrue_EncPrior/magpieTTS/checkpoints/magpieTTS--val_loss=5.1938-epoch=247-last.ckpt"
# HPARAM="/workspace/NeMo/magpietts_en_w1_sw_encprior_inf.yaml"

# Great for nonstreaming Best for streaming with exponential weight
# CKPT="/checkpoints/streaming/magpie/sugh_BIN_F2P1E0.0/magpieTTS--val_loss=5.1851-epoch=143-last.ckpt"
# HPARAM="/checkpoints/streaming/magpie/sugh_BIN_F2P1E0.0/magpietts_en_subhankarg_BIN_F2P1E0.0.yaml"

CKPT="/checkpoints/streaming/magpie/jensen/magpieTTS.nemo"
HPARAM="/checkpoints/streaming/magpie/jensen/config_jensen.yaml"

### NO GOOD
# CKPT="/checkpoints/streaming/magpie/subhankarg_nobin/magpieTTS--val_loss=7.6005-epoch=337-last.ckpt"
# HPARAM="/checkpoints/streaming/magpie/subhankarg_nobin/magpietts_en_subhankarg_nobin.yaml"

### NO GOOD
# CKPT="/checkpoints/streaming/magpie/subhankarg_bin/magpieTTS--val_loss=7.5660-epoch=299-last.ckpt"
# HPARAM="/checkpoints/streaming/magpie/subhankarg_bin/magpietts_en_subhankarg_bin.yaml"

# CODEC="/nemo_codec_checkpoints/AudioCodec_21Hz_no_eliz.nemo"
CODEC="/nemo_codec_checkpoints/21fps_causal_codecmodel.nemo"
# CODEC="/nemo_codec_checkpoints/Low_Frame-rate_Speech_Codec++.nemo"

# EPS=0.1
EPS=0.1
DATASET=local_longer_1 # riva_challenging_nozeros # local_test_100 # local_test_20 # local_test # local_long_20 # local_longer_20 # local_longer_10
MODELTYPE=streaming
# --model_type ${MODELTYPE} \

export CUDA_VISIBLE_DEVICES=0

# --checkpoint_files $CKPT \
# --hparams_files ${HPARAM} \

python scripts/magpietts/infer_and_evaluate.py \
--codecmodel_path ${CODEC} \
--nemo_file $CKPT \
--datasets $DATASET \
--out_dir /checkpoints/results/dev_${MODELTYPE}_sw_${EPS}ep_${DATASET}_newcode_cache_jensen \
--batch_size 8 \
--use_cfg \
--attention_prior_epsilon ${EPS} \
--attention_prior_lookahead_window 5 \
--apply_attention_prior \
--estimate_alignment_from_layers="4,5,6,7" \
--apply_prior_to_layers="0,1,2,3,4,5,6,7,8,9,10,11" \
--start_prior_after_n_audio_steps 0 \
--cfg_scale 2.5 \
--temperature 0.7 \
--streaming ;

# --legacy_codebooks \