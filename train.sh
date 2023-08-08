!/usr/bin/env bash

rm -rf nemo_experiments

#### pip install --no-deps encodec
### vi /usr/local/lib/python3.8/dist-packages/encodec/utils.py


###### 220M model ########
CUDA_LAUNCH_BLOCKING=1 python examples/nlp/language_modeling/megatron_t5_speechlm.py \
--config-name=megatron_t5_speechlm_220m_sft.yaml \
model.data.train_task="tts" \
+model.freeze_model=False \
model.data.max_seq_length=2048 \
model.data.sup_data_path=/data/speech/LibriTTS2/FP_Codec_sup_24khz \
model.language_model_path=/data/megatron_t5_220m/tp1_pp1/megatron_t5_expanded_vocab_posemb.nemo > debug.log


###### 3BM model ########
# CUDA_LAUNCH_BLOCKING=1 python examples/nlp/language_modeling/megatron_t5_speechlm.py \
# --config-name=megatron_t5_speechlm_3b_sft.yaml \
# model.data.train_task="tts" \
# +model.freeze_model=False \
# model.data.max_seq_length=2048 \
# model.data.sup_data_path=/data/speech/LibriTTS2/FP_Codec_sup_24khz \
# model.language_model_path=/data/megatron_3b_xTP/megatron_t5_expanded_vocab_2_expanded_vocab_2.nemo > debug.log
