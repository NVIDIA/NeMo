!/usr/bin/env bash

rm -rf nemo_experiments

#### pip install --no-deps encodec

# python examples/nlp/language_modeling/megatron_t5_prompt_learning.py \
# model.language_model_path=/data/megatron_t5_220m/tp1_pp1/megatron_t5.nemo > debug.log

python examples/nlp/language_modeling/megatron_t5_speechlm.py \
--config-name=megatron_t5_speechlm.yaml \
model.data.sup_data_path=/data/speech/LibriTTS2/FP_Codec_sup_24khz \
model.language_model_path=/data/megatron_t5_220m/tp1_pp1/megatron_t5.nemo > debug.log

# model.language_model_path=/data/megatron_t5_3b/tp2_pp1/megatron_t5.nemo > debug.log


# python examples/nlp/language_modeling/megatron_t5_lm_adaptation_finetune.py \
# model.pretrained_model_path=/data/megatron_t5_220m/tp1_pp1/megatron_t5.nemo > debug_finetune.log