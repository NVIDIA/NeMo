!/usr/bin/env bash

rm -rf nemo_experiments

#### pip install --no-deps encodec
### vi /usr/local/lib/python3.8/dist-packages/encodec/utils.py

LR=5e-5
MAX_STEPS=300000

WANDB_PROJECT="SpeechLM_T5"
WANDB="439d5b4a58e063dfc8b5aba424f545e537f9ec10"
WANDB_NAME="220MSpeechEmb_ASR_2K_FP32_lr${LR}_nosched_posemb1536"

###### 220M model ########
export WANDB_API_KEY=${WANDB}

CUDA_LAUNCH_BLOCKING=1 python examples/nlp/language_modeling/megatron_t5_speechlm.py \
--config-name=megatron_t5_speechlm_220m_sft.yaml \
model.data.train_task="all" \
model.optim.lr=${LR} \
model.data.max_seq_length=1536 \
model.data.sup_data_path=/data/speech/speechlm_codec_sup_24khz \
model.language_model_path=/data/megatron_t5_220m/tp1_pp1/megatron_t5_expanded_vocab_posemb1536.nemo \
model.data.train_ds=["train_clean_300_speechlm_alltasks.json"] \
model.data.validation_ds=["val_clean_300_speechlm_alltasks.json"] \
+model.freeze_model=False \
exp_manager.create_early_stopping_callback=False \
~exp_manager.early_stopping_callback_params \
trainer.val_check_interval=500 \
trainer.max_steps=${MAX_STEPS} > debug.log


# exp_manager.create_wandb_logger=true \
# exp_manager.wandb_logger_kwargs.name=${WANDB_NAME} \
# exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT} \

# +init_from_ptl_ckpt=/data/SpeechLM/pretrained/SeqLen1536_MaskProb0.5_MaskLen6_4Nodes/Step60k.ckpt \


###### 3BM model ########
# CUDA_LAUNCH_BLOCKING=1 python examples/nlp/language_modeling/megatron_t5_speechlm.py \
# --config-name=megatron_t5_speechlm_3b_sft.yaml \
# model.data.train_task="tts" \
# +model.freeze_model=False \
# model.data.max_seq_length=2048 \
# model.data.sup_data_path=/data/speech/LibriTTS2/FP_Codec_sup_24khz \
# model.language_model_path=/data/megatron_3b_xTP/megatron_t5_expanded_vocab_2_expanded_vocab_2.nemo > debug.log
