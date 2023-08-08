NEMO_DIR=/workspace/nemo/works/mod_speech_llm/NeMo
export PYTHONPATH=$NEMO_DIR:$PYTHONPATH

MEGATRON_CKPT=/media/data3/pretrained_models/megatron_gpt/gpt_pretrain_220m_len_4096_pos_alibi_step_595508_gbs256.nemo
ASR_MODEL="stt_en_fastconformer_transducer_large"

TRAIN_MANIFESTS=/media/data/datasets/LibriSpeech/train_clean_100_cleaned.json
VAL_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean.json

python -m pdb -c continue run_sft_audio_lm.py --config-path="../examples/multimodel/conf/speechllm/" --config-name "modularized_speech_gpt_config" \
    model.pretrained_audio_model=$ASR_MODEL \
    model.restore_from_path=$MEGATRON_CKPT \
    model.data.train_ds.file_names=$TRAIN_MANIFESTS \
    model.data.validation_ds.file_names=$VAL_MANIFESTS

