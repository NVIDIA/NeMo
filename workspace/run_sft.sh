NEMO_DIR=/workspace/nemo/works/mod_speech_llm/NeMo
export PYTHONPATH=$NEMO_DIR:$PYTHONPATH

MEGATRON_CKPT=/media/data3/pretrained_models/megatron_gpt/gpt_pretrain_220m_len_4096_pos_alibi_step_595508_gbs256.nemo
ASR_MODEL="stt_en_fastconformer_transducer_large"
GLOBAL_BATCH=1
MICRO_BATCH=1

TRAIN_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean.json
TRAIN_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean_150_r.json
TRAIN_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean_300.json
TRAIN_MANIFESTS=/media/data/datasets/LibriSpeech/train_clean_100_cleaned.json
TRAIN_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean_140_r.shuf.json
TRAIN_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean_10.json
VAL_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean_150_r.json
VAL_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean_2.json
VAL_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean_300.json
VAL_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean.json
VAL_MANIFESTS=[/media/data/datasets/LibriSpeech/dev_clean_11.json,/media/data/datasets/LibriSpeech/dev_clean_10.json]
VAL_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean_10.json

# python \
python -m pdb -c continue \
run_sft_audio_lm.py --config-path="../examples/multimodel/conf/speechllm/" --config-name "modularized_speech_gpt_config" \
    model.pretrained_audio_model=$ASR_MODEL \
    model.restore_from_path=$MEGATRON_CKPT \
    model.global_batch_size=$GLOBAL_BATCH \
    model.micro_batch_size=$MICRO_BATCH \
    model.data.train_ds.manifest_filepath=$TRAIN_MANIFESTS \
    model.data.validation_ds.manifest_filepath=$VAL_MANIFESTS

