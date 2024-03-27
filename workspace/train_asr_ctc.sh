
NEMO_BASEPATH="/home/heh/codes/nemo-ssl"
export PYTHONPATH=$NEMO_BASEPATH:$PYTHONPATH

data_dir="/media/data3/datasets/librispeech_origin"
train_manifests="[${data_dir}/train_clean_360_cleaned.json,${data_dir}/train_clean_100_cleaned.json,${data_dir}/train_other_500_cleaned.json]"
dev_manifests="[${data_dir}/dev_clean_cleaned.json,${data_dir}/dev_other.json]"


SSL_CKPT=""
TOKENIZER_DIR="/home/heh/codes/nemo-ssl/workspace/nemo_experiments/tokenizers/ls960_spe_v128"

batch_size=8
num_workers=8

CUDA_VISIBLE_DEVICES="0" python speech_to_text_ctc_bpe.py \
    --config-path="/home/heh/codes/nemo-ssl/examples/asr/conf/conformer" \
    --config-name="conformer_ctc_bpe" \
    ++init_from_ptl_ckpt=$SSL_CKPT \
    model.tokenizer.dir=$TOKENIZER_DIR \
    model.train_ds.manifest_filepath=$train_manifests \
    model.validation_ds.manifest_filepath=$dev_manifests \
    model.train_ds.batch_size=$batch_size \
    model.validation_ds.batch_size=$batch_size \
    model.train_ds.num_workers=$num_workers \
    model.validation_ds.num_workers=$num_workers \
    exp_manager.name="stt_conformer_ctc_large_ls_debug" \
    exp_manager.create_wandb_logger=False \
    exp_manager.wandb_logger_kwargs.name="ssl_conformer_large_rq_ls_debug" \
    exp_manager.wandb_logger_kwargs.project="ssl_asr_ctc"
