
NEMO_BASEPATH="/home/heh/codes/nemo-ssl"
export PYTHONPATH=$NEMO_BASEPATH:$PYTHONPATH

data_dir="/media/data3/datasets/librispeech_origin"
train_manifests="${data_dir}/train_clean_100_cleaned_phonene.json"
dev_manifests="[${data_dir}/dev_clean_cleaned_phonene.json,${data_dir}/test_clean_phonene.json]"

batch_size=32
num_workers=8
epochs=100
lr=1e-4
wd=1e-4
warmup_steps=2000


EXP_NAME="superb_pr_ssl_fc_large_ctc"
POSTFIX=r1


SSL_CKPT=/home/heh/codes/nemo-ssl/workspace/nemo_experiments/pretrained_checkpoints/oci_ll_vox_asrset_bs2048_adamwlr0.004_wd1e-3_warmup25000_epoch200_mask0.01x40pre_conv_wavLM0.2x0.1_n16_r1--val_loss5.6475-epoch71-last.ckpt
TOKENIZER_DIR="/home/heh/codes/nemo-ssl/workspace/local_files/ls100_phonene/tokenizer_spe_word_v72"



CUDA_VISIBLE_DEVICES="0" python speech_to_text_ctc_bpe.py \
    --config-path="/home/heh/codes/nemo-ssl/examples/asr/conf/fastconformer" \
    --config-name="fast-conformer_ctc_bpe" \
    ++init_from_ptl_ckpt.ssl.path=$SSL_CKPT \
    ++init_from_ptl_ckpt.ssl.include=["encoder"] \
    ++init_from_ptl_ckpt.ssl.exclude=["decoder"] \
    trainer.log_every_n_steps=10 \
    model.tokenizer.dir=$TOKENIZER_DIR \
    model.train_ds.manifest_filepath=$train_manifests \
    model.validation_ds.manifest_filepath=$dev_manifests \
    model.train_ds.batch_size=$batch_size \
    model.validation_ds.batch_size=$batch_size \
    model.train_ds.num_workers=$num_workers \
    model.validation_ds.num_workers=$num_workers \
    exp_manager.checkpoint_callback_params.save_top_k=1 \
    exp_manager.name="$EXP_NAME-${POSTFIX}" \
    exp_manager.create_wandb_logger=false \
    exp_manager.wandb_logger_kwargs.name="$EXP_NAME-${POSTFIX}" \
    exp_manager.wandb_logger_kwargs.project="superb"
