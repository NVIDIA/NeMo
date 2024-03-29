
NEMO_BASEPATH="/home/heh/codes/nemo-ssl"
export PYTHONPATH=$NEMO_BASEPATH:$PYTHONPATH

data_dir="/media/data3/datasets/librispeech_origin"
train_manifests="[${data_dir}/train_clean_360_cleaned.json,${data_dir}/train_clean_100_cleaned.json,${data_dir}/train_other_500_cleaned.json]"
dev_manifests="[${data_dir}/dev_clean_cleaned.json,${data_dir}/dev_other.json]"

CKPT_DIR=/home/heh/codes/nemo-ssl/workspace/nemo_experiments/nemo_experiments/ssl_WavLM/
EXP_NAME="rno_ls960_bs2048_adamwlr0.004_wd1e-3_warmup25000_epoch1000_mask0.01x40pre_conv_wavLM0.2x0.1_dgx2h_n8_r1"
CKPT_NAME="rno_ls960_bs2048_adamwlr0.004_wd1e-3_warmup25000_epoch1000_mask0.01x40pre_conv_wavLM0.2x0.1_dgx2h_n8_r1--val_loss=5.3095-epoch=100-last.ckpt"
POSTFIX=epoch100

SSL_CKPT="'${CKPT_DIR}/${EXP_NAME}/checkpoints/${CKPT_NAME}'"
TOKENIZER_DIR="/home/heh/codes/nemo-ssl/workspace/nemo_experiments/tokenizers/ls960_spe_v128"

batch_size=512
num_workers=16

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
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="$EXP_NAME-${POSTFIX}" \
    exp_manager.wandb_logger_kwargs.project="ssl_asr_ctc"
