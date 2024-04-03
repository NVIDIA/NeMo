
NEMO_BASEPATH="/home/heh/codes/nemo-ssl"
export PYTHONPATH=$NEMO_BASEPATH:$PYTHONPATH

data_dir="/media/data2/datasets/speaker_datasets/voxceleb1"
train_manifests="${data_dir}/train_chunk_manifest_rel.json"
dev_manifests="${data_dir}/dev_manifest_rel.json"
noise_manifest="/media/data2/simulated_data/rir_noise_data/white_noise_1ch_37h.json"
CKPT_DIR=/home/heh/codes/nemo-ssl/workspace/nemo_experiments/nemo_experiments/ssl_WavLM/
EXP_NAME="oci_ll_unlab-60k_bs2048_adamwlr0.004_wd1e-3_warmup25000_epoch1000_mask0.01x40pre_conv_wavLM0.2x0.1_n8_r2"
CKPT_NAME="oci_ll_unlab-60k_bs2048_adamwlr0.004_wd1e-3_warmup25000_epoch1000_mask0.01x40pre_conv_wavLM0.2x0.1_n8_r2--val_loss=5.3055-epoch=20-last.ckpt"
POSTFIX=epoch20

SSL_CKPT="'${CKPT_DIR}/${EXP_NAME}/checkpoints/${CKPT_NAME}'"

batch_size=8
num_workers=8

CUDA_VISIBLE_DEVICES="0" python speaker_id_train.py \
    --config-path="configs" \
    --config-name="ecapa_tdnn_ssl" \
    ++init_from_ptl_ckpt=$SSL_CKPT \
    trainer.log_every_n_steps=10 \
    model.train_ds.manifest_filepath=$train_manifests \
    model.train_ds.augmentor.noise.manifest_path=$noise_manifest \
    ++model.train_ds.min_duration=0.5 \
    model.validation_ds.manifest_filepath=$dev_manifests \
    model.train_ds.batch_size=$batch_size \
    model.validation_ds.batch_size=$batch_size \
    model.train_ds.num_workers=$num_workers \
    model.validation_ds.num_workers=$num_workers \
    exp_manager.checkpoint_callback_params.save_top_k=1 \
    exp_manager.name="$EXP_NAME-${POSTFIX}" \
    exp_manager.create_wandb_logger=False \
    exp_manager.wandb_logger_kwargs.name="$EXP_NAME-${POSTFIX}" \
    exp_manager.wandb_logger_kwargs.project="ssl_sid_tdnn"
