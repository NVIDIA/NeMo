
NEMO_BASEPATH="/home/heh/codes/nemo-ssl"
export PYTHONPATH=$NEMO_BASEPATH:$PYTHONPATH


data_dir="/media/data3/datasets/LibriTTS"

train_manifests="${data_dir}/libri_train_selfvc.json"
dev_manifests="${data_dir}/libri_val_selfvc.json"

batch_size=32
num_workers=8

ssl_ckpt=/home/heh/codes/nemo-ssl/workspace/nemo_experiments/pretrained_checkpoints/oci_ll_vox_asrset_bs2048_adamwlr0.004_wd1e-3_warmup25000_epoch200_mask0.01x40pre_conv_wavLM0.2x0.1_n16_r1--val_loss5.6475-epoch71-last.ckpt
noise_manifest="[/media/data3/datasets/noise_data/musan/musan_nonspeech_manifest.json,/media/data3/datasets/noise_data/freesound/freesound_noise_manifest_filtered.json]"

exp_name=selfvc_ssl_fastconformer_large_rq_ls_dns_d256_r4

CUDA_VISIBLE_DEVICES="1" python speech_pretrain_denoise.py \
    --config-path="configs" \
    --config-name="fastconformer_large_ssl_rq_dns_selfvc" \
    ++init_from_ptl_ckpt.ssl.path=$ssl_ckpt \
    ++init_from_ptl_ckpt.ssl.exclude=["preprocessor","decoder"] \
    model.train_ds.manifest_filepath=${train_manifests} \
    model.train_ds.noise_manifest=$noise_manifest \
    model.validation_ds.manifest_filepath=$dev_manifests \
    model.validation_ds.noise_manifest=$noise_manifest \
    model.train_ds.batch_size=$batch_size \
    model.validation_ds.batch_size=$batch_size \
    model.train_ds.num_workers=$num_workers \
    model.validation_ds.num_workers=$num_workers \
    model.optim.name="adamw" \
    model.optim.lr=0.00005 \
    model.optim.betas=[0.9,0.999] \
    model.optim.weight_decay=0.0001 \
    model.train_ds.min_duration=3.0 \
    model.validation_ds.min_duration=3.0 \
    ++trainer.gradient_clip_val=1.0 \
    ++trainer.val_check_interval=1.0 \
    ++trainer.max_epochs=100 \
    exp_manager.name=$exp_name \
    exp_manager.create_wandb_logger=true \
    exp_manager.wandb_logger_kwargs.name=$exp_name \
    exp_manager.wandb_logger_kwargs.project="ssl_WavLM_selfvc"

