
NEMO_BASEPATH="/home/heh/codes/nemo-ssl"
export PYTHONPATH=$NEMO_BASEPATH:$PYTHONPATH

DATA_DIR=/media/data2/datasets/speaker_datasets
NUM_CLASSES=7205
vox1_dir="/media/data2/datasets/speaker_datasets/voxceleb1"
vox2_dir="/media/data2/datasets/speaker_datasets/voxceleb2"

train_manifests="[${vox1_dir}/vox1_train_manifest_train_chunk3s.json,${vox2_dir}/vox2_all_manifest_train_chunk3s.json]"
dev_manifests="[${vox1_dir}/vox1_train_manifest_val_chunk30s.json,${vox2_dir}/vox2_all_manifest_val_chunk30s.json]"

# train_manifests="[${vox2_dir}/vox2_all_manifest_train_chunk3s.json]"
# dev_manifests="[${vox2_dir}/vox2_all_manifest_val_chunk30s.json]"

# train_manifests="[${DATA_DIR}/vox1_musan_val/vox1_noisy_val_manifest.json,${DATA_DIR}/vox2_musan_val/vox2_noisy_val_manifest.json]"
# dev_manifests="[${DATA_DIR}/vox1_musan_val/vox1_noisy_val_manifest.json,${DATA_DIR}/vox2_musan_val/vox2_noisy_val_manifest.json]"

train_manifests="[${DATA_DIR}/vox2_musan_val/debug.json]"
dev_manifests="[${DATA_DIR}/vox2_musan_val/debug.json,${DATA_DIR}/vox2_musan_val/debug2.json]"

# noise_manifest="/media/data2/simulated_data/rir_noise_data/white_noise_1ch_37h.json"
noise_manifest="[/media/data3/datasets/noise_data/musan/musan_nonspeech_manifest.json,/media/data3/datasets/noise_data/freesound/freesound_noise_manifest_filtered.json]"
rir_manifest="/media/data2/simulated_data/rir_noise_data/real_rirs_isotropic_noises_1ch.json"

POSTFIX=debug3
EXP_NAME="titanet_small"

SSL_CKPT="/home/heh/codes/nemo-ssl/workspace/nemo_experiments/pretrained_checkpoints/oci_ll_unlab-60k_bs2048_adamwlr0.004_wd1e-3_warmup25000_epoch1000_mask0.01x40pre_conv_wavLM0.2x0.1_n16_r5--val_loss5.0808-epoch43-last.ckpt"

batch_size=4
num_workers=8

# model.train_ds.augmentor.impulse.manifest_path=$rir_manifest \
SCRIPT_DIR=$NEMO_BASEPATH/examples/speaker_tasks/recognition
SCRIPT=$SCRIPT_DIR/speaker_reco.py

CUDA_VISIBLE_DEVICES="0" python $SCRIPT \
    --config-path="conf" \
    --config-name="titanet-small" \
    trainer.devices=-1 \
    trainer.log_every_n_steps=10 \
    model.decoder.num_classes=$NUM_CLASSES \
    model.train_ds.manifest_filepath=$train_manifests \
    model.train_ds.augmentor.noise.manifest_path=$noise_manifest \
    model.train_ds.augmentor.impulse.manifest_path=$rir_manifest \
    ++model.train_ds.min_duration=0.3 \
    ++model.validation_ds.min_duration=0.3 \
    model.train_ds.shuffle=False \
    ++model.train_ds.drop_last=True \
    ++model.validation_ds.drop_last=True \
    model.validation_ds.manifest_filepath=$dev_manifests \
    model.train_ds.batch_size=$batch_size \
    model.validation_ds.batch_size=4 \
    ++model.train_ds.num_workers=$num_workers \
    ++model.validation_ds.num_workers=$num_workers \
    trainer.val_check_interval=1.0 \
    ++exp_manager.checkpoint_callback_params.save_top_k=1 \
    exp_manager.name="$EXP_NAME-${POSTFIX}" \
    ++exp_manager.create_wandb_logger=True \
    ++exp_manager.wandb_logger_kwargs.name="$EXP_NAME-${POSTFIX}" \
    ++exp_manager.wandb_logger_kwargs.project="debug_spk_id"
