
NEMO_BASEPATH="/home/heh/codes/nemo-ssl"
export PYTHONPATH=$NEMO_BASEPATH:$PYTHONPATH

data_dir="/media/data3/datasets/librispeech_origin"
# train_manifests="[${data_dir}/train_clean_360_cleaned.json,${data_dir}/train_clean_100_cleaned.json,${data_dir}/train_other_500_cleaned.json]"
train_manifests="[${data_dir}/test_clean.json]"
dev_manifests="[${data_dir}/dev_clean_cleaned.json,${data_dir}/dev_other.json]"

noise_manifest="/media/data3/datasets/noise_data/musan/musan_nonspeech_manifest-max30s.json"

batch_size=8
num_workers=0

exp_name=ssl_fastconformer_large_rq_ls_dns_debug

CUDA_VISIBLE_DEVICES="0" python speech_pretrain_denoise.py \
    --config-path="configs/" \
    --config-name="fastconformer_large_ssl_rq_dns" \
    model.encoder.n_layers=4 \
    model.train_ds.manifest_filepath=$train_manifests \
    model.train_ds.noise_manifest=$noise_manifest \
    model.validation_ds.manifest_filepath=$dev_manifests \
    model.validation_ds.noise_manifest=$noise_manifest \
    model.train_ds.batch_size=$batch_size \
    model.validation_ds.batch_size=$batch_size \
    model.train_ds.num_workers=$num_workers \
    model.validation_ds.num_workers=$num_workers \
    exp_manager.name=$exp_name \
    exp_manager.create_wandb_logger=False \
    exp_manager.wandb_logger_kwargs.name=$exp_name \
    exp_manager.wandb_logger_kwargs.project="ssl_ConformerL-RQ_ls"
