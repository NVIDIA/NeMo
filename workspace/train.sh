
NEMO_BASEPATH="/home/heh/codes/nemo-ssl"
export PYTHONPATH=$NEMO_BASEPATH:$PYTHONPATH

data_dir="/media/data/datasets/LibriSpeech"
train_manifests="[${data_dir}/train_clean_360_cleaned.json,${data_dir}/train_clean_100_cleaned.json,${data_dir}/train_other_500_cleaned.json]"
# train_manifests="[${data_dir}/test_clean.json]"
dev_manifests="${data_dir}/dev_clean_cleaned.json"
batch_size=16
num_workers=8

CUDA_VISIBLE_DEVICES="0,1" python speech_pretrain.py \
    --config-path="configs/" \
    --config-name="conformer_large_ssl_rq" \
    model.train_ds.manifest_filepath=$train_manifests \
    model.validation_ds.manifest_filepath=$dev_manifests \
    model.train_ds.batch_size=$batch_size \
    model.validation_ds.batch_size=$batch_size \
    model.train_ds.num_workers=$num_workers \
    model.validation_ds.num_workers=$num_workers \
    exp_manager.name="ssl_conformer_large_rq_ls_debug" \
    exp_manager.create_wandb_logger=False \
    exp_manager.wandb_logger_kwargs.name="ssl_conformer_large_rq_ls" \
    exp_manager.wandb_logger_kwargs.project="ssl_ConformerL-RQ_ls" \
    ++trainer.accumulate_grad_batches=64

