
NEMO_BASEPATH="/home/heh/codes/nemo-ssl"
export PYTHONPATH=$NEMO_BASEPATH:$PYTHONPATH

data_dir="/media/data/datasets/LibriSpeech"
train_manifests="[${data_dir}/train_clean_360_cleaned.json,${data_dir}/train_clean_100_cleaned.json,${data_dir}/train_other_500_cleaned.json]"
# train_manifests="[${data_dir}/test_clean.json]"
dev_manifests="${data_dir}/dev_clean_cleaned.json"
batch_size=16
num_workers=8

TRAIN_IS_TARRED=True
TRAIN_MANIFEST='/media/data3/librispeech_tarred/tarred_audio_manifest.json'
TRAIN_FILEPATHS="/media/data3/librispeech_tarred/audio__OP_0..511_CL_.tar"

CUDA_VISIBLE_DEVICES="0,1" python speech_pretrain.py \
    --config-path="configs/" \
    --config-name="conformer_large_ssl_rq" \
    model.train_ds.manifest_filepath=${TRAIN_MANIFEST} \
    ++model.train_ds.is_tarred=${TRAIN_IS_TARRED} \
    ++model.train_ds.tarred_audio_filepaths=${TRAIN_FILEPATHS} \
    model.validation_ds.manifest_filepath=$dev_manifests \
    model.train_ds.batch_size=$batch_size \
    model.validation_ds.batch_size=$batch_size \
    model.train_ds.num_workers=$num_workers \
    model.validation_ds.num_workers=$num_workers \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="ssl_conformer_large_rq_ls_tarred" \
    exp_manager.wandb_logger_kwargs.project="ssl_ConformerL-RQ_ls" \
    ++trainer.accumulate_grad_batches=64

