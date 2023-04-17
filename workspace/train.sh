
NEMO_BASEPATH="/home/heh/codes/nemo-ssl"
export PYTHONPATH=$NEMO_BASEPATH:$PYTHONPATH

data_dir="/media/data/datasets/LibriSpeech"
train_manifests="[${data_dir}/train_clean_360.json,${data_dir}/train_clean_100.json,${data_dir}/train_other_500.json]"
dev_manifests="${data_dir}/dev_clean.json"
num_workers=8

CUDA_VISIBLE_DEVICES="0" python speech_pretrain.py \
    --config-path="configs/" \
    --config-name="conformer_ssl_rq" \
    model.train_ds.manifest_filepath=$train_manifests \
    model.validation_ds.manifest_filepath=$dev_manifests
