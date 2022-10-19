NEMO_PATH=/home/aleksandraa/nemo

DATA_PATH=datasets
DATASET="sent_3m_half_neg"

python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/spellchecking_asr_customization_train.py \
lang="en" \
data.validation_ds.data_path=${DATA_PATH}/${DATASET}/valid.tsv \
data.train_ds.data_path=${DATA_PATH}/${DATASET}/train.tsv \
data.train_ds.batch_size=32 \
data.train_ds.num_workers=8 \
model.max_sequence_len=512 \
model.language_model.pretrained_model_name=huawei-noah/TinyBERT_General_6L_768D \
model.language_model.config_file=${DATA_PATH}/${DATASET}/config.json \
model.label_map=${DATA_PATH}/${DATASET}/label_map.txt \
model.semiotic_classes=${DATA_PATH}/${DATASET}/semiotic_classes.txt \
model.optim.lr=3e-5 \
trainer.devices=[1] \
trainer.num_nodes=1 \
trainer.accelerator=gpu \
trainer.strategy=ddp \
trainer.max_epochs=5
