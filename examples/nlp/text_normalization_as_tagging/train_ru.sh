NEMO_PATH=/home/aleksandraa/nemo
DATA_PATH=/home/aleksandraa/data/tn_data/ru_pipeline13/datasets
DATASET="itn_sample500k_rest1500k_select_vocab"
LANG=ru
LANGUAGE_MODEL=DeepPavlov/rubert-base-cased

python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/normalization_as_tagging_train.py \
  lang=${LANG} \
  data.validation_ds.data_path=${DATA_PATH}/${DATASET}/valid.tsv \
  data.train_ds.data_path=${DATA_PATH}/${DATASET}/train.tsv \
  data.train_ds.batch_size=128 \
  data.train_ds.num_workers=8 \
  model.language_model.pretrained_model_name=${LANGUAGE_MODEL} \
  model.label_map=${DATA_PATH}/label_map.txt \
  model.semiotic_classes=${DATA_PATH}/semiotic_classes.txt \
  model.optim.lr=3e-5 \
  trainer.devices=[0] \
  trainer.num_nodes=1 \
  trainer.accelerator=gpu \
  trainer.strategy=ddp \
  trainer.max_epochs=5