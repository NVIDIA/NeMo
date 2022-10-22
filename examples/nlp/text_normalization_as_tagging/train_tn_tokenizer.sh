WANDB_PROJECT="tn_as_tagging"
WORK_DIR=`pwd`   # directory from which this bash-script is run
echo "Working directory:" ${WORK_DIR}

DATA_PATH=${WORK_DIR}/datasets

EXP_NAME="en9a_tn_tokenizer_ep5_bs128"
DATASET="tn_tokenizer_sample_200k_200k"

#wandb login 8fcf490c8b4f21f2da3465b9f5899bc22fdcfb72 \
#&& echo "***** Starting training *****" \
#&& 
python /home/aleksandraa/nemo/examples/nlp/text_normalization_as_tagging/normalization_as_tagging_train.py \
lang="en" \
data.validation_ds.data_path=${DATA_PATH}/${DATASET}/valid.tsv \
data.train_ds.data_path=${DATA_PATH}/${DATASET}/train.tsv \
data.train_ds.batch_size=128 \
data.train_ds.num_workers=8 \
model.language_model.pretrained_model_name=bert-base-uncased \
model.label_map=${DATA_PATH}/${DATASET}/tn_tokenizer_label_map.txt \
model.semiotic_classes=${DATA_PATH}/${DATASET}/semiotic_classes.txt \
model.optim.lr=3e-5 \
trainer.devices=[0] \
trainer.num_nodes=1 \
trainer.accelerator=gpu \
trainer.strategy=ddp \
trainer.max_epochs=5 \
#+exp_manager.create_wandb_logger=true \
#+exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT} \
#+exp_manager.wandb_logger_kwargs.name=${EXP_NAME}
