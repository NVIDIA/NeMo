#!/bin/bash

TASK=msr_ab_sum

TRAIN_FILE=./data/${TASK}/train.tsv
EVAL_FILE=./data/${TASK}/valid.tsv
TEST_FILE=./data/${TASK}/test.tsv

PHRASE_VOCAB_SIZE=500
MAX_INPUT_EXAMPLES=1000000
OUTPUT_DIR=./outputs/${TASK}

# Phrase Vocabulary Optimization
python phrase_vocabulary_optimization.py \
  --input_file=${TRAIN_FILE} \
  --vocabulary_size=${PHRASE_VOCAB_SIZE} \
  --max_input_examples=${MAX_INPUT_EXAMPLES} \
  --output_file=${OUTPUT_DIR}/label_map.txt


VOCAB_FILE=./data/bert_base_cased/vocab.txt
MAX_SEQ_LENGTH=128
LASERTAGGER_CONFIG=./configs/lasertagger_config.json

# Preprocess text to tags
python lasertagger_preprocessor.py \
    --train_file=${TRAIN_FILE} \
    --eval_file=${EVAL_FILE} \
    --test_file=${TEST_FILE} \
    --label_map_file=${OUTPUT_DIR}/label_map.txt \
    --vocab_file=${VOCAB_FILE} \
    --save_path=${OUTPUT_DIR}

# Training and evaluation, comment --eval_file to skip evaluation
python lasertagger_main.py train \
    --train_file=${OUTPUT_DIR}/lt_train_examples.pkl \
    --eval_file=${OUTPUT_DIR}/lt_eval_examples.pkl \
    --test_file=${OUTPUT_DIR}/lt_test_examples.pkl \
    --label_map_file=${OUTPUT_DIR}/label_map.txt \
    --vocab_file=${VOCAB_FILE} \
    --max_seq_length=${MAX_SEQ_LENGTH} \
    --model_config_file=${LASERTAGGER_CONFIG} \
    --work_dir=${OUTPUT_DIR}/lt

# Infer
python lasertagger_main.py infer \
    --test_file_raw=${TEST_FILE} \
    --test_file_preprocessed=${OUTPUT_DIR}/lt_test_examples.pkl \
    --label_map_file=${OUTPUT_DIR}/label_map.txt \
    --vocab_file=${VOCAB_FILE} \
    --max_seq_length=${MAX_SEQ_LENGTH} \
    --model_config_file=${LASERTAGGER_CONFIG} \
    --work_dir=${OUTPUT_DIR}/lt
