#!/bin/bash
# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# Download the preprocessed version of MSR Abstractive Text Compression Dataset
# Original dataset: https://www.microsoft.com/en-us/download/details.aspx?id=54262
wget https://dldata-public.s3.us-east-2.amazonaws.com/msr_ab_sum.tar.gz
tar xzvf msr_ab_sum.tar.gz
rm msr_ab_sum.tar.gz

PRETRAINED_MODEL_NAME=bert-base-cased

TASK=msr_ab_sum

TRAIN_FILE=./data/${TASK}/train.tsv
EVAL_FILE=./data/${TASK}/valid.tsv
TEST_FILE=./data/${TASK}/test.tsv

PHRASE_VOCAB_SIZE=500
MAX_INPUT_EXAMPLES=1000000
OUTPUT_DIR=./outputs/${TASK}
MAX_SEQ_LENGTH=128

if [ ! -d ${OUTPUT_DIR} ]; then
    mkdir -p ${OUTPUT_DIR};
fi

# Phrase Vocabulary Optimization to generate train vocabulary tags
# for KEEP/DELETE/ADD/SWAP
python phrase_vocabulary_optimization.py \
  --input_file=${TRAIN_FILE} \
  --vocabulary_size=${PHRASE_VOCAB_SIZE} \
  --max_input_examples=${MAX_INPUT_EXAMPLES} \
  --output_file=${OUTPUT_DIR}/label_map.txt


# Preprocess text to tags
python lasertagger_preprocessor.py \
    --train_file=${TRAIN_FILE} \
    --eval_file=${EVAL_FILE} \
    --test_file=${TEST_FILE} \
    --pretrained_model_name=${PRETRAINED_MODEL_NAME} \
    --label_map_file=${OUTPUT_DIR}/label_map.txt \
    --max_seq_length=${MAX_SEQ_LENGTH} \
    --save_path=${OUTPUT_DIR}


# Training and evaluation, comment --eval_file_preprocessed to skip evaluation
python lasertagger_main.py train \
    --train_file_preprocessed=${OUTPUT_DIR}/lt_train_examples.pkl \
    --eval_file_preprocessed=${OUTPUT_DIR}/lt_eval_examples.pkl \
    --test_file_preprocessed=${OUTPUT_DIR}/lt_test_examples.pkl \
    --pretrained_model_name=${PRETRAINED_MODEL_NAME} \
    --label_map_file=${OUTPUT_DIR}/label_map.txt \
    --max_seq_length=${MAX_SEQ_LENGTH} \
    --work_dir=${OUTPUT_DIR}/lt


# Infer
python lasertagger_main.py infer \
    --test_file=${TEST_FILE} \
    --test_file_preprocessed=${OUTPUT_DIR}/lt_test_examples.pkl \
    --pretrained_model_name=${PRETRAINED_MODEL_NAME} \
    --label_map_file=${OUTPUT_DIR}/label_map.txt \
    --max_seq_length=${MAX_SEQ_LENGTH} \
    --work_dir=${OUTPUT_DIR}/lt
