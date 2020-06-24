# LaserTagger Abstractive Summarization Example

LaserTagger is a text-editing model which predicts a sequence of token-level
edit operations to transform a source text into a target text. The model
currently supports four different edit operations:

1. *Keep* the token.
2. *Delete* the token.
3. *Add* a phrase before the token.
4. *Swap* the order of input sentences (if there are two of them).

Operation 3 can be combined with 1 and 2. Compared to sequence-to-sequence
models, LaserTagger is (1) less prone to hallucination, (2) more data efficient,
and (3) faster at inference time.

A detailed method description and evaluation can be found in this EMNLP'19 paper:
[https://arxiv.org/abs/1909.01187](https://arxiv.org/abs/1909.01187)

This example contains code artifacts adapted from the original implementation:
https://github.com/google-research/lasertagger

## Usage Instructions

Running an experiment with LaserTagger consists of the following steps:

1. Optimize the vocabulary of phrases that can be added by LaserTagger.
2. Convert target texts into target tag sequences.
3. Finetune a pretrained BERT model to predict the tags.
4. Compute predictions.
5. Evaluate the predictions.

Next we go through these steps, using Abstractive Summarization
([MSR Abstractive Text Compression](https://www.microsoft.com/en-us/research/publication/dataset-evaluation-metrics-abstractive-sentence-paragraph-compression/)) task as a
running example.

You can run all of the steps with

```
sh run_summarization.sh
```

after setting the paths in the beginning of the script.

**Note:** Text should be tokenized with spaces separating the tokens before applying LaserTagger.

### 1. Dataset Download

Download the preprocessed version of [MSR Abstractive Text Compression Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54262)

```
wget https://dldata-public.s3.us-east-2.amazonaws.com/msr_ab_sum.tar.gz
tar xzvf msr_ab_sum.tar.gz
rm msr_ab_sum.tar.gz
```

The MSR Abstractive Summarization dataset after extracting can be found in the `data` folder. This has been pretokenized with the BERT basic tokenizer (punctuation splitting, lower casing, etc.).

### 2. Phrase Vocabulary Optimization

```
export TASK=msr_ab_sum

export TRAIN_FILE=./data/${TASK}/train.tsv
export EVAL_FILE=./data/${TASK}/valid.tsv
export TEST_FILE=./data/${TASK}/test.tsv

export PHRASE_VOCAB_SIZE=500
export MAX_INPUT_EXAMPLES=1000000
export OUTPUT_DIR=./outputs/${TASK}
export MAX_SEQ_LENGTH=128

if [ ! -d ${OUTPUT_DIR} ]; then
    mkdir -p ${OUTPUT_DIR};
fi

python phrase_vocabulary_optimization.py \
  --input_file=${TRAIN_FILE} \
  --vocabulary_size=${PHRASE_VOCAB_SIZE} \
  --max_input_examples=${MAX_INPUT_EXAMPLES} \
  --output_file=${OUTPUT_DIR}/label_map.txt
```

Note that you can also set `MAX_INPUT_EXAMPLES` to a smaller value to get a
reasonable vocabulary.

### 3. Converting Target Texts to Tags

We've used the 12-layer "BERT-Base, Cased" model for of our experiments.
Then convert the original TSV datasets into pkl format.

```
# Preprocess text to tags
python lasertagger_preprocessor.py \
    --train_file=${TRAIN_FILE} \
    --eval_file=${EVAL_FILE} \
    --test_file=${TEST_FILE} \
    --label_map_file=${OUTPUT_DIR}/label_map.txt \
    --max_seq_length=${MAX_SEQ_LENGTH} \
    --save_path=${OUTPUT_DIR}
```

### 4. Model Training

Model hyperparameters are specified in [lasertagger_main.py](lasertagger_main.py).

```
# Training and evaluation, comment --eval_file_preprocessed to skip evaluation
python lasertagger_main.py train \
    --train_file_preprocessed=${OUTPUT_DIR}/lt_train_examples.pkl \
    --eval_file_preprocessed=${OUTPUT_DIR}/lt_eval_examples.pkl \
    --test_file_preprocessed=${OUTPUT_DIR}/lt_test_examples.pkl \
    --label_map_file=${OUTPUT_DIR}/label_map.txt \
    --max_seq_length=${MAX_SEQ_LENGTH} \
    --work_dir=${OUTPUT_DIR}/lt
```

### 5. Prediction and Evaluation

The inference relies on `ROUGE-L` metric which computes the longest common subsequence (LCS) between two pieces of text.

The inference also calculates another metric called `SARI (System output Against References and against the Input sentence)`. It explicitly measures the goodness of words that are added, deleted and kept by the systems.

```
# Infer
python lasertagger_main.py infer \
    --test_file=${TEST_FILE} \
    --test_file_preprocessed=${OUTPUT_DIR}/lt_test_examples.pkl \
    --label_map_file=${OUTPUT_DIR}/label_map.txt \
    --max_seq_length=${MAX_SEQ_LENGTH} \
    --work_dir=${OUTPUT_DIR}/lt
```

### 6. Acknowledgement

We would like to thank the authors of the original work done by the Google Research team on [Encode, Tag, Realize: High-Precision Text Editing](https://arxiv.org/abs/1909.01187)
