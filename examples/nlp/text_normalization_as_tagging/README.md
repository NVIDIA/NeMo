# Thuthmose-tagger, a single-pass model for inverse text normalization (ITN).

## ITN Pipeline

1. Data preparation
    `prepare_dataset_en.sh` - English
   or
    `prepare_dataset_ru.sh` - Russian

2. Training a model
    `normalization_as_tagging_train.py`

3. Inference and evaluation
    `run_infer.sh`

## TN Pipeline (experimental)
The difference between TN and ITN is the following:
    - in ITN each input word is mapped to some tag
    - in TN we first have to tokenize input words (e.g. long numbers) into parts that can be mapped to words.

1. Data preparation
    `prepare_dataset_en.sh` - same as for ITN
    `get_tn_tokenizer_dataset.sh` - prepare data for training tokenizer
    `get_tn_dataset.sh` - prepare data for training TN tagger

2. Training two models
    `train_tn_tokenizer.sh`
    `train_tn.sh`

3. Inference and evaluation
    `run_infer_tn.sh`
