#!/bin/bash

DATA_DIR=slurp_data
RAW_ANNO_DIR=$DATA_DIR/raw_annotations
MANIFESTS_DIR=$DATA_DIR/raw_manifests

echo "Preparing manifests..."
python data_utils/prepare_slurp.py --data_root $RAW_ANNO_DIR --output $MANIFESTS_DIR

echo "Decoding audios and updating manifests..."
python data_utils/decode_resample.py --data_root $DATA_DIR --manifest $MANIFESTS_DIR

echo "Parsing manifests for ASR..."
python data_utils/get_asr_manifests.py $DATA_DIR
