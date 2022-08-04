#!/bin/bash
DATA_DIR="slurp_data"
mkdir -p $DATA_DIR

echo "Downloading slurp audio data..."
wget https://zenodo.org/record/4274930/files/slurp_real.tar.gz -P $DATA_DIR
wget https://zenodo.org/record/4274930/files/slurp_synth.tar.gz -P $DATA_DIR

echo "Extracting audio files to ${DATA_DIR}/slurp*"
tar -zxvf $DATA_DIR/slurp_real.tar.gz -C $DATA_DIR
tar -zxvf $DATA_DIR/slurp_synth.tar.gz -C $DATA_DIR

echo "Downloading annotations..."
mkdir -p $DATA_DIR/raw_annotations
wget https://github.com/pswietojanski/slurp/raw/master/dataset/slurp/test.jsonl -P $DATA_DIR/raw_annotations
wget https://github.com/pswietojanski/slurp/raw/master/dataset/slurp/devel.jsonl -P $DATA_DIR/raw_annotations
wget https://github.com/pswietojanski/slurp/raw/master/dataset/slurp/train_synthetic.jsonl -P $DATA_DIR/raw_annotations
wget https://github.com/pswietojanski/slurp/raw/master/dataset/slurp/train.jsonl -P $DATA_DIR/raw_annotations

echo "Downloading evaluation code..."
wget https://github.com/pswietojanski/slurp/raw/master/scripts/evaluation/util.py -P eval_utils/evaluation
wget https://github.com/pswietojanski/slurp/raw/master/scripts/evaluation/metrics/distance.py -P eval_utils/evaluation/metrics
wget https://github.com/pswietojanski/slurp/raw/master/scripts/evaluation/metrics/metrics.py -P eval_utils/evaluation/metrics

echo "Done."