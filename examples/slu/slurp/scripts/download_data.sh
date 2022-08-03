#!/bin/bash
# script adapted from https://github.com/pswietojanski/slurp/blob/master/scripts/download_audio.sh

DATA_DIR=slurp_data
mkdir -p $DATA_DIR

wget --help | grep -q '\--show-progress' && \
  PROGRESS_OPT="--show-progress --progress=bar:force" || PROGRESS_OPT=""

echo "Downloading slurp audio data..."
wget -c -q $PROGRESS_OPT \
     https://zenodo.org/record/4274930/files/slurp_real.tar.gz \
     -O $DATA_DIR/slurp_real.tar.gz 2>&1 | tee $DATA_DIR/slurp_real_download.log 

wget -c -q $PROGRESS_OPT \
     https://zenodo.org/record/4274930/files/slurp_synth.tar.gz \
     -O $DATA_DIR/slurp_synth.tar.gz 2>&1 | tee $DATA_DIR/slurp_synth_download.log

echo "Extracting audio files to ${DATA_DIR}/slurp*"
tar -zxvf $DATA_DIR/slurp_real.tar.gz -C $DATA_DIR
tar -zxvf $DATA_DIR/slurp_synth.tar.gz -C $DATA_DIR

echo "Downloading annotations..."
mkdir -p $DATA_DIR/raw_annotations
wget https://github.com/pswietojanski/slurp/raw/master/dataset/slurp/test.jsonl -P $DATA_DIR/raw_annotations
wget https://github.com/pswietojanski/slurp/raw/master/dataset/slurp/devel.jsonl -P $DATA_DIR/raw_annotations
wget https://github.com/pswietojanski/slurp/raw/master/dataset/slurp/train_synthetic.jsonl -P $DATA_DIR/raw_annotations
wget https://github.com/pswietojanski/slurp/raw/master/dataset/slurp/train.jsonl -P $DATA_DIR/raw_annotations

echo "Done."