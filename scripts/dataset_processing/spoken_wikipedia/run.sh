#!/bin/bash

## Download the Spoken Wikipedia corpus for English
## Note, that there are some other other languages available
wget https://corpora.uni-hamburg.de/hzsk/de/islandora/object/file:swc-2.0_en-with-audio/datastream/TAR/en-with-audio.tar .
tar -xvf en-with-audio.tar

##  We get a folder English with 1339 subfolders, each subfolder corresponds to a Wikipedia article. Example:
##  ├── Universal_suffrage
##  │   ├── aligned.swc
##  │   ├── audiometa.txt
##  │   ├── audio.ogg
##  │   ├── info.json
##  │   ├── wiki.html
##  │   ├── wiki.txt
##  │   └── wiki.xml

##  We will use two files: audio.ogg and wiki.txt

##  Some folders have multiple .ogg files, this will be handled during preprocess.py. Example:
##  |── Universe
##  │   ├── aligned.swc
##  │   ├── audio1.ogg
##  │   ├── audio2.ogg
##  │   ├── audio3.ogg
##  │   ├── audio4.ogg
##  │   ├── audiometa.txt
##  │   ├── info.json
##  │   ├── wiki.html
##  │   ├── wiki.txt
##  │   └── wiki.xml

##  Some rare folders are incomplete, these will be skipped during preprocessing.

## path to NeMo repository, e.g. /home/user/nemo
NEMO_PATH=

INPUT_DIR="english"
OUTPUT_DIR=${INPUT_DIR}_result

rm -rf $OUTPUT_DIR
rm -rf ${INPUT_DIR}_prepared
mkdir ${INPUT_DIR}_prepared
mkdir ${INPUT_DIR}_prepared/audio
mkdir ${INPUT_DIR}_prepared/text
python ${NEMO_PATH}/scripts/dataset_processing/spoken_wikipedia/preprocess.py --input_folder ${INPUT_DIR} --destination_folder ${INPUT_DIR}_prepared


MODEL_FOR_SEGMENTATION="QuartzNet15x5Base-En" 
MODEL_FOR_RECOGNITION="stt_en_conformer_ctc_large"
WINDOW=8000
OFFSET=0
## We set this threshold as very permissive, later we will use other metrics for filtering
THRESHOLD=-10


python ${NEMO_PATH}/tools/ctc_segmentation/scripts/prepare_data.py \
  --in_text=${INPUT_DIR}_prepared/text \
  --output_dir=$OUTPUT_DIR/processed/ \
  --language='en' \
  --model=${MODEL_FOR_SEGMENTATION} \
  --audio_dir=${INPUT_DIR}_prepared/audio \
  --use_nemo_normalization

python ${NEMO_PATH}/tools/ctc_segmentation/scripts/run_ctc_segmentation.py \
  --output_dir=${OUTPUT_DIR} \
  --data=$OUTPUT_DIR/processed \
  --model=${MODEL_FOR_SEGMENTATION} \
  --window_len=$WINDOW 

python ${NEMO_PATH}/tools/ctc_segmentation/scripts/cut_audio.py \
  --output_dir=${OUTPUT_DIR} \
  --alignment=${OUTPUT_DIR}/segments/ \
  --threshold=$THRESHOLD \
  --offset=$OFFSET

## We transcribe the manifest in order to get additional metrics for filtering, based on WER, CER an so on, for individual sentences
python ${NEMO_PATH}/examples/asr/transcribe_speech.py \
  pretrained_name=${MODEL_FOR_RECOGNITION} \
  dataset_manifest=${OUTPUT_DIR}/manifests/manifest.json  \
  output_filename=${OUTPUT_DIR}/manifests/manifest_transcribed.json \
  batch_size=1 \
  cuda=1 \
  amp=True

# Preliminary thresholds for filtering
CER_THRESHOLD=30
WER_THRESHOLD=75
CER_EDGE_THRESHOLD=60
LEN_DIFF_RATIO_THRESHOLD=0.3

python ${NEMO_PATH}/tools/ctc_segmentation/scripts/get_metrics_and_filter.py \
  --manifest=${OUTPUT_DIR}/manifests/manifest_transcribed.json \
  --audio_dir=${INPUT_DIR}_prepared/audio \
  --max_cer=${CER_THRESHOLD} \
  --max_wer=${WER_THRESHOLD} \
  --max_len_diff_ratio=${LEN_DIFF_RATIO_THRESHOLD} \
  --max_edge_cer=${CER_EDGE_THRESHOLD} \
  --edge_len=25

# This script applies final filtering thresholds on metrics (default parameters that can be redefined)
python ${NEMO_PATH}/scripts/dataset_processing/spoken_wikipedia/filter_manifest.py --input_manifest ${OUTPUT_DIR}/manifests/manifest_transcribed_metrics.json --output_manifest ${OUTPUT_DIR}/manifests/manifest_filtered.json

## Calculate CER and WER of the final manifest
python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
  dataset_manifest=${OUTPUT_DIR}/manifests/manifest_filtered.json \
  use_cer=True \
  only_score_manifest=True

python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
  dataset_manifest=${OUTPUT_DIR}/manifests/manifest_filtered.json \
  use_cer=False \
  only_score_manifest=True

