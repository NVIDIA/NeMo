#!/bin/bash

# default values for optional arguments
MIN_SCORE=-5
CUT_PREFIX=0
SCRIPTS_DIR="scripts"
OFFSET=0
LANGUAGE='eng' # 'eng', 'ru', 'other'

for ARG in "$@"
do
    key=$(echo $ARG | cut -f1 -d=)
    value=$(echo $ARG | cut -f2 -d=)

    if [[ $key == *"--"* ]]; then
        v="${key/--/}"
        declare $v="${value}"
    fi
done

echo "MODEL_NAME_OR_PATH = $MODEL_NAME_OR_PATH"
echo "DATA_DIR = $DATA_DIR"
echo "OUTPUT_DIR = $OUTPUT_DIR"
echo "MIN_SCORE = $MIN_SCORE"
echo "CUT_PREFIX = $CUT_PREFIX"
echo "SCRIPTS_DIR = $SCRIPTS_DIR"
echo "OFFSET = $OFFSET"
echo "LANGUAGE = $LANGUAGE"

if [[ -z $MODEL_NAME_OR_PATH ]] || [[ -z $DATA_DIR ]] || [[ -z $OUTPUT_DIR ]]; then
  echo "Usage: $(basename "$0")
  --MODEL_NAME_OR_PATH=[model_name_or_path]
  --DATA_DIR=[data_dir]
  --OUTPUT_DIR=[output_dir]
  --LANGUAGE=[language (Optional)]
  --OFFSET=[offset value (Optional)]
  --CUT_PREFIX=[cut prefix in sec (Optional)]
  --SCRIPTS_DIR=[scripts_dir_path (Optional)]"
  exit 1
fi

# STEP #1
# Prepare text and audio data for segmentation
python $SCRIPTS_DIR/prepare_data.py \
--in_text=$DATA_DIR \
--output_dir=$OUTPUT_DIR/processed/ \
--language=$LANGUAGE \
--cut_prefix=$CUT_PREFIX \
--model=$MODEL_NAME_OR_PATH \
--audio_dir=$DATA_DIR || exit

# STEP #2
# Run CTC-segmenatation
# one might want to perform alignment with various window sizes
# note if the alignment with the initial window size isn't found, the window size will be double to re-attempt
# alignment
for WINDOW in 8000 12000
do
  python $SCRIPTS_DIR/run_ctc_segmentation.py \
  --output_dir=$OUTPUT_DIR \
  --data=$OUTPUT_DIR/processed/ \
  --model=$MODEL_NAME_OR_PATH  \
  --window_len $WINDOW || exit
done

# STEP #3 (Optional)
# Verify aligned segments only if multiple WINDOWs used in the Step #2)
python $SCRIPTS_DIR/verify_segments.py \
--base_dir=$OUTPUT_DIR  || exit

# STEP #4
# Cut the original audio files based on the alignments
# (use --alignment=$OUTPUT_DIR/segments if only 1 WINDOW size was used in the Step #2)
# Three manifests and corresponding clips folders will be created:
#   - high scored clips
#   - low scored clips
#   - deleted segments
python $SCRIPTS_DIR/cut_audio.py \
--output_dir=$OUTPUT_DIR \
--model=$MODEL_NAME_OR_PATH \
--alignment=$OUTPUT_DIR/verified_segments \
--threshold=$MIN_SCORE \
--offset=$OFFSET || exit

# STEP #5 (Optional)
# If multiple audio files were segmented in the step #2, this step will aggregate manifests with high scored segments
# for all audio files into all_manifest.json
# Also a separate manifest with samples from across all high scored segments will be credated if --num_samples > 0
# --num_samples samples will be taken from the beginning, end and the middle of the each audio file high score manifest
# and will be stored at sample_manifest.json
python $SCRIPTS_DIR/process_manifests.py \
--output_dir=$OUTPUT_DIR \
--manifests_dir=$OUTPUT_DIR/manifests/ \
--num_samples 0

exit

