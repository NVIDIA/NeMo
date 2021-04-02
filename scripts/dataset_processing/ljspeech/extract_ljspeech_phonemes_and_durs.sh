#!/bin/bash

# Extracts the phonemes and alignments for the LJSpeech dataset via the
# Montreal Forced Aligner (MFA) library, and computes durations per phoneme.
# Assumes you have downloaded and expanded the LJSpeech dataset, and that they
# are located at the directory specified.
# Also requires that you have manifests set up pointing to the appropriate
# train/val/test wav files.
#
# This script will create:
# - <LJSPEECH_BASE>/mappings.json: Contains word->phone and phone->idx mappings
# - <LJSPEECH_BASE>/phoneme_durations/LJ*.npz: Numpy files for each wavfile
#     with fields 'tokens' (token indices) and 'durs' (durations per token)
#
# It will also create intermediate files:
# - <LJSPEECH_BASE>/wavs/LJ*.txt: Transcripts per wavfile (forA MFA alignment)
# - <LJSPEECH_BASE>/alignments/wavs/LJ*.TextGrid: MFA output
# - <LJSPEECH_BASE>/ljspeech_dict.txt: MFA-generated word->phone mapping
#
# Note: This script will create a conda environment to set up MFA, so please
#       make sure that conda is active before running this script.
#
# Note: You may need to install OpenBlas yourself if the MFA script can't find
#       it. (e.g. sudo apt-get install libopenblas-dev)
#
# Example Usage:
# ./extract_ljspeech_phonemes_and_durs.sh \
#   --train_manifest=/data/manifests/ljspeech_train.json \
#   --val_manifest=/data/manifests/ljspeech_val.json \
#   --test_manifest=/data/manifests/ljspeech_test.json \
#   /data/LJSpeech-1.1

ENV_NAME='aligner'

SAMPLE_RATE=22050
WINDOW_STRIDE=256

# Usage info
show_help() {
cat << EOF
Usage: $(basename "$0") [-h] \
          [--skip_env_setup] \
          [--skip_txt_setup] \
          [--train_manifest=<TRAIN_MANIFET_PATH>] \
          [--val_manifest=<VAL_MANIFEST_PATH>] \
          [--test_manifest=<TEST_MANIFEST_PATH>] \
          [--g2p_dict=<G2P_DICT_PATH>] \
          <LJSPEECH_BASE>
Extracts phonemes and their respective durations for the LJSpeech dataset using
the Montreal Forced Aligner (MFA).
This script assumes you already have manifests set up with the text
normalization that you expect.

    -h                Help message
    --skip_env_setup  (Optional) Skips setting up the MFA conda environment
                      "aligner".  Use only if you already have this set up.
    --skip_txt_setup  (Optional) Skips creation of the .txt files corresponding
                      to the .wav files.
    --g2p_dict        (Optional) Path to the grapheme to phoneme dictionary
                      text file, if already generated. If set, skips the G2P
                      step.
    --train_manifest  (Optional) Path to the training split manifest. Not
                      needed if --skip_txt_setup is set.
    --val_manifest    (Optional) Path to the validation split manifest. Not
                      needed if --skip_txt_setup is set.
    --test_manifest   (Optional) Path to the test split manifest. Not needed if
                      --skip_txt_setup is set.
EOF
}

SKIP_ENV_SETUP=false
SKIP_TXT_SETUP=false
G2P_DICT=''

TRAIN_MANIFEST=''
VAL_MANIFEST=''
TEST_MANIFEST=''

while :; do
  case $1 in
    -h|-\?|--help)
      show_help
      exit
      ;;
    --g2p_dict=?*)
      G2P_DICT=${1#*=}
      ;;
    --skip_env_setup)
      SKIP_ENV_SETUP=true
      ;;
    --skip_txt_setup)
      SKIP_TXT_SETUP=true
      ;;
    --train_manifest=?*)
      TRAIN_MANIFEST=${1#*=}
      ;;
    --val_manifest=?*)
      VAL_MANIFEST=${1#*=}
      ;;
    --test_manifest=?*)
      TEST_MANIFEST=${1#*=}
      ;;
    *)
      break
  esac
  shift
done

if [[ -z $1 ]]; then
  echo "Must specify a LJSpeech base directory."
  exit 1
fi
LJSPEECH_BASE=$1

# Check for conda
read -r -d '' CONDA_MESSAGE << EOM
This script requires either Anaconda or Miniconda to be active, as it installs the Montreal Forced Aligner via a conda environment.
See their documentation (https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html) for more details.
EOM
conda -h > /dev/null || echo $CONDA_MESSAGE

CONDA_PREFIX=$(conda info --base)
source $CONDA_PREFIX/etc/profile.d/conda.sh

# Set up env for MFA (env name "aligner") and install
if $SKIP_ENV_SETUP; then
  echo "Skipping environment setup. Assuming env name "aligner" exists."
else
  echo "Setting up conda environment for MFA (env name \"aligner\")..."
  conda create -n $ENV_NAME -c conda-forge openblas python=3.8 openfst pynini ngram baumwelch unidecode
  conda activate $ENV_NAME
  pip install montreal-forced-aligner
  mfa thirdparty download
  echo "Conda environment \"$ENV_NAME\" set up."
fi
conda activate $ENV_NAME

# Set up transcript .txt files for the MFA library to find later.
# Make sure that the file paths in your manifests point to the correct directory.
if $SKIP_TXT_SETUP; then
  echo "Skipping generation of .txt files."
else
  for f in $TRAIN_MANIFEST $VAL_MANIFEST $TEST_MANIFEST; do
    echo "Creating .txt files with basic text normalization for files in: $f"
    python manifest_to_textfiles.py --manifest=$f
  done
fi

# Download English G2P model and extract phonemes
if [ -z $G2P_DICT ]; then
  echo "Downloading the English G2P model and extracting phonemes..."
  mfa download g2p english_g2p
  G2P_DICT=$LJSPEECH_BASE/ljspeech_dict.txt
  mfa g2p english_g2p $LJSPEECH_BASE $G2P_DICT
fi

# Run alignment
echo "Starting MFA..."
mfa download acoustic english
mfa align --clean $LJSPEECH_BASE $G2P_DICT english $LJSPEECH_BASE/alignments

# Create JSON mappings from word to phonemes and phonemes to indices
echo "Creating word->phone and phone->idx mappings at $LJSPEECH_BASE/mappings.json..."
python create_token2idx_dict.py \
  --dictionary=$LJSPEECH_BASE/ljspeech_dict.txt \
  --dict_out=$LJSPEECH_BAES/mappings.json

# Calculate phoneme durations
echo "Calculating phoneme durations..."
python calculate_durs \
  --ljspeech_dir=$LJSPEECH_BASE \
  --mappings=$LJSPEECH_BASE/mappings.json \
  --sr=$SAMPLE_RATE \
  --window_stride=$WINDOW_STRIDE
echo "Phoneme durations and tokens written to .npz files."
