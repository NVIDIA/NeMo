#!/bin/bash

# Extracts the phonemes and alignments for the LJSpeech dataset via the
# Montreal Forced Aligner (MFA) library, and computes durations per phoneme.
# Assumes you have downloaded and expanded the LJSpeech dataset, and that they
# are located at the directory specified.
#
# This script will create:
# - <LJSPEECH_BASE>/mappings.json: Contains word->phone and phone->idx mappings
# - <LJSPEECH_BASE>/wavs_to_ignore.pkl: A pickled list of wavs to ignore
#     because of OOV words in their transcripts
# - <LJSPEECH_BASE>/phoneme_durations/LJ*.npz: Numpy files for each wavfile
#     with fields 'tokens' (token indices) and 'durs' (durations per token)
#
# It will also create intermediate files:
# - <LJSPEECH_BASE>/alignments/wavs/LJ*.TextGrid: MFA output
#
# Note: This script will create a conda environment to set up MFA, so please
#       make sure that conda is active before running this script.
#
# Note: You may need to install OpenBlas yourself if the MFA script can't find
#       it. (e.g. sudo apt-get install libopenblas-dev)
#
# Example Usage:
# ./extract_ljspeech_phonemes_and_durs.sh /data/LJSpeech-1.1

ENV_NAME='aligner'

SAMPLE_RATE=22050
WINDOW_STRIDE=256

CMUDICT_URL='https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict'
SPLITS_BASE_URL='https://raw.githubusercontent.com/NVIDIA/tacotron2/master/filelists/'

# Usage info
show_help() {
cat << EOF
Usage: $(basename "$0") [-h] \
          [--skip_env_setup] \
          [--g2p_dict=<G2P_DICT_PATH>] \
          <LJSPEECH_BASE>
Extracts phonemes and their respective durations for the LJSpeech dataset using
the Montreal Forced Aligner (MFA).

    -h                Help message
    --skip_env_setup  (Optional) Skips setting up the MFA conda environment
                      "aligner".  Use only if you already have this set up.
    --g2p_dict        (Optional) Path to the grapheme to phoneme dictionary
                      text file, if already generated. If set, skips the G2P
                      step.
EOF
}

SKIP_ENV_SETUP=false
G2P_DICT=''

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
if [ -z "$(which conda)" ]; then
  echo $CONDA_MESSAGE
  exit
fi

CONDA_PREFIX=$(conda info --base)
source $CONDA_PREFIX/etc/profile.d/conda.sh

# Set up env for MFA (env name "aligner") and install
if $SKIP_ENV_SETUP; then
  echo "Skipping environment setup. Assuming env name "aligner" exists."
else
  echo "Setting up conda environment for MFA (env name \"aligner\")..."
  conda create -n $ENV_NAME -c conda-forge openblas python=3.8 openfst pynini ngram baumwelch
  conda activate $ENV_NAME
  pip install montreal-forced-aligner tgt torch
  mfa thirdparty download
  echo "Conda environment \"$ENV_NAME\" set up."
fi

if ! conda activate $ENV_NAME; then
  echo "Could not activate environment, see Conda output above."
  exit
fi

# Download CMU word-to-phoneme dict and clean out comments so they're not mistaken for tokens
if [ -z $G2P_DICT ]; then
  if [ ! -f $LJSPEECH_BASE/cmudict.dict ]; then
    echo "Downloading CMU dict."
    wget -P $LJSPEECH_BASE $CMUDICT_URL
  fi
  if [ ! -f $LJSPEECH_BASE/uncommented_cmudict.dict ]; then
    echo "Creating uncommented version of CMUdict."
    sed 's/\ \#.*//' $LJSPEECH_BASE/cmudict.dict > $LJSPEECH_BASE/uncommented_cmudict.dict
  fi
  G2P_DICT=$LJSPEECH_BASE/uncommented_cmudict.dict
fi

# Run alignment
echo "Starting MFA with dictionary at: $G2P_DICT"

read -r -d '' MFA_ERROR_MSG << EOM
Could not run MFA. If it could not find OpenBlas, you may need to install it manually.
(e.g. sudo apt-get install libopenblas-dev)
EOM

if ! mfa download acoustic english; then
  echo $MFA_ERROR_MSG
fi
mfa align --clean $LJSPEECH_BASE $G2P_DICT english $LJSPEECH_BASE/alignments

# Create JSON mappings from word to phonemes and phonemes to indices
echo "Creating word->phone and phone->idx mappings at $LJSPEECH_BASE/mappings.json..."
python create_token2idx_dict.py \
  --dictionary=$G2P_DICT \
  --dict_out=$LJSPEECH_BASE/mappings.json

# Calculate phoneme durations
echo "Calculating phoneme durations..."
python calculate_durs.py \
  --ljspeech_dir=$LJSPEECH_BASE \
  --mappings=$LJSPEECH_BASE/mappings.json \
  --sr=$SAMPLE_RATE \
  --hop_length=$WINDOW_STRIDE
echo "Phoneme durations and tokens written."
