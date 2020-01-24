#!/usr/bin/env bash
set -e

echo 'Uninstalling stuff'
pip uninstall -y nemo_toolkit

# Kept for legacy purposes
pip uninstall -y nemo_asr
pip uninstall -y nemo_nlp
pip uninstall -y nemo_tts
pip uninstall -y nemo_simple_gan

echo 'Installing stuff'
pip install -e ".[all]"

echo 'All done!'
