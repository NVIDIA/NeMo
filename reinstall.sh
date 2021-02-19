#!/usr/bin/env bash
set -e

INSTALL_OPTION=${1:-"development"}

if [[ "$INSTALL_OPTION" == "development" ]]; then
    PIP_INSTALL_OPTION="--editable" # for development mode, if unset release mode
else
    PIP_INSTALL_OPTION=""
fi

PIP=pip

echo 'Uninstalling stuff'
${PIP} uninstall -y nemo_toolkit
${PIP} uninstall -y sacrebleu

# Kept for legacy purposes
${PIP} uninstall -y nemo_asr
${PIP} uninstall -y nemo_nlp
${PIP} uninstall -y nemo_tts
${PIP} uninstall -y nemo_simple_gan

${PIP} install -U setuptools

echo 'Installing stuff'
${PIP} install ${PIP_INSTALL_OPTION} ".[all]"

echo 'All done!'
