#!/usr/bin/env bash
set -e

INSTALL_OPTION=${1:-"dev"}


PIP=pip

echo 'Uninstalling stuff'
${PIP} uninstall -y nemo_toolkit
${PIP} uninstall -y sacrebleu

# Kept for legacy purposes
${PIP} uninstall -y nemo_asr
${PIP} uninstall -y nemo_nlp
${PIP} uninstall -y nemo_tts
${PIP} uninstall -y nemo_simple_gan
${PIP} uninstall -y nemo_cv

${PIP} install -U setuptools

echo 'Installing nemo'
if [[ "$INSTALL_OPTION" == "dev" ]]; then
    ${PIP} install --editable ".[all]"
else
    rm -rf dist/
    python setup.py bdist_wheel
    DIST_FILE=$(find ./dist -name "*.whl" | head -n 1)
    ${PIP} install "${DIST_FILE}[all]"
fi

echo 'All done!'
