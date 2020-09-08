#!/usr/bin/env bash
set -e

PIP=pip

echo 'Uninstalling stuff'
${PIP} uninstall -y nemo_toolkit

# Kept for legacy purposes
${PIP} uninstall -y nemo_asr
${PIP} uninstall -y nemo_nlp
${PIP} uninstall -y nemo_tts
${PIP} uninstall -y nemo_simple_gan

${PIP} install -U setuptools

for f in $(ls requirements/*.txt); do ${PIP} install ${PIP_FLAGS}--disable-pip-version-check --no-cache-dir -r $f; done 

echo 'Installing stuff'
${PIP} install -e ".[all]"

echo 'All done!'
