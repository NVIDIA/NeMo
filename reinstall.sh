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

if [ "${NVIDIA_PYTORCH_VERSION}" = "22.01" ]
then
  echo 'Installing NeMo in NVIDIA PyTorch container:' ${NVIDIA_PYTORCH_VERSION} 'so will not install numba'
else
  if [ -x "$(command -v conda)" ]
  then
    NUMBA_VERSION=0.55
    echo 'Installing numba=='${NUMBA_VERSION}
    conda install -y -c conda-forge numba==${NUMBA_VERSION}
  fi
fi

echo 'Installing nemo and nemo_text_processing'
if [[ "$INSTALL_OPTION" == "dev" ]]; then
    ${PIP} install --editable ".[all]"
else
    rm -rf dist/
    ${PIP} install build
    python -m build --no-isolation --wheel
    DIST_FILE=$(find ./dist -name "*.whl" | head -n 1)
    ${PIP} install "${DIST_FILE}[all]"
fi

echo 'Installing additional nemo_text_processing conda dependency'
bash nemo_text_processing/setup.sh > /dev/null 2>&1 && echo "nemo_text_processing installed!" || echo "nemo_text_processing could not be installed!"

if [ -x "$(command -v conda)" ]; then
  # we need at least numba .53, and .54 breaks the PyTorch 21.06 container
  echo 'Installing numba=0.53.1'
  conda install -y -c numba numba=0.53.
  # echo 'Attempting update to numba installation via conda'
  # conda update -c numba numba -y >  /dev/null 2>&1 && echo "Numba updated!" || echo "Numba could not be updated!"
fi

echo 'Installing k2 FSA framework'
bash scripts/speech_recognition/k2/setup.sh && echo "k2 installed!" || echo "k2 could not be installed!"

echo 'All done!'
