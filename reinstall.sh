#!/usr/bin/env bash
set -e

INSTALL_OPTION=${1:-"dev"}

PIP=pip

${PIP} install -U ${PIP}

echo 'Uninstalling stuff'
${PIP} uninstall -y nemo_toolkit
${PIP} uninstall -y sacrebleu

# Kept for legacy purposes
${PIP} uninstall -y nemo_asr
${PIP} uninstall -y nemo_nlp
${PIP} uninstall -y nemo_tts

if [ -n "${NVIDIA_PYTORCH_VERSION}" ]; then
  echo 'Installing NeMo in NVIDIA PyTorch container:' "${NVIDIA_PYTORCH_VERSION}" 'so will not install numba'
else
  if [ -n "${CONDA_PREFIX}" ]; then
    NUMBA_VERSION=0.57.1
    echo 'Installing numba=='${NUMBA_VERSION}
    conda install -y -c conda-forge numba==${NUMBA_VERSION}
  fi
fi

export MAMBA_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
${PIP} install \
  "transformer-engine @ git+https://github.com/NVIDIA/TransformerEngine.git@${TE_TAG}" \
  "apex @ git+https://github.com/NVIDIA/apex.git@${APEX_TAG}" \
  "git+https://github.com/Dao-AILab/causal-conv1d.git@${CAUSAL_CONV_TAG}" \
  "git+https://github.com/state-spaces/mamba.git@${MAMBA_TAG}"

echo 'Installing nemo'
if [[ "$INSTALL_OPTION" == "dev" ]]; then
  ${PIP} install --editable ".[all]"
else
  rm -rf dist/
  ${PIP} install build pytest-runner
  python -m build --no-isolation --wheel
  DIST_FILE=$(find ./dist -name "*.whl" | head -n 1)
  ${PIP} install "${DIST_FILE}[all]"
fi

echo 'All done!'
