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

${PIP} install --no-cache-dir --no-build-isolation \
  git+https://github.com/NVIDIA/apex.git@${APEX_TAG}
${PIP} install --no-cache-dir \
  git+https://github.com/NVIDIA/nvidia-resiliency-ext.git@97aad77609d2e25ed38ac5c99f0c13f93c48464e \
  git+https://github.com/NVIDIA/TransformerEngine.git@${TE_TAG} \
  git+https://github.com/Dao-AILab/causal-conv1d.git@v1.2.2.post1 \
  git+https://github.com/state-spaces/mamba.git@v2.2.2

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
