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
export TE_TAG=7d576ed25266a17a7b651f2c12e8498f67e0baea
export NEMO_RUN_TAG=34259bd3e752fef94045a9a019e4aaf62bd11ce2
export APEX_TAG=810ffae374a2b9cb4b5c5e28eaeca7d7998fca0c
export CAUSAL_CONV_TAG=v1.2.2.post1
export MAMBA_TAG=v2.2.0
export MCORE_TAG=4dc8977167d71f86bdec47a60a98e85c4cfa0031
export NV_RESILIENCY_EXT_TAG=97aad77609d2e25ed38ac5c99f0c13f93c48464e

${PIP} install \
  "nemo_run@git+https://github.com/NVIDIA/NeMo-Run.git@${NEMO_RUN_TAG}" \
  "transformer-engine @ git+https://github.com/NVIDIA/TransformerEngine.git@${TE_TAG}" \
  "apex @ git+https://github.com/NVIDIA/apex.git@${APEX_TAG}" \
  "git+https://github.com/Dao-AILab/causal-conv1d.git@${CAUSAL_CONV_TAG}" \
  "git+https://github.com/state-spaces/mamba.git@${MAMBA_TAG}" \
  "git+https://github.com/NVIDIA/Megatron-LM.git@{MCORE_TAG}" \
  "git+https://github.com/NVIDIA/nvidia-resiliency-ext.git@{NV_RESILIENCY_EXT_TAG}" \
  "onnxscript @ git+https://github.com/microsoft/onnxscript"

echo 'Installing nemo'
if [[ "$INSTALL_OPTION" == "dev" ]]; then
  ${PIP} install --editable --extra-index-url https://pypi.nvidia.com ".[all]"
else
  rm -rf dist/
  ${PIP} install build pytest-runner
  python -m build --no-isolation --wheel
  DIST_FILE=$(find ./dist -name "*.whl" | head -n 1)
  ${PIP} install "${DIST_FILE}[all]"
fi

echo 'All done!'
