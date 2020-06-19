#!/bin/bash
#set -Eueo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

${DIR}/build_onnxruntime.sh ${DIR}/../packages

echo "======================= Building Dockerfile.jetson ... ======================="

sudo -H DOCKER_BUILDKIT=1 nvidia-docker build -f Dockerfile.jetson -t nemo . || exit 1

echo "======================= Successfully built container with Dockerfile.jetson! ======================="
