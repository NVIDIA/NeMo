#!/usr/bin/env bash

set -euo pipefail

IMAGE="us-central1-docker.pkg.dev/supercomputer-testing/crankshaw-nemo-stagetest/nemo"

cd ../Megatron-LM
MEGATRON_VER=$(git rev-parse --short HEAD)
cd -

TAG="$(git rev-parse --short HEAD)-$MEGATRON_VER"

IMAGE_FULL=$IMAGE:$TAG

DOCKER_BUILDKIT=1 docker build --build-arg MEGATRON_VER=$MEGATRON_VER -f Dockerfile -t $IMAGE_FULL .

docker push $IMAGE_FULL

echo "New tag: $TAG"
