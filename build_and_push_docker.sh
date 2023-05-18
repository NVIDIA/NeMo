#!/usr/bin/env bash

set -euo pipefail

IMAGE="us-central1-docker.pkg.dev/supercomputer-testing/crankshaw-nemo-stagetest/nemo"
TAG=$(git rev-parse --short HEAD)

IMAGE_FULL=$IMAGE:$TAG

DOCKER_BUILDKIT=1 docker build -f Dockerfile -t $IMAGE_FULL .

docker push $IMAGE_FULL
