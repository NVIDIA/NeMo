#!/usr/bin/env bash

set -euo pipefail


IMAGE="us-central1-docker.pkg.dev/supercomputer-testing/redrock-dev/nemo-deps"



TAG=$(date +%Y%m%d-%H%M%S)
FULL="${IMAGE}:${TAG}"


DOCKER_BUILDKIT=1 docker build -f NemoDeps.Dockerfile -t $FULL .

echo $FULL
docker push $FULL


echo "New Deps tag: $TAG"
