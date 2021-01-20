#!/bin/bash

TRG="gitlab-master.nvidia.com:5005/fkreuk/uniglow"
VER=$(git rev-parse --abbrev-ref HEAD); 

echo "pushing to:"
echo "repo: $TRG"
echo "version: $VER"
echo "----------------------------"

DOCKER_BUILDKIT=1 \
nvidia-docker build \
  --network=host \
  --build-arg CACHE_VER=$(date +%Y%m%d-%H%M%S) \
  --tag $TRG:$VER .

nvidia-docker push $TRG:$VER
