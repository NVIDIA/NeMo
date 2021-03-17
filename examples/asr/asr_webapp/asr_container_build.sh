#!/bin/bash

# build container
DOCKER_BUILDKIT=1 docker build -f Dockerfile --progress=plain -t nemo-asr-service:latest .

