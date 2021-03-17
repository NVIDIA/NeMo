#!/bin/bash

# run container
docker run -d --gpus=all -p 8000:8000 --shm-size=8g --ulimit memlock=-1 --ulimit \
  stack=67108864 nemo-asr-service:latest

