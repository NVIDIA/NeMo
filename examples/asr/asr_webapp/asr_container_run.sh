#!/bin/bash

# run container
docker run -d --gpus=all -p 8000:8000 nemo-asr-service:latest

