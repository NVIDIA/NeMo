!/usr/bin/env bash

docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -it --rm -v /home/subhankarg/Projects/NeMo_PR/NeMo:/workspace/NeMo -v /mnt/sda/:/data/speech/ --shm-size=8g \
-p 8887:8887 -p 6007:6007 --ulimit memlock=-1 --ulimit \
stack=67108864 --device=/dev/snd nvcr.io/nvidia/pytorch:22.09-py3


# docker run --runtime=nvidia --gpus 0 -e NVIDIA_VISIBLE_DEVICES=0 -it --rm -v /mnt/sda/:/data/speech/ --shm-size=8g \
# -p 8885:8885 -p 6007:6007 --ulimit memlock=-1 --ulimit \
# stack=67108864 --device=/dev/snd nvcr.io/nvidian/swdl/fastpitch_gst:22.09
