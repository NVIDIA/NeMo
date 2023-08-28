# syntax=docker/dockerfile:experimental

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:23.04-py3

# build an image that includes only the nemo dependencies, ensures that dependencies
# are included first for optimal caching, and useful for building a development
# image (by specifying build target as `nemo-deps`)
FROM ${BASE_IMAGE} as nemo-deps

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
# libavdevice-dev rerquired for latest torchaudio
RUN apt-get update && \
  apt-get upgrade -y && \
  apt-get install -y \
  libsndfile1 sox \
  libfreetype6 \
  swig \
  ffmpeg \
  libavdevice-dev && \
  rm -rf /var/lib/apt/lists/*

# GCSfuse components
RUN apt-get update && apt-get install --yes --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
  && echo "deb http://packages.cloud.google.com/apt gcsfuse-focal main" \
    | tee /etc/apt/sources.list.d/gcsfuse.list \
  && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
  && apt-get update \
  && apt-get install --yes gcsfuse \
  && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
  && mkdir -p /gcs/training-data && mkdir -p /gcs/checkpoints

# NFS components (needed if using PD-SSD for shared file-system)
RUN apt-get -y update && apt-get install -y nfs-common

WORKDIR /workspace/

WORKDIR /tmp/
# TODO: Remove once this Apex commit (2/24/23) is included in PyTorch
# container
RUN git clone https://github.com/NVIDIA/apex.git && \
  cd apex && \
  git checkout 57057e2fcf1c084c0fcc818f55c0ff6ea1b24ae2 && \
  pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" \
    --global-option="--cuda_ext" --global-option="--fast_layer_norm" --global-option="--distributed_adam" \
    --global-option="--deprecated_fused_adam" --global-option="--bnp" --global-option="--fast_multihead_attn" \
    --global-option="--xentropy" ./ 

# uninstall stuff from base container
RUN pip3 uninstall -y sacrebleu torchtext

# install nemo dependencies
WORKDIR /tmp/nemo
COPY requirements .
RUN for f in $(ls requirements*.txt); do pip3 install --disable-pip-version-check --no-cache-dir -r $f; done

# Networking tools in order to record tx and rx across NICs
RUN apt-get update && apt-get install -y net-tools gawk bc jq sysstat

# NVIDIA DLProf components
RUN pip install nvidia-pyindex &&\
  pip install nvidia-dlprof[pytorch] &&\
  pip install nvtx &&\
  pip install --upgrade requests yq

RUN pip install --upgrade git+https://github.com/NVIDIA/TransformerEngine.git@stable
