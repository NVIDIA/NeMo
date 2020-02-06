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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:19.11-py3

FROM ${FROM_IMAGE_NAME}

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y libsndfile1 sox \
    python-setuptools python-dev && rm -rf /var/lib/apt/lists/*

ENV PATH=$PATH:/usr/src/tensorrt/bin
WORKDIR /tmp/onnx-trt
COPY scripts/docker/onnx-trt.patch .
RUN git clone -n https://github.com/onnx/onnx-tensorrt.git && cd onnx-tensorrt && git checkout 8716c9b && git submodule update --init --recursive && patch -f < ../onnx-trt.patch && \
    mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DGPU_ARCHS="60 70 75" && make -j16 && make install && mv -f /usr/lib/libnvonnx* /usr/lib/x86_64-linux-gnu/ && ldconfig

WORKDIR /workspace/nemo

COPY requirements/requirements_docker.txt requirements.txt
RUN pip install --disable-pip-version-check -U -r requirements.txt

COPY . .
RUN pip install ".[all]"
RUN printf "#!/bin/bash\njupyter lab --no-browser --allow-root --ip=0.0.0.0" >> start-jupyter.sh && \
    chmod +x start-jupyter.sh

WORKDIR /workspace/nemo




