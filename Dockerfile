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

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:20.01-py3

# build an image that includes only the nemo dependencies, ensures that dependencies
# are included first for optimal caching, and useful for building a development
# image (by specifying build target as `nemo-deps`)
FROM ${BASE_IMAGE} as nemo-deps

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
    libsndfile1 sox \
    python-setuptools \
    python-dev && \
    rm -rf /var/lib/apt/lists/*

# build trt open source plugins
ENV PATH=$PATH:/usr/src/tensorrt/bin
WORKDIR /tmp/trt-oss
RUN git clone --recursive --branch release/7.0 https://github.com/NVIDIA/TensorRT.git && cd TensorRT && \
     mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DTRT_LIB_DIR=/usr/lib/x86_64-linux-gnu/ -DTRT_BIN_DIR=`pwd` \
    -DBUILD_PARSERS=OFF -DBUILD_SAMPLES=OFF -DBUILD_PLUGINS=ON -DGPU_ARCHS="70 75" && \
    make -j nvinfer_plugin

# install nemo dependencies
WORKDIR /tmp/nemo
COPY requirements/requirements_docker.txt requirements.txt
RUN pip install --disable-pip-version-check --no-cache-dir -r requirements.txt

# copy nemo source into a scratch image
FROM scratch as nemo-src
COPY . .

# start building the final container
FROM nemo-deps as nemo
ARG NEMO_VERSION
ARG BASE_IMAGE

# copy oss trt plugins
COPY --from=nemo-deps /tmp/trt-oss/TensorRT/build/libnvinfer_plugin.so* /usr/lib/x86_64-linux-gnu/

# Check that NEMO_VERSION is set. Build will fail without this. Expose NEMO and base container
# version information as runtime environment variable for introspection purposes
RUN /usr/bin/test -n "$NEMO_VERSION" && \
    /bin/echo "export NEMO_VERSION=${NEMO_VERSION}" >> /root/.bashrc && \
    /bin/echo "export BASE_IMAGE=${BASE_IMAGE}" >> /root/.bashrc
RUN --mount=from=nemo-src,target=/tmp/nemo cd /tmp/nemo && pip install ".[all]"

# copy scripts/examples/tests into container for end user
WORKDIR /workspace/nemo
COPY scripts /workspace/nemo/scripts
COPY examples /workspace/nemo/examples
COPY tests /workspace/nemo/tests
COPY README.rst LICENSE /workspace/nemo/

RUN printf "#!/bin/bash\njupyter lab --no-browser --allow-root --ip=0.0.0.0" >> start-jupyter.sh && \
    chmod +x start-jupyter.sh

