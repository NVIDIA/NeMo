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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:19.08-py3

FROM ${FROM_IMAGE_NAME}

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y libsndfile1 sox \
    python-setuptools python-dev && rm -rf /var/lib/apt/lists/*

# Make sure we have the latest apex
RUN pip uninstall -y apex || :
RUN pip uninstall -y apex && \
    cd /tmp && \
    git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install -v --disable-pip-version-check --no-cache-dir \
        --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && \
    cd .. && rm -rf apex

# Install Jupyterlab, other dependencies
RUN pip install ipython[all] tqdm sox ruamel.yaml && \
    pip install -U jupyterlab

# Assumes we are in the root of the nemo git clone
WORKDIR /workspace/nemo
COPY . .

#RUN pip install --disable-pip-version-check -U -r requirements.txt

RUN cd nemo && \
    python setup.py install && \
    cd ../collections/nemo_asr && \
    python setup.py install && \
    cd ../nemo_nlp && \
    python setup.py install && \
    cd ../nemo_lpr && \
    python setup.py install

RUN printf "#!/bin/bash\njupyter lab --no-browser --allow-root --ip=0.0.0.0" >> start-jupyter.sh && \
    chmod +x start-jupyter.sh

