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

ARG BASE_IMAGE=us-central1-docker.pkg.dev/supercomputer-testing/redrock-dev/nemo-deps:20231011-205139

# build an image that includes only the nemo dependencies, ensures that dependencies
# are included first for optimal caching, and useful for building a development
# image (by specifying build target as `nemo-deps`)
FROM ${BASE_IMAGE} as nemo-deps

ARG MEGATRON_VER=unknown
RUN git clone https://github.com/crankshaw-google/Megatron-LM.git \
  && cd Megatron-LM \
  && git checkout $MEGATRON_VER \
  && pip install .


# copy nemo source into a scratch image
FROM scratch as nemo-src
COPY . .

# start building the final container
FROM nemo-deps as nemo
ARG NEMO_VERSION=1.19.0

# Check that NEMO_VERSION is set. Build will fail without this. Expose NEMO and base container
# version information as runtime environment variable for introspection purposes
RUN /usr/bin/test -n "$NEMO_VERSION" && \
  /bin/echo "export NEMO_VERSION=${NEMO_VERSION}" >> /root/.bashrc && \
  /bin/echo "export BASE_IMAGE=${BASE_IMAGE}" >> /root/.bashrc

# Install NeMo
RUN --mount=from=nemo-src,target=/tmp/nemo cd /tmp/nemo && pip install ".[all]"

# Check install
RUN python -c "import nemo.collections.nlp as nemo_nlp" && \
  python -c "import nemo.collections.tts as nemo_tts" && \
  python -c "import nemo_text_processing.text_normalization as text_normalization"



# copy scripts/examples/tests into container for end user
WORKDIR /workspace/nemo
RUN wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json \
  && wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

COPY scripts /workspace/nemo/scripts
COPY examples /workspace/nemo/examples
COPY LICENSE /workspace/nemo/

RUN printf "#!/bin/bash\njupyter lab --no-browser --allow-root --ip=0.0.0.0" >> start-jupyter.sh && \
  chmod +x start-jupyter.sh

