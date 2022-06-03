#!/bin/bash

# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

# Torch and torchaudio versions must match. Othervise, there will be no CUDA support.
# See https://github.com/pytorch/audio/blob/f0bc00c980012badea8db011f84a0e9ef33ba6c1/README.md?plain=1#L66

TORCHAUDIO_REPO=https://github.com/pytorch/audio
# expected LATEST_RELEASE=release/*.**
LATEST_RELEASE=$(git -c 'versionsort.suffix=-' \
    ls-remote --exit-code --refs --sort='version:refname' --heads ${TORCHAUDIO_REPO} 'release/*.*' \
    | tail --lines=1 \
    | cut --delimiter='/' --fields=3,4)
# expected TORCHAUDIO_BUILD_VERSION=*.**.*
TORCHAUDIO_BUILD_VERSION=${LATEST_RELEASE:8:1}${PYTORCH_VERSION:1:5}

git clone --depth 1 --branch ${LATEST_RELEASE} https://github.com/pytorch/audio.git && \
cd audio && \
git submodule update --init --recursive && \
BUILD_SOX=1 BUILD_VERSION=${TORCHAUDIO_BUILD_VERSION} python setup.py install || \
(echo "Failed to install torchaudio!"; exit 1);
# RNNT loss is built with CUDA, so checking it will suffice
cd .. && \
pytest audio/test/torchaudio_unittest/functional/torchscript_consistency_cuda_test.py -k 'test_rnnt_loss' || \
(echo "Failed to install torchaudio with CUDA support!"; exit 1);
rm -rf audio && \
echo "Torchaudio installed successfully!"
