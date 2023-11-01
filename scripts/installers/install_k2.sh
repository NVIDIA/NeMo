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

K2_REPO=https://github.com/k2-fsa/k2
LATEST_RELEASE=d12eec7 # Temporary fix for PyTorch 2.1.0
# uncomment the following line after the next k2 version is released (>1.24.4)
#LATEST_RELEASE=$(git -c 'versionsort.suffix=-' \
#    ls-remote --exit-code --refs --sort='version:refname' --tags ${K2_REPO} '*.*' \
#    | tail --lines=1 \
#    | cut -d '/' -f 3)
# "cut --delimiter '/' --fields 3" doesn't work on macOS, use "-d ... -f ..." instead

GRAPHVIZ_REPO=https://gitlab.com/graphviz/graphviz.git
GRAPHVIZ_LATEST_RELEASE=9.0.0
GRAPHVIZ_PY_REPO=https://github.com/xflr6/graphviz
GRAPHVIZ_PY_LATEST_RELEASE=448d1a0  # Temporary fix until 0.20.2

K2_MAKE_ARGS="-j" pip install -v "git+${K2_REPO}@${K2_LATEST_RELEASE}#egg=k2" || { echo "k2 could not be installed!"; exit 1; }
python3 -m k2.version > /dev/null || { echo "k2 installed with errors! Please check installation manually."; exit 1; }
echo "k2 installed successfully!"

# reinstalling graphviz
{
    apt-get update
    apt-get remove -y graphviz && pip uninstall -y graphviz
    apt-get install -y libtool libltdl-dev automake autoconf bison flex tcl \
        libperl-dev python-dev-is-python3 libpython3-dev dh-python ghostscript libgd-dev fontconfig \
        libcairo2-dev libpango1.0-dev libgts-dev
    git clone ${GRAPHVIZ_REPO} -b ${GRAPHVIZ_LATEST_RELEASE} && cd graphviz
    ./autogen.sh && ./configure --disable-python2 --enable-python3
    make -j && make install
    cd .. && rm -rf graphviz
    pip install -v "git+${GRAPHVIZ_PY_REPO}@${GRAPHVIZ_PY_LATEST_RELEASE}#egg=graphviz"
} || { echo "graphviz installed with errors! Please check installation manually."; exit 1; }
echo "graphviz reinstalled successfully!"
