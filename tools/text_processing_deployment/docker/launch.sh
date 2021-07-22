# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

# !/bin/bash

MODE=${1:-"export"}
LANGUAGE=${2:-"en"}
SCRIPT_DIR=$(cd $(dirname $0); pwd)
: ${CLASSIFY_DIR:="$SCRIPT_DIR/../$LANGUAGE/classify"}
: ${VERBALIZE_DIR:="$SCRIPT_DIR/../$LANGUAGE/verbalize"}
: ${CMD:=${3:-"/bin/bash"}}

MOUNTS=""
MOUNTS+=" -v $CLASSIFY_DIR:/workspace/sparrowhawk/documentation/grammars/en_toy/classify"
MOUNTS+=" -v $VERBALIZE_DIR:/workspace/sparrowhawk/documentation/grammars/en_toy/verbalize"

WORK_DIR="/workspace/sparrowhawk/documentation/grammars"
if [[ $MODE == "test_tn_grammars" ]]; then
  CMD="bash test_sparrowhawk_normalization.sh"
  WORK_DIR="/workspace/tests/${LANGUAGE}"
elif [[ $MODE == "test_itn_grammars" ]]; then
  CMD="bash test_sparrowhawk_inverse_text_normalization.sh"
  WORK_DIR="/workspace/tests/${LANGUAGE}"
fi

echo $MOUNTS
docker run -it --rm \
  --shm-size=4g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  $MOUNTS \
  -v $SCRIPT_DIR/../../../tests/nemo_text_processing/:/workspace/tests/ \
  -w $WORK_DIR \
  sparrowhawk:latest $CMD