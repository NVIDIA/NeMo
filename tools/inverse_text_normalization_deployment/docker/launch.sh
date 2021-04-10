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

#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
: ${CLASSIFY_DIR:="$SCRIPT_DIR/../classify"}
: ${VERBALIZE_DIR:="$SCRIPT_DIR/../verbalize"}
: ${CMD:=${1:-"/bin/bash"}}

MOUNTS=""
MOUNTS+=" -v $CLASSIFY_DIR:/workspace/sparrowhawk/documentation/grammars/en_toy/classify"
MOUNTS+=" -v $VERBALIZE_DIR:/workspace/sparrowhawk/documentation/grammars/en_toy/verbalize"

echo $MOUNTS
docker run -it --rm \
  --shm-size=4g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  $MOUNTS \
  -w /workspace/sparrowhawk/documentation/grammars \
  sparrowhawk:latest $CMD