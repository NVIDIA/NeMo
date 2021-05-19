# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

# This script compiles and exports WFST-grammars from nemo_text_processing, builds C++ production backend Sparrowhawk (https://github.com/google/sparrowhawk) in docker, 
# plugs grammars into Sparrowhawk and returns prompt inside docker.
# For inverse text normalization run:
#       echo "two dollars fifty" | ../../src/bin/normalizer_main --config=sparrowhawk_configuration.ascii_proto
# For text normalization run:
#       echo "\$2.5" | ../../src/bin/normalizer_main --config=sparrowhawk_configuration.ascii_proto

GRAMMARS=${1:-"itn_grammars"} # tn_grammars
INPUT_CASE=${2:-"cased"} # lower_cased, only for tn_grammars
python pynini_export.py --output_dir=. --grammars=${GRAMMARS} --input_case=${INPUT_CASE}
find . -name "Makefile" -type f -delete
bash docker/build.sh 
bash docker/launch.sh 