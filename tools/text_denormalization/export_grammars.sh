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

# This script compiles and exports WFST-grammars from nemo_tools inverse text normalization, builds C++ production backend Sparrowhawk (https://github.com/google/sparrowhawk) in docker, 
# pluggs grammars into Sparrowhawk and returns prompt inside docker.
# To run inverse text normalization, run e.g.
#       echo "two dollars fifty" | ../../src/bin/normalizer_main --config=sparrowhawk_configuration.ascii_proto
python pynini_export.py .
cd classify && thraxmakedep tokenize_and_classify.grm ; make; cd ..
cd verbalize && thraxmakedep verbalize.grm ; make; cd ..
#rm -rf classify/tokenize_and_classify_tmp.far classify/puntuation.far verbalize/verbalize_tmp.far util.far Makefile classify/Makefile verbalize/Makefile
cp classify/tokenize_and_classify.far /home/ebakhturina/misc_scripts/normalization/debug_denorm/.
cp verbalize/verbalize.far /home/ebakhturina/misc_scripts/normalization/debug_denorm/.

docker run -it -v /home/ebakhturina/misc_scripts/normalization/debug_denorm:/grammars gitlab-master.nvidia.com:5005/dl/sw-ai-app-design/jarvis-api/denorm-jarvis-api
#bash docker/build.sh
#bash docker/launch.sh