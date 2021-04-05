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
python pynini_export.py .
cd classify; thraxmakedep tokenize_and_classify.grm ; make; cd ..
cd verbalize; thraxmakedep verbalize.grm ; make; cd ..
mv classify/tokenize_and_classify.far .
mv verbalize/verbalize.far .
rm -rf classify/*.far verbalize/*.far util.far Makefile classify/Makefile verbalize/Makefile