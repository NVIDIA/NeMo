# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

# python materialize_text_data.py /media/data/datasets/LibriSpeech/en /tmp/t.manifest
import json
import random
import sys

en_data = open(sys.argv[1], 'r').readlines()
content = ""
for en in en_data:
    en = en.strip().lower()
    ens = en.split()
    if len(ens) < 5 or (ens[0] == ens[1] and ens[0] == ens[2]):
        continue
    ens = ens[:64]
    en = " ".join(ens)
    record = {}
    record['text'] = f"{en}"
    content += json.dumps(record)
    content += "\n"
final = open(sys.argv[2], "w")
final.write(content)
final.close()
