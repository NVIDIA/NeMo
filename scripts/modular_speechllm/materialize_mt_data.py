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

# python materialize_mt_data.py /media/data/datasets/LibriSpeech/en /media/data/datasets/LibriSpeech/zh /media/data/datasets/LibriSpeech/question_en_zh /tmp/t.manifest
import json
import random
import sys

en_data = open(sys.argv[1], 'r').readlines()
zh_data = open(sys.argv[2], 'r').readlines()
question_data = open(sys.argv[3], 'r').readlines()
content = ""
for en, zh in zip(en_data, zh_data):
    en = en.strip()
    zh = zh.strip()
    question = random.choice(question_data).strip()
    record = {}
    record['question'] = f"{question}: {en}\n"
    record['text'] = f"{zh}"
    content += json.dumps(record)
    content += "\n"
final = open(sys.argv[4], "w")
final.write(content)
final.close()
