#!/bin/bash

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""
This file is adapted from
https://github.com/salesforce/awd-lstm-lm/blob/master/getdata.sh
Copyright by the AWD LSTM authors.
"""
DATA_DIR=$1
echo "- Downloading WikiText-2"

wget --continue -P $DATA_DIR https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
unzip -q $DATA_DIR/wikitext-2-v1.zip -d $DATA_DIR
cd $DATA_DIR/wikitext-2
mv wiki.train.tokens train.txt
sed -i -e "s/<unk>/[UNK]/g" train.txt
mv wiki.valid.tokens valid.txt
sed -i -e "s/<unk>/[UNK]/g" valid.txt
mv wiki.test.tokens test.txt
sed -i -e "s/<unk>/[UNK]/g" test.txt
cd ..
rm wikitext-2-v1.zip

echo "- WikiText-2 saved at $DATA_DIR/wikitext-2"
