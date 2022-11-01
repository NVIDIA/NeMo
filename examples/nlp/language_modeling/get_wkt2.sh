#!/bin/bash

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
