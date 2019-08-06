"""
This file is adapted from
https://github.com/salesforce/awd-lstm-lm/blob/master/getdata.sh
Copyright by the AWD LSTM authors.
"""

echo "- Downloading WikiText-2"

wget --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
unzip -q wikitext-2-v1.zip
cd wikitext-2
mv wiki.train.tokens train.txt
sed -i -e "s/<unk>/[UNK]/g" train.txt
mv wiki.valid.tokens valid.txt
sed -i -e "s/<unk>/[UNK]/g" valid.txt
mv wiki.test.tokens test.txt
sed -i -e "s/<unk>/[UNK]/g" test.txt
cd ..
rm wikitext-2-v1.zip
