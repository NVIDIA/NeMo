#!/usr/bin/env bash
set -e
if [ ! -d "language_model" ]; then
  mkdir language_model
fi
cd language_model
if [ ! -f "librispeech-lm-norm.txt.gz" ]; then
  wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz
fi
gzip -d librispeech-lm-norm.txt.gz
# convert all upper case characters to lower case
tr '[:upper:]' '[:lower:]' < librispeech-lm-norm.txt > 6-gram.txt
cd ..
# build a language model
pip install pandas
python build_lm_text.py language_model/6-gram.txt --n 6
