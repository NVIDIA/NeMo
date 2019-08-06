#!/bin/sh
set -xe
git clone https://github.com/PaddlePaddle/DeepSpeech 
cd DeepSpeech
git checkout a76fc69
cd ..
mv DeepSpeech/decoders/swig_wrapper.py DeepSpeech/decoders/swig/ctc_decoders.py
mv DeepSpeech/decoders/swig ./decoders
rm -rf DeepSpeech
cd decoders
sed -i "s/\.decode('utf-8')//g" ctc_decoders.py
sed -i 's/\.decode("utf-8")//g' ctc_decoders.py
sed -i "s/name='swig_decoders'/name='ctc_decoders'/g" setup.py
sed -i "s/-space_prefixes\[i\]->approx_ctc/space_prefixes\[i\]->score/g" decoder_utils.cpp
sed -i "s/py_modules=\['swig_decoders'\]/py_modules=\['ctc_decoders', 'swig_decoders'\]/g" setup.py
chmod +x setup.sh
./setup.sh
echo 'Installing kenlm'
cd kenlm
mkdir build
cd build
cmake ..
make -j
cd ..
cd ..
