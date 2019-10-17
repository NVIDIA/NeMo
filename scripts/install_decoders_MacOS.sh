#!/bin/sh
# Make sure swig is installed first. On Anaconda do:
# conda install swig
set -xe
brew update
brew install wget
brew install boost
brew install cmake
export CFLAGS="-stdlib=libc++"
export MACOSX_DEPLOYMENT_TARGET=10.14
git clone https://github.com/PaddlePaddle/DeepSpeech
cd DeepSpeech
git checkout a76fc69
cd ..
mv DeepSpeech/decoders/swig_wrapper.py DeepSpeech/decoders/swig/ctc_decoders.py
mv DeepSpeech/decoders/swig ./decoders
rm -rf DeepSpeech
cd decoders
sed -i'.original' -e "s/\.decode('utf-8')//g" ctc_decoders.py
sed -i'.original' -e 's/\.decode("utf-8")//g' ctc_decoders.py
sed -i'.original' -e "s/name='swig_decoders'/name='ctc_decoders'/g" setup.py
sed -i'.original' -e "s/-space_prefixes\[i\]->approx_ctc/space_prefixes\[i\]->score/g" decoder_utils.cpp
sed -i'.original' -e "s/py_modules=\['swig_decoders'\]/py_modules=\['ctc_decoders', 'swig_decoders'\]/g" setup.py
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
