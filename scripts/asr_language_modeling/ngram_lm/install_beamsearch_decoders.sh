#!/usr/bin/env bash
cd /workspace/nemo
apt-get update && apt-get upgrade -y && apt-get install -y liblzma-dev && rm -rf /var/lib/apt/lists/*

git clone https://github.com/NVIDIA/OpenSeq2Seq
cd OpenSeq2Seq
git checkout ctc-decoders
cd ..
mv OpenSeq2Seq/decoders .
rm -rf OpenSeq2Seq
cd decoders
# patch setup code to support the recent distutils
sed -i 's/, distutils/, distutils\nimport distutils.ccompiler/g' setup.py

cp /workspace/nemo/scripts/installers/setup_os2s_decoders.py ./setup.py
./setup.sh

# install Boost package for KenLM
wget https://boostorg.jfrog.io/artifactory/main/release/1.80.0/source/boost_1_80_0.tar.bz2 --no-check-certificate && tar --bzip2 -xf /workspace/nemo/decoders/boost_1_80_0.tar.bz2 && cd boost_1_80_0 && ./bootstrap.sh && ./b2 --layout=tagged link=static,shared threading=multi,single install -j4 || echo FAILURE
export BOOST_ROOT=/workspace/nemo/decoders/boost_1_80_0

# install KenLM
cd /workspace/nemo/decoders/kenlm/build && cmake -DKENLM_MAX_ORDER=10 .. && make -j2
cd /workspace/nemo/decoders/kenlm
python setup.py install --max_order=10
export KENLM_LIB=/workspace/nemo/decoders/kenlm/build/bin
export KENLM_ROOT=/workspace/nemo/decoders/kenlm
cd ..

# install Flashlight
git clone https://github.com/flashlight/text && cd text
python setup.py bdist_wheel
pip install dist/*.whl
cd ..
