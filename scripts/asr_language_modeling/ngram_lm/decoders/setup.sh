#!/usr/bin/env bash

if [ ! -d kenlm ]; then
    git clone https://github.com/kpu/kenlm.git
    echo -e "\n"
fi

if [ ! -d kenlm/build ]; then
    mkdir kenlm/build
    cd kenlm/build
    cmake ..
    make
    cd ../..
    echo -e "\n"
fi

if [ ! -d openfst-1.6.3 ]; then
    echo "Download and extract openfst ..."
    wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.3.tar.gz
    tar -xzvf openfst-1.6.3.tar.gz
    echo -e "\n"
fi

if [ ! -d ThreadPool ]; then
    git clone https://github.com/progschj/ThreadPool.git
    echo -e "\n"
fi

echo "Install decoders ..."
python setup.py install --num_processes 4
echo "Testing installation ..."
python ctc_decoders_test.py
