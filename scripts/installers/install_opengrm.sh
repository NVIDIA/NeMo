# Miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
# Opengrm
/root/miniconda3/bin/conda install -c conda-forge ngram -y
# cd /workspace/nemo/decoders
# wget https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.8.2.tar.gz --no-check-certificate && tar xvzf openfst-1.8.2.tar.gz && cd openfst-1.8.2 && ./configure --enable-grm && make -j4 && make -j4 install && cd ..
# OPENFSTPREFIX=/workspace/nemo/decoders/openfst-1.8.2/src &&  wget https://www.opengrm.org/twiki/pub/GRM/NGramDownload/ngram-1.3.14.tar.gz --no-check-certificate && tar xvzf ngram-1.3.14.tar.gz && cd ngram-1.3.14 && LDFLAGS="-L${OPENFSTPREFIX}/lib" CXXFLAGS="-I${OPENFSTPREFIX}/include" ./configure --prefix ${OPENFSTPREFIX} && make -j4 && make -j4 install && cd ..
