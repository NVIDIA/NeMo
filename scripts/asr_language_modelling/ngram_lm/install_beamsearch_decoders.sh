git clone https://github.com/NVIDIA/OpenSeq2Seq -b ctc-decoders
mv OpenSeq2Seq/decoders .
rm -rf OpenSeq2Seq
cd decoders
./setup.sh
cd ..
