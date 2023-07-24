export POLYGRAPHY_AUTOINSTALL_DEPS=1


polygraphy surgeon extract /home/dgalvez/scratch/code/asr/nemo_conformer_benchmark/NeMo/examples/asr/export/transducer/encoder-temp_rnnt.onnx \
           --inputs length:auto:auto \
           --outputs encoded_lengths:auto \
           -o just_length_computation2.onnx
