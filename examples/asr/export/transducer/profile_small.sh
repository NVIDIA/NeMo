#!/bin/bash

set -euo pipefail

polygraphy surgeon extract /home/dgalvez/scratch/code/asr/nemo_conformer_benchmark/NeMo/examples/asr/export/transducer/encoder-temp_rnnt.onnx --inputs audio_signal:auto:auto length:auto:auto --outputs /layers.0/norm_out/LayerNormalization_output_0:auto -o encoder_subgraph.onnx

trtexec --onnx=encoder_subgraph.onnx --saveEngine=encoder_subgraph.trt \
        --verbose --profilingVerbosity=detailed \
        --fp16 --avgRuns=100 \
        --minShapes=audio_signal:1x80x25,length:1 \
        --optShapes=audio_signal:16x80x1024,length:16 \
        --maxShapes=audio_signal:32x80x4096,length:32 \
        --dumpProfile --separateProfileRun \
        --dumpLayerInfo \
        2>&1 > encoder_subgraph.perf

nsys profile -s none --cpuctxsw=none         -c cudaProfilerApi  -f true -o encoder_subgraph.nsys-rep trtexec --loadEngine=encoder_subgraph.trt \
     --shapes=audio_signal:16x80x1024,length:16

