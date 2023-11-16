#!/bin/bash

set -euo pipefail

python -c "from nemo.collections.asr.models import ASRModel; nemo_model = ASRModel.from_pretrained('stt_en_conformer_transducer_large', map_location='cuda'); nemo_model.freeze(); nemo_model.export('temp_rnnt.onnx', onnx_opset_version=18)"

trtexec --onnx=encoder-temp_rnnt.onnx --saveEngine=encoder-temp_rnnt.trt \
        --verbose --profilingVerbosity=detailed \
        --fp16 --avgRuns=100 \
        --minShapes=audio_signal:1x80x25,length:1 \
        --optShapes=audio_signal:16x80x1024,length:16 \
        --maxShapes=audio_signal:32x80x4096,length:32 \
        --dumpProfile --separateProfileRun \
        --dumpLayerInfo \
        2>&1 > encoder-temp_rnnt.perf

nsys profile -s none --cpuctxsw=none         -c cudaProfilerApi  -f true -o encoder-temp_rnnt.nsys-rep trtexec --loadEngine=encoder-temp_rnnt.trt \
     --shapes=audio_signal:16x80x1024,length:16
