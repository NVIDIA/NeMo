trtexec --onnx=encoder-temp_rnnt.onnx --saveEngine=encoder_fp16.trt \
        --verbose --profilingVerbosity=detailed \
        --fp16 --avgRuns=100 \
        --minShapes=audio_signal:1x80x25,length:1 \
        --optShapes=audio_signal:16x80x1024,length:16 \
        --maxShapes=audio_signal:32x80x4096,length:32 \
        --dumpProfile --separateProfileRun \
        --dumpLayerInfo \
        2>&1 > encoder_fp16_layer_info.perf 
