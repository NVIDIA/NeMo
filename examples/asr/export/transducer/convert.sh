# export LD_LIBRARY_PATH="/home/dgalvez/scratch/code/asr/alexa/work/conformer/old_trts/TensorRT-9.0.0.2/lib:/home/dgalvez/scratch/code/asr/alexa/work/conformer/trt-dbg-9.0.0.1/cudnn/lib64:$LD_LIBRARY_PATH"
# export PATH="/home/dgalvez/scratch/code/asr/alexa/work/conformer/old_trts/TensorRT-9.0.0.2/bin:$PATH"

export LD_LIBRARY_PATH="/home/dgalvez/scratch/code/asr/alexa/work/conformer/old_trts/TensorRT-9.0.1.4/lib:/home/dgalvez/scratch/code/asr/alexa/work/conformer/trt-dbg-9.0.0.1/cudnn/lib64:$LD_LIBRARY_PATH"
export PATH="/home/dgalvez/scratch/code/asr/alexa/work/conformer/old_trts/TensorRT-9.0.1.4/bin:$PATH"

export CUDA_MODULE_LOADING=LAZY

# change max shape from 2048 to 4096 to accomodate Librispeech. Though we should probably infer on chunks instead...

trtexec --onnx=encoder-temp_rnnt.onnx --saveEngine=encoder_fp16.trt \
        --verbose --profilingVerbosity=detailed \
        --fp16 --avgRuns=100 \
        --minShapes=audio_signal:1x80x25,length:1 \
        --optShapes=audio_signal:16x80x1024,length:16 \
        --maxShapes=audio_signal:16x80x4096,length:16 2>&1 > encoder_fp16.perf

trtexec --onnx=decoder_joint-temp_rnnt.onnx --saveEngine=decoder_fp16.trt \
        --verbose --profilingVerbosity=detailed \
        --avgRuns=100 --fp16 \
        --minShapes=encoder_outputs:1x512x1,targets:1x1,target_length:1,input_states_1:1x1x640,input_states_2:1x1x640 \
        --optShapes=encoder_outputs:16x512x1,targets:16x1,target_length:16,input_states_1:1x16x640,input_states_2:1x16x640 \
        --maxShapes=encoder_outputs:16x512x1,targets:16x1,target_length:16,input_states_1:1x16x640,input_states_2:1x16x640 2>&1 > decoder_fp16.perf

