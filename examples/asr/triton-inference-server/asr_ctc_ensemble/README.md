convert nemo ctc model to onnx:
```
mkdir onnx_model/
python3 NeMo/examples/asr/triton-inference-server/scripts/export_asr_ctc_onnx.py --nemo_model=parakeet-ctc-1.1b/parakeet-ctc-1.1b.nemo --onnx_model=onnx_model/model.onnx

# OR

python3 NeMo/examples/asr/triton-inference-server/scripts/export_asr_ctc_onnx.py --pretrained_model=parakeet-ctc-1.1b/parakeet-ctc-1.1b.nemo --onnx_model=onnx_model/model.onnx

```

*Don't use `NeMo/scripts/export.py` to export the model to onnx.*

Convert onnx model to tensorrt:
```
docker run --gpus all -it -v $PWD:/ws nvcr.io/nvidia/tensorrt:24.02-py3
trtexec --onnx=onnx_model/model.onnx \
        --minShapes=audio_signal:1x80x100,length:1 \
        --optShapes=audio_signal:16x80x1000,length:16 \
        --maxShapes=audio_signal:32x80x60000,length:32
                                     
```
The input is from 1 seconds to 600 seconds (10min).
