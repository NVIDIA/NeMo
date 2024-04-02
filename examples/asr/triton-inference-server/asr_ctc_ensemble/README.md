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
        --maxShapes=audio_signal:32x80x6000,length:32 \
        --fp16 \
        --saveEngine=encoder.trt
                                     
```
The input is from 1 seconds to 60 seconds (1min).

V100 trtexec log:
```
[04/01/2024-21:10:48] [I] === Performance summary ===
[04/01/2024-21:10:48] [I] Throughput: 9.56737 qps
[04/01/2024-21:10:48] [I] Latency: min = 104.92 ms, max = 106.878 ms, mean = 105.678 ms, median = 105.593 ms, percentile(90%) = 106.558 ms, percentile(95%) = 106.747 ms, percentile(99%) = 106.878 ms
[04/01/2024-21:10:48] [I] Enqueue Time: min = 103.742 ms, max = 105.498 ms, mean = 104.401 ms, median = 104.297 ms, percentile(90%) = 105.173 ms, percentile(95%) = 105.428 ms, percentile(99%) = 105.498 ms
[04/01/2024-21:10:48] [I] H2D Latency: min = 0.438232 ms, max = 0.87085 ms, mean = 0.527107 ms, median = 0.458984 ms, percentile(90%) = 0.635376 ms, percentile(95%) = 0.685303 ms, percentile(99%) = 0.87085 ms
[04/01/2024-21:10:48] [I] GPU Compute Time: min = 103.806 ms, max = 105.511 ms, mean = 104.457 ms, median = 104.337 ms, percentile(90%) = 105.17 ms, percentile(95%) = 105.42 ms, percentile(99%) = 105.511 ms
[04/01/2024-21:10:48] [I] D2H Latency: min = 0.629639 ms, max = 1.24341 ms, mean = 0.693772 ms, median = 0.668213 ms, percentile(90%) = 0.69986 ms, percentile(95%) = 0.722656 ms, percentile(99%) = 1.24341 ms
[04/01/2024-21:10:48] [I] Total Host Walltime: 3.24018 s
[04/01/2024-21:10:48] [I] Total GPU Compute Time: 3.23816 s
[04/01/2024-21:10:48] [W] * Throughput may be bound by Enqueue Time rather than GPU Compute and the GPU may be under-utilized.
[04/01/2024-21:10:48] [W]   If not already in use, --useCudaGraph (utilize CUDA graphs where possible) may increase the throughput.
[04/01/2024-21:10:48] [I] Explanations of the performance metrics are printed in the verbose logs.
[04/01/2024-21:10:48] [I]
&&&& PASSED TensorRT.trtexec [TensorRT v8603] # trtexec --onnx=model.onnx --minShapes=audio_signal:1x80x100,length:1 --optShapes=audio_signal:16x80x1000,length:16 --maxShapes=audio_signal:32x80x6000,length:32 --fp16 --saveEngine=../encoder.trt
```

```
mv encoder.trt encoder/1/
```