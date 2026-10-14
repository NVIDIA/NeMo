# Build a Transformer encoder with Torch-TensorRT

[build_torch_trt_encoder.py](build_torch_trt_encoder.py) converts a NeMo Transformer encoder directly into a TensorRT engine file.
It traces the encoder with `torch.export`, then uses Torch-TensorRT to compile the graph, without writing an ONNX file.

Install compatible versions of PyTorch, Torch-TensorRT and TensorRT in an isolated environment.
The tested combination is PyTorch 2.14, Torch-TensorRT 2.14 and TensorRT 11.1 with CUDA 13.
Build the engine from your checkpoint:

```bash
python examples/asr/export/build_torch_trt_encoder.py \
    --model model.nemo --engine encoder.plan
```

The builder uses FP32 with TF32 disabled.
It selects the encoder's SDPA export path and requires TensorRT to support the complete graph.
Conversion fails if an operation is unsupported.

By default, the engine accepts batches of 1–16 items with 16–4000 padded feature frames.
Use `--help` to see the options for changing these limits.
With feature stacking, the minimum padded frame count must be at least twice the stacking factor.
This lets tracing preserve variable sequence lengths; individual true lengths may still be smaller or zero.

The engine inputs are `audio_signal` and `length`; its outputs are `outputs` and `encoded_lengths`.
Running the engine requires TensorRT, but does not require Torch-TensorRT or ONNX conversion libraries.
Preprocessing, decoding and tokenization remain separate steps, and streaming caches are not supported.

Build for the TensorRT version and GPU environment you will use for inference.
When comparing this route with ONNX conversion, use the same weights, precision, TensorRT version and input shape limits.
