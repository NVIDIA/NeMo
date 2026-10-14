# Direct Transformer encoder conversion

`build_torch_trt_encoder.py` captures a NeMo Transformer encoder with `torch.export` and converts the complete graph directly to a standalone TensorRT plan.
It uses the encoder's portable SDPA export implementation and does not write an ONNX model or allow PyTorch fallback.
The initial implementation builds FP32 engines with TF32 disabled.

```bash
python examples/asr/export/build_torch_trt_encoder.py \
  --model model.nemo --engine encoder.plan
```

The tested development environment uses PyTorch 2.14, Torch-TensorRT 2.14 and TensorRT 11.1 with CUDA 13.
Install mutually compatible versions in an isolated environment; plans from another TensorRT version may need rebuilding.
The plan uses the usual NeMo encoder bindings: `audio_signal`, `length`, `outputs` and `encoded_lengths`.
Runtime execution requires TensorRT, without Torch-TensorRT or ONNX conversion libraries.

CLI arguments define the dynamic batch and feature-frame profile.
The defaults cover batches 1–16 and 16–4000 padded feature frames.
For feature stacking, keep the minimum padded frame count at least twice the stacking factor; true lengths may still be smaller or zero.
This avoids symbolic-shape specializations for a single output frame during capture.
Preprocessing, autoregressive decoding, tokenization and streaming caches are outside this encoder plan.

Validation must compare this route with ONNX under the same TensorRT version, precision, weights and shape profile before choosing a deployment path.
