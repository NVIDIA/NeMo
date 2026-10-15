# Build an encoder with Torch-TensorRT

[build_torch_trt_encoder.py](build_torch_trt_encoder.py) traces a NeMo Transformer encoder with `torch.export` and compiles it into a standalone TensorRT plan, without ONNX.

Install compatible PyTorch, Torch-TensorRT and TensorRT versions.
The tested combination is 2.14, 2.14 and 11.1 respectively, with CUDA 13.

```bash
python examples/asr/export/build_torch_trt_encoder.py \
    --model model.nemo --engine encoder.plan
```

The builder uses the encoder's SDPA export path, FP32 and disabled TF32.
Unsupported operations fail conversion.
Defaults cover batches 1–16 and 16–4000 padded feature frames; see `--help` for options.
With feature stacking, keep the minimum padded frame count at least twice the stacking factor; true lengths may be smaller or zero.

Running the plan requires TensorRT, without Torch-TensorRT.
Its inputs are `audio_signal` and `length`; outputs are `outputs` and `encoded_lengths`.
Preprocessing, decoding and tokenization remain separate; streaming caches are unsupported.
Build for the target GPU/runtime and compare conversion routes with matching weights, precision and shape limits.
