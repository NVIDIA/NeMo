# Complete TDT decoding in Triton

This example exposes the entire offline greedy TDT decoder as one Triton model.
An encoder, including a TensorRT encoder, supplies `ENCODER_OUTPUT` with shape `[B, C, T]` and FP32 dtype, and `ENCODED_LENGTH` with shape `[B, 1]` and INT64 dtype.
The component runs prediction, joint evaluation, blank handling, duration advancement and termination locally.
There are no per-token BLS requests.

The runtime reuses NeMo's label-looping implementation, including cached encoder projections, prediction outputs and CUDA graphs when available.
Prediction and joint networks currently execute with PyTorch.
The combined TensorRT single-step engine is not used: substituting it directly would repeat prediction and projection work on blank steps.
A future TensorRT integration should preserve these caches and validate CUDA graph capture.

## Package and run

From a Speech checkout containing this implementation:

```bash
python examples/asr/export/transducer/export_tdt_decoder.py \
  --model model.nemo --output model_repository/tdt_decoder --max-batch-size 16
tritonserver --model-repository=model_repository
```

The server's Python backend environment must contain this Speech checkout, its inference dependencies and CUDA-compatible PyTorch and cuda-python.
The backend stub and Python environment must use compatible Python versions; installing NeMo in an unrelated host environment does not configure the stub.
See the [Python backend environment documentation](https://github.com/triton-inference-server/python_backend#managing-python-runtime-and-libraries).
This example does not install or launch a public service.

Packaging creates `config.pbtxt`, `1/model.py` and `1/decoder.pt`.
The artifact contains prediction/joint weights and configuration, without encoder weights or a tokenizer.
Only load artifacts from a trusted source; module configuration is used to instantiate NeMo classes.
FP32 parameters and disabled TF32 provide the initial numerical reference.
The checkpoint's `decoding.greedy.max_symbols` limit is preserved.
Beam search, language-model fusion, confidence outputs and streaming state are not part of this interface.

The outputs are INT64 `TOKENS`, `TOKEN_LENGTH`, `TIMESTAMPS` and `TOKEN_DURATIONS`.
`TOKEN_LENGTH` has shape `[B, 1]`; the other outputs have shape `[B, U]`.
Only entries before each token length are valid; padding is -1 for tokens and timestamps, and zero for durations.
Timestamps and durations use encoder frames, not seconds.
Empty utterances return zero token lengths and at least one padding column.
Use the checkpoint's tokenizer to produce text.

Triton dynamically groups requests with equal non-batch input shapes.
Pad encoder outputs to common frame buckets to batch unequal utterance lengths, while retaining their true `ENCODED_LENGTH` values.
The adapter combines the scheduled requests into one decoder call and splits the outputs back into their original request order.
Invalid length values are rejected per request before batching.
GPU tensors cross the Python boundary through DLPack; CPU tensors are also accepted and transferred before decoding.
Each model instance has independent weights and decoding buffers.
Calls to an instance are serialized; use additional model instances for concurrency, accounting for their GPU memory.
CUDA graph setup occurs on first use and when batch/time capacity grows; warm up the intended shape buckets before measuring latency.

To use the same component without a server:

```python
from nemo.collections.asr.parts.submodules.tdt_decoder import TDTDecoder

decoder = TDTDecoder.load("model_repository/tdt_decoder/1/decoder.pt", device="cuda")
result = decoder(encoded, encoded_lengths)  # [B, C, T], [B]
```

An ensemble or BLS pipeline can call preprocessing, encoder and this complete decoder as coarse stages.
The example provides the decoder model; assembling and benchmarking a complete waveform-to-text server remains a separate deployment step.

## Live server validation

After starting the packaged model, run the optional HTTP parity test against the same artifact:

```bash
NEMO_TDT_TRITON_URL=http://localhost:8000 \
NEMO_TDT_DECODER_ARTIFACT=model_repository/tdt_decoder/1/decoder.pt \
pytest tests/collections/asr/test_tdt_triton.py
```

The test requires CUDA and compares tokens, lengths, timestamps and durations with direct decoding.
It checks changing shapes, empty utterances, concurrent requests and recovery after an invalid length.
Set `NEMO_TDT_TRITON_MODEL` if the served model has another name.
To require observed request batching, configure a nonzero `max_queue_delay_microseconds` in `dynamic_batching` and set `NEMO_TDT_TRITON_REQUIRE_BATCHING=1`.
The test neither starts nor stops the server.
