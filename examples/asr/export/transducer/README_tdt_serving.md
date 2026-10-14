# Serve a TDT decoder with Triton

This example serves complete offline greedy TDT decoding as one Triton model.
Send it encoder outputs and their lengths; it returns token IDs, timestamps and durations.
The prediction network, joint network and decoding loop run in one call, using NeMo's PyTorch implementation and its cached computations.
CUDA graphs are used when available.

## Package and start the model

Run these commands from a Speech checkout containing this example:

```bash
python examples/asr/export/transducer/export_tdt_decoder.py \
    --model model.nemo --output model_repository/tdt_decoder --max-batch-size 16
tritonserver --model-repository=model_repository
```

The Triton Python backend must have access to this Speech checkout, its inference dependencies, and compatible PyTorch and cuda-python installations.
Its backend stub must also match the Python environment.
Installing NeMo in another host environment does not configure the backend; see [Triton's Python environment guide](https://github.com/triton-inference-server/python_backend#managing-python-runtime-and-libraries).

The export creates `config.pbtxt`, `1/model.py` and `1/decoder.pt`.
The artifact contains prediction and joint weights and configuration.
Load only trusted artifacts, because that configuration is used to instantiate NeMo classes.
Parameters use FP32 with TF32 disabled, and the checkpoint's `decoding.greedy.max_symbols` limit is preserved.

## Send encoder outputs

The server accepts these tensors, where `B` is the batch size, `C` the encoder channel count and `T` the padded frame count:

| Input | Type | Shape |
| --- | --- | --- |
| `ENCODER_OUTPUT` | FP32 | `[B, C, T]` |
| `ENCODED_LENGTH` | INT64 | `[B, 1]` |

Lengths give the valid frame count for each utterance, excluding padding.
Triton can batch requests with equal channel and padded frame dimensions.
To batch utterances of different lengths, pad their encoder outputs to common frame counts and retain their true lengths.
Invalid lengths are rejected per request.

The adapter makes one decoder call for each scheduled batch and returns results in the original request order.
It accepts CPU inputs, transferring them to the GPU, and exchanges GPU tensors through DLPack.
Each model instance has its own weights and buffers and handles calls serially.
Additional instances allow concurrent execution but require additional GPU memory.
Warm up the intended batch and frame sizes before measuring latency; CUDA graph setup runs on first use and when capacity grows.

## Read the results

All outputs use INT64.
`TOKEN_LENGTH` has shape `[B, 1]`; `TOKENS`, `TIMESTAMPS` and `TOKEN_DURATIONS` have shape `[B, U]`, where `U` is the padded token count.
Only entries before each token length are valid.
Token and timestamp padding is -1; duration padding is zero.
Empty utterances return zero token lengths and at least one padding column.

Timestamps and durations are measured in encoder frames.
Use the checkpoint's tokenizer to turn token IDs into text.
The package does not include that tokenizer or the encoder.

You can also use the decoder directly:

```python
from nemo.collections.asr.parts.submodules.tdt_decoder import TDTDecoder

decoder = TDTDecoder.load("model_repository/tdt_decoder/1/decoder.pt", device="cuda")
result = decoder(encoded, encoded_lengths)  # Shapes: [B, C, T] and [B].
```

Preprocessing, the encoder and tokenization remain separate stages.
A Triton ensemble or business logic scripting (BLS) pipeline can call this complete decoder once after the encoder, without a server request for every token.
The example uses PyTorch prediction and joint networks; it does not use the separate TensorRT single-step engine.
Beam search, language-model fusion, confidence outputs and streaming state are not supported.

## Check a running server

Run the optional HTTP comparison test against the same packaged artifact:

```bash
NEMO_TDT_TRITON_URL=http://localhost:8000 \
NEMO_TDT_DECODER_ARTIFACT=model_repository/tdt_decoder/1/decoder.pt \
pytest tests/collections/asr/test_tdt_triton.py
```

The test requires CUDA and compares all four outputs with direct decoding.
It covers changing shapes, empty utterances, concurrent requests and recovery after an invalid length.
Set `NEMO_TDT_TRITON_MODEL` if you used another model name.
To check that Triton actually batches requests, configure a nonzero `max_queue_delay_microseconds` in `dynamic_batching` and set `NEMO_TDT_TRITON_REQUIRE_BATCHING=1`.
The test uses an already-running server and leaves it running.
