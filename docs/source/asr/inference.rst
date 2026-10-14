.. _asr-inference:

=========
Inference
=========

This page covers how to load ASR models and run inference in NeMo.


Loading Checkpoints
-------------------

**From a local file:**

.. code-block:: python

    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.restore_from("path/to/checkpoint.nemo")

**From HuggingFace:**

.. code-block:: python

    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")


Basic Transcription
-------------------

**Python API:**

.. code-block:: python

    outputs = model.transcribe(audio=["file1.wav", "file2.wav"], batch_size=2)
    print(outputs[0].text)

The ``audio`` argument accepts file paths (strings), lists of paths, numpy arrays, or PyTorch tensors.
Audio must be 16 kHz mono-channel.

**Numpy/Tensor inputs:**

.. code-block:: python

    import soundfile as sf
    audio, sr = sf.read("audio.wav", dtype='float32')
    outputs = model.transcribe([audio], batch_size=1)

**Command line:**

.. code-block:: bash

    python examples/asr/transcribe_speech.py \
        pretrained_name="nvidia/parakeet-tdt-0.6b-v2" \
        audio_dir=<path_to_audio_dir>

**Batch generator (for incremental processing):**

``model.transcribe()`` already handles large file lists internally via batching. Use ``transcribe_generator`` only when you need to process results incrementally (e.g., writing to disk batch-by-batch to avoid holding all outputs in memory):

.. code-block:: python

    config = model.get_transcribe_config()
    config.batch_size = 32
    for batch_outputs in model.transcribe_generator(audio_files, override_config=config):
        # write batch results to disk immediately
        ...

``TranscribeConfig`` fields:

.. list-table::
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``batch_size``
     - 4
     - Batch size for inference. Larger = better throughput but more memory.
   * - ``return_hypotheses``
     - False
     - If True, return ``Hypothesis`` objects (with timestamps, scores) instead of plain strings.
   * - ``use_lhotse``
     - True
     - Use Lhotse dataloading for inference.
   * - ``num_workers``
     - None
     - Number of DataLoader workers.
   * - ``channel_selector``
     - None
     - Select channel(s) from multi-channel audio. ``'average'`` to average across channels.
   * - ``timestamps``
     - None
     - If True, return word/segment timestamps (requires ``return_hypotheses=True``).
   * - ``augmentor``
     - None
     - Optional augmentation config to apply during transcription.
   * - ``verbose``
     - True
     - Show progress bar.

For multi-task models (Canary), ``MultiTaskTranscriptionConfig`` extends this with ``prompt``, ``text_field``, ``lang_field``, and ``enable_chunking``.

**Alignments:**

.. code-block:: python

    hyps = model.transcribe(audio=["file.wav"], return_hypotheses=True)
    alignments = hyps[0].alignments


Timestamps
----------

Obtain word, segment, or character timestamps with Parakeet models (CTC/RNNT/TDT):

**Simple usage:**

.. code-block:: python

    hypotheses = model.transcribe(["audio.wav"], timestamps=True)

    for stamp in hypotheses[0].timestamp['word']:
        print(f"{stamp['start']}s - {stamp['end']}s : {stamp['word']}")

    for stamp in hypotheses[0].timestamp['segment']:
        print(f"{stamp['start']}s - {stamp['end']}s : {stamp['segment']}")

**Advanced configuration:**

For Transducer decoding strategies (greedy, beam, TSD, ALSD, mAES) and their parameters, see the Transducer Decoding section in :doc:`Configs <./configs>`.
For CTC and AED decoding classes, see the :doc:`API reference <./api>`.
For decoding customization (confidence, CUDA graphs, language models, word boosting), see :doc:`ASR Language Modeling and Customization <./asr_language_modeling_and_customization>`.

.. code-block:: python

    from omegaconf import open_dict

    decoding_cfg = model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.preserve_alignments = True
        decoding_cfg.compute_timestamps = True
        decoding_cfg.segment_seperators = [".", "?", "!"]
        decoding_cfg.word_seperator = " "
        model.change_decoding_strategy(decoding_cfg)

    hypotheses = model.transcribe(["audio.wav"], return_hypotheses=True)
    timestamp_dict = hypotheses[0].timestamp

    time_stride = 8 * model.cfg.preprocessor.window_stride
    for stamp in timestamp_dict['word']:
        start = stamp['start_offset'] * time_stride
        end = stamp['end_offset'] * time_stride
        word = stamp['char'] if 'char' in stamp else stamp['word']
        print(f"{start:0.2f} - {end:0.2f} : {word}")


Long Audio Inference
--------------------

For audio longer than what fits in memory (especially with Conformer's quadratic attention):

**Buffered / chunked inference:**

Divide audio into overlapping chunks and merge outputs. Scripts are in
`examples/asr/asr_chunked_inference <https://github.com/NVIDIA-NeMo/Speech/tree/main/examples/asr/asr_chunked_inference>`_.

**Local attention (recommended for Fast Conformer):**

Switch to Longformer-style local+global attention for linear-cost inference on audio >1 hour:

.. code-block:: python

    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-ctc-1.1b")
    model.change_attention_model(
        self_attention_model="rel_pos_local_attn",
        att_context_size=[128, 128]
    )

Or via CLI:

.. code-block:: bash

    python examples/asr/speech_to_text_eval.py \
        (...other parameters...) \
        ++model_change.conformer.self_attention_model="rel_pos_local_attn" \
        ++model_change.conformer.att_context_size=[128, 128]

**Subsampling memory optimization:**

For very long files where even the subsampling module runs out of memory:

.. code-block:: python

    model.change_subsampling_conv_chunking_factor(1)  # auto-chunk subsampling


Multi-task Inference (Canary)
-----------------------------

Canary models use prompt slots to control transcription behavior.

**Via manifest:**

.. code-block:: python

    from nemo.collections.asr.models import EncDecMultiTaskModel

    canary = EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b-v2")
    decode_cfg = canary.cfg.decoding
    decode_cfg.beam.beam_size = 1
    canary.change_decoding_strategy(decode_cfg)

    results = canary.transcribe("manifest.json", batch_size=16)

For the manifest format required by Canary models, see :ref:`Canary Manifest Format <canary-manifest-format>`.

**Via direct parameters:**

.. code-block:: python

    results = canary.transcribe(
        audio=["audio.wav"],
        batch_size=4,
        source_lang="en",
        target_lang="en",
        pnc=True,
    )


.. _asr-enforcing-single-language:

Enforcing a Single Language
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For multilingual Canary models, you can enforce a specific output language by explicitly setting ``source_lang`` and
``target_lang``. When both are set to the same language, the model will transcribe in that language only:

.. code-block:: python

    results = canary.transcribe(
        audio=["audio.wav"],
        source_lang="de",
        target_lang="de",
    )

This prevents phonetic drift where the model may switch languages mid-utterance.


Streaming Inference
-------------------

NeMo provides a unified streaming-first Pipeline API for real-time ASR under ``nemo.collections.asr.inference``.
It supports buffered CTC/RNNT/TDT pipelines (overlapping chunks with any offline model) and cache-aware CTC/RNNT pipelines (processes each frame once using cached activations).

See the `Streaming ASR Pipelines tutorial <https://github.com/NVIDIA-NeMo/Speech/blob/main/tutorials/asr/Streaming_ASR_Pipelines.ipynb>`_ for a comprehensive walkthrough covering buffered and cache-aware pipelines, per-stream options, EoU detection, word timestamps, per-stream biasing, ITN, and speech translation.

See :ref:`cache-aware streaming conformer` for model architecture details.


Prompt-conditioned Streaming (Multilingual)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prompt-conditioned cache-aware streaming models
(:class:`~nemo.collections.asr.models.EncDecRNNTBPEModelWithPrompt` and
:class:`~nemo.collections.asr.models.EncDecHybridRNNTCTCBPEModelWithPrompt`) select a language
prompt at inference time via ``target_lang``. The cache-aware simulation script accepts the
flag directly:

.. code-block:: bash

    python examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
        model_path=<path_to_nemo_checkpoint> \
        dataset_manifest=<path_to_manifest> \
        target_lang=en-US \
        decoder_type=rnnt \
        strip_lang_tags=true

Use ``target_lang=<lang-code>`` to pin every sample in the batch to one language, or
``target_lang=auto`` to read the per-sample ``target_lang`` field from each manifest entry.
Setting ``strip_lang_tags=true`` removes ``<xx-XX>`` language tags from the decoded text
(pattern is customizable via ``lang_tag_pattern``).

In Python, call ``set_inference_prompt`` once before decoding:

.. code-block:: python

    model = nemo_asr.models.EncDecRNNTBPEModelWithPrompt.restore_from("path/to/model.nemo")
    model.set_inference_prompt("en-US")
    # ... subsequent conformer_stream_step / streaming pipeline calls use this prompt ...

See :ref:`RNN-T with Prompt Conditioning Configuration <RNNT-Prompt_model__Config>` for the
full model config reference.


Apple MPS Support
-----------------

Inference on Apple M-Series GPUs is supported with PyTorch 2.0+:

.. code-block:: bash

    PYTORCH_ENABLE_MPS_FALLBACK=1 python examples/asr/speech_to_text_eval.py \
        (...other parameters...) \
        allow_mps=true


Execution Flow
--------------

When writing custom inference scripts, follow the execution flow diagram at the
`ASR examples README <https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/README.md>`_.


Offline TensorRT encoder
------------------------

An exported offline ASR encoder can run in TensorRT while preprocessing and RNNT/TDT greedy decoding remain in PyTorch.
Install a CUDA-compatible TensorRT 10.x package and Polygraphy in addition to NeMo's ASR dependencies.
The encoder ONNX file and the decoding checkpoint must come from the same model.
Only the encoder is compiled; this workflow does not export the autoregressive decoding loop.

Export the encoder using the model's supported ONNX export interface, then build an engine:

.. code-block:: bash

    python examples/asr/export/build_trt_encoder.py --onnx encoder.onnx --engine encoder.plan --features 128

Set the feature count to the model's input feature dimension.
The example uses batches 1–16 and 8–4000 feature frames before subsampling; adjust the profile flags for your workload.
The ONNX export's positional-cache capacity must also cover the requested maximum length.
FP32 with TF32 disabled is the default.
Use --fp16 to permit FP16 internal kernels, and validate recognition accuracy separately.
Input and output dtypes are determined by the engine, independently of its internal kernel precision.
A timing cache can be saved and reused with --timing-cache.
Engines and timing caches must be built for compatible GPU hardware and TensorRT/CUDA versions.

Transcribe mono audio at the checkpoint's sample rate:

.. code-block:: bash

    python examples/asr/export/transducer/infer_transducer_trt.py --model model.nemo --engine encoder.plan --audio a.wav b.wav --output predictions.jsonl --batch-size 2 --compare-native

The optional native comparison records both predictions in the JSONL output.
It is a correctness comparison, not a throughput measurement.
Preprocessing and the decoder remain FP32 in this example, and CUDA graph decoding is disabled.
The example retains the native encoder for comparison; it does not measure minimum deployment memory.

For an existing inference loop:

.. code-block:: python

    from nemo.collections.asr.parts.submodules.tensorrt_encoder import TensorRTEncoder

    encoder = TensorRTEncoder("encoder.plan", device="cuda:0")
    encoded, encoded_lengths = encoder(features, feature_lengths)
    hypotheses = model.decoding.rnnt_decoder_predictions_tensor(
        encoded.float(), encoded_lengths, return_hypotheses=True
    )

Features and integer lengths must be CUDA tensors on the selected device.
The runtime uses optimization profile zero and accepts dynamic shapes within that profile.
It requires linear device I/O named audio_signal, length, outputs and encoded_lengths.
Host I/O, data-dependent output shapes, streaming caches, autograd and CUDA graph capture are outside this interface.
Each instance belongs to one CUDA stream after its first call.
Use a separate instance for another stream, serialize calls, and keep the instance alive until its queued work completes.
As with other asynchronous PyTorch operations, the caller must establish dependencies between producer and consumer streams.


Waveform preprocessing engine
----------------------------

The log-mel preprocessor can be exported separately from the encoder.
Its input is a zero-padded float32 array of mono waveform samples, input_signal, with shape [batch, samples],
and an int64 length array containing each waveform's sample count.
Decode audio files and resample to the checkpoint's sample rate before invoking the engine.
The outputs are processed_signal with shape [batch, features, frames] and processed_length in feature frames.
These outputs can feed the encoder's audio_signal and length inputs.

Build from the same checkpoint used for the encoder:

.. code-block:: bash

    python examples/asr/export/build_trt_preprocessor.py --model model.nemo --onnx preprocessor.onnx --engine preprocessor.plan

The builder uses native NeMo preprocessing in evaluation mode, including its window, preemphasis, mel filter bank, log scaling, normalization, length calculation and padding.
Dithering and training augmentation are disabled in evaluation mode.
The engine uses FP32 with TF32 disabled; building requires PyTorch, NeMo, TensorRT and Polygraphy.
The ONNX path uses opset 17 STFT with real/imaginary output pairs.
Dynamic export currently supports per-feature or no normalization and requires frame_splicing=1.
The default profile accepts batches 1–16 with 800–640000 padded waveform samples; adjust --min-samples, --opt-samples, --max-samples, --opt-batch and --max-batch for the workload.
Profile dimensions are sample counts, not seconds or feature frames.
Keep the resulting feature widths within the separate encoder engine's profile.
Validate feature values, lengths and recognition accuracy after building for the deployment environment.

The standalone runner requires only NumPy, TensorRT and cuda-python at runtime:

.. code-block:: bash

    python examples/asr/export/run_trt_preprocessor.py --engine preprocessor.plan --input waveforms.npz --output features.npz

The input NPZ contains input_signal and length; the output contains processed_signal and processed_length.
The example loads the engine and copies data between CPU and GPU for each call.
For serving, retain an execution context and buffers and connect the preprocessing device outputs directly to the encoder.
This example removes PyTorch from preprocessing execution; the encoder transcription example still uses the NeMo RNNT/TDT decoder.

Numerical validation is required even for an FP32 engine.
Small floating-point differences before per-feature normalization can produce larger differences when the signal variance is very low.
Check quiet audio and silence explicitly, as well as representative recognition data; FP32 does not guarantee elementwise equivalence to native preprocessing.
