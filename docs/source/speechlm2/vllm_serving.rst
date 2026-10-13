Serving SpeechLM with vLLM
==========================

The NeMo SpeechLM vLLM plugin serves exported SpeechLM checkpoints with
their speech encoder and language backbone. It supports ordinary generation
and checkpoint-backed speculative decoding.

CTC word timestamp alignment
---------------------------

When CTC timestamps are requested, native SpeechLM and the vLLM timestamp
worker use the same single-pass banded CTC word aligner. Speaker mapping
uses greedy token distributions; the search retains speaker-active frame
support and scores an anchor-guided band. There is no preliminary DP pass
or selectable legacy alignment backend.

The Sortformer log-probability prior defaults to weight ``1.0``. The vLLM
checkpoint setting ``ctc_timestamps.speaker_logprob_weight`` may override
this weight without changing the alignment algorithm. Diarization-assisted
word-boundary refinement always runs, with a default cap of 12 frames
(at most 0.96 seconds). Native inference automatically chooses transcript
units from the writing script unless ``timestamp_language`` is explicit.

Remove the former ``alignment_backend`` option from native aligner
configuration; it is no longer accepted. Timestamp generation remains
opt-in and requires a CTC adapter.

Installation
------------

Install the ASR runtime and vLLM 0.28.0, pinned by NeMo's ``vllm`` extra:

.. code-block:: bash

   pip install -e ".[asr,vllm]"

The equivalent uv command is:

.. code-block:: bash

   uv sync --extra asr --extra vllm

Do not combine ``vllm`` with the ``speechlm2``, ``speechlm2-only``,
``all``, ``cu12``, ``cu13``, ``compiled``, or ``compiled-a100`` extras.
These include Automodel training dependencies; vLLM owns the exact Torch
and CUDA-kernel stack for this serving environment. Automodel is not
required to serve an exported checkpoint.

Serving an exported checkpoint
------------------------------

Start ordinary BF16 serving with a vLLM-ready NeMo SpeechLM checkpoint:

.. code-block:: bash

   vllm serve /path/to/vllm-ready-speechlm-checkpoint \
     --trust-remote-code \
     --dtype bfloat16

Hybrid checkpoints with exported MTP weights can also use vLLM's MTP draft.
Set the number of speculative tokens according to the checkpoint's trained
prediction depths, and use ``--mamba-cache-mode all`` with MTP. For a separate
checkpoint-backed draft, see :doc:`vllm_dflash`.

.. _speechlm2-vllm-hybrid-precision:

Hybrid checkpoint precision
---------------------------

The SpeechLM wrapper preserves FP32 Mamba ``A``, ``D``, and ``dt_bias``
parameters before checkpoint loading, for both the target and MTP draft.
MoE routers use FP32 weights, operands, and output logits, matching the
Automodel training precision contract. Other language-model and perception
weights retain their configured serving precision. FP32 SSM cache defaults
are delegated to vLLM's NemotronH configuration hook.

.. _speechlm2-vllm-encoder-cuda-graphs:

Audio encoder CUDA graphs
-------------------------

vLLM encodes the audio of the requests it admits in each engine step, often a few dozen short clips. For a
``ParallelExpertEncoder`` at that batch size, the encoder's cost is mostly kernel launches. Setting
``encoder_cuda_graphs`` replays the encoder (everything after the mel front end) from CUDA graphs captured per
batch size and audio length:

.. code-block:: bash

   vllm serve /path/to/vllm-ready-speechlm-checkpoint \
     --trust-remote-code \
     --dtype bfloat16 \
     --hf-overrides '{"encoder_cuda_graphs": true, "encoder_compile": true}'

The keys can also be set in the checkpoint's ``config.json``.

- ``encoder_cuda_graphs``: capture graphs for batches of 8 to 128 clips and lengths up to two 40-second encoder
  windows, at the first encoder call. Longer audio runs the encoder without graphs. Capture adds start-up time and
  GPU memory; the graphs reproduce the encoder's outputs.
- ``encoder_compile``: also compile each encoder transformer block with ``torch.compile``, fusing its
  memory-bound operations. This lowers encoder GPU time further and changes the outputs at bf16 rounding level.

Both are off by default. The graphs do not apply while CTC timestamp capture is enabled.
