
|status| |documentation| |codeql| |license| |pypi| |pyversion| |downloads| |black|

.. |status| image:: http://www.repostatus.org/badges/latest/active.svg
  :target: http://www.repostatus.org/#active
  :alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.

.. |documentation| image:: https://readthedocs.com/projects/nvidia-nemo/badge/?version=main
  :alt: Documentation
  :target: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/

.. |license| image:: https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg
  :target: https://github.com/NVIDIA/NeMo/blob/master/LICENSE
  :alt: NeMo core license and license for collections in this repo

.. |pypi| image:: https://badge.fury.io/py/nemo-toolkit.svg
  :target: https://badge.fury.io/py/nemo-toolkit
  :alt: Release version

.. |pyversion| image:: https://img.shields.io/pypi/pyversions/nemo-toolkit.svg
  :target: https://badge.fury.io/py/nemo-toolkit
  :alt: Python version

.. |downloads| image:: https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads
  :target: https://pepy.tech/project/nemo-toolkit
  :alt: PyPi total downloads

.. |codeql| image:: https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push
  :target: https://github.com/nvidia/nemo/actions/workflows/codeql.yml
  :alt: CodeQL

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
  :target: https://github.com/psf/black
  :alt: Code style: black

.. _main-readme:

**NVIDIA NeMo**
===============

Introduction
------------

NVIDIA NeMo is a conversational AI toolkit built for researchers working on automatic speech recognition (ASR),
text-to-speech synthesis (TTS), large language models (LLMs), and
natural language processing (NLP).
The primary objective of NeMo is to help researchers from industry and academia to reuse prior work (code and pretrained models)
and make it easier to create new `conversational AI models <https://developer.nvidia.com/conversational-ai#started>`_.

All NeMo models are trained with `Lightning <https://github.com/Lightning-AI/lightning>`_ and
training is automatically scalable to 1000s of GPUs.
Additionally, NeMo Megatron LLM models can be trained up to 1 trillion parameters using tensor and pipeline model parallelism.
NeMo models can be optimized for inference and deployed for production use-cases with `NVIDIA Riva <https://developer.nvidia.com/riva>`_.

Getting started with NeMo is simple.
State of the Art pretrained NeMo models are freely available on `HuggingFace Hub <https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia>`_ and
`NVIDIA NGC <https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC>`_.
These models can be used to transcribe audio, synthesize speech, or translate text in just a few lines of code.

We have extensive `tutorials <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html>`_ that
can all be run on `Google Colab <https://colab.research.google.com>`_.

For advanced users that want to train NeMo models from scratch or finetune existing NeMo models
we have a full suite of `example scripts <https://github.com/NVIDIA/NeMo/tree/main/examples>`_ that support multi-GPU/multi-node training.

For scaling NeMo LLM training on Slurm clusters or public clouds, please see the `NVIDIA NeMo Megatron Launcher <https://github.com/NVIDIA/NeMo-Megatron-Launcher>`_.
The NM launcher has extensive recipes, scripts, utilities, and documentation for training NeMo LLMs and also has an `Autoconfigurator <https://github.com/NVIDIA/NeMo-Megatron-Launcher#53-using-autoconfigurator-to-find-the-optimal-configuration>`_
which can be used to find the optimal model parallel configuration for training on a specific cluster.

Also see our `introductory video <https://www.youtube.com/embed/wBgpMf_KQVw>`_ for a high level overview of NeMo.

Key Features
------------

* Speech processing
    * `HuggingFace Space for Audio Transcription (File, Microphone and YouTube) <https://huggingface.co/spaces/smajumdar/nemo_multilingual_language_id>`_
    * `Automatic Speech Recognition (ASR) <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/intro.html>`_
        * Supported ASR models: `<https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/models.html>`_
            * Jasper, QuartzNet, CitriNet, ContextNet
            * Conformer-CTC, Conformer-Transducer, FastConformer-CTC, FastConformer-Transducer
            * Squeezeformer-CTC and Squeezeformer-Transducer
            * LSTM-Transducer (RNNT) and LSTM-CTC
        * Supports the following decoders/losses:
            * CTC
            * Transducer/RNNT
            * Hybrid Transducer/CTC
            * NeMo Original `Multi-blank Transducers <https://arxiv.org/abs/2211.03541>`_ and `Token-and-Duration Transducers (TDT) <https://arxiv.org/abs/2304.06795>`_
        * Streaming/Buffered ASR (CTC/Transducer) - `Chunked Inference Examples <https://github.com/NVIDIA/NeMo/tree/stable/examples/asr/asr_chunked_inference>`_
        * Cache-aware Streaming Conformer with multiple lookaheads - `<https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/models.html#cache-aware-streaming-conformer>`_
        * Beam Search decoding
        * `Language Modelling for ASR (CTC and RNNT) <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_language_modeling.html>`_: N-gram LM in fusion with Beam Search decoding, Neural Rescoring with Transformer
        * `Support of long audios for Conformer with memory efficient local attention <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/results.html#inference-on-long-audio>`_
    * `Speech Classification, Speech Command Recognition and Language Identification <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speech_classification/intro.html>`_: MatchboxNet (Command Recognition), AmberNet (LangID)
    * `Voice activity Detection (VAD) <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speech_classification/models.html#marblenet-vad>`_: MarbleNet
        * ASR with VAD Inference - `Example <https://github.com/NVIDIA/NeMo/tree/stable/examples/asr/asr_vad>`_
    * `Speaker Recognition <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_recognition/intro.html>`_: TitaNet, ECAPA_TDNN, SpeakerNet
    * `Speaker Diarization <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_diarization/intro.html>`_
        * Clustering Diarizer: TitaNet, ECAPA_TDNN, SpeakerNet
        * Neural Diarizer: MSDD (Multi-scale Diarization Decoder)
    * `Speech Intent Detection and Slot Filling <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speech_intent_slot/intro.html>`_: Conformer-Transformer
    * `Pretrained models on different languages. <https://ngc.nvidia.com/catalog/collections/nvidia:nemo_asr>`_: English, Spanish, German, Russian, Chinese, French, Italian, Polish, ...
    * `NGC collection of pre-trained speech processing models. <https://ngc.nvidia.com/catalog/collections/nvidia:nemo_asr>`_
* Natural Language Processing
    * `NeMo Megatron pre-training of Large Language Models <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/nemo_megatron/intro.html>`_
    * `Neural Machine Translation (NMT) <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/machine_translation/machine_translation.html>`_
    * `Punctuation and Capitalization <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/punctuation_and_capitalization.html>`_
    * `Token classification (named entity recognition) <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/token_classification.html>`_
    * `Text classification <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/text_classification.html>`_
    * `Joint Intent and Slot Classification <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/joint_intent_slot.html>`_
    * `Question answering <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/question_answering.html>`_
    * `GLUE benchmark <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/glue_benchmark.html>`_
    * `Information retrieval <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/information_retrieval.html>`_
    * `Entity Linking <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/entity_linking.html>`_
    * `Dialogue State Tracking <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/sgd_qa.html>`_
    * `Prompt Learning <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/nemo_megatron/prompt_learning.html>`_
    * `NGC collection of pre-trained NLP models. <https://ngc.nvidia.com/catalog/collections/nvidia:nemo_nlp>`_
    * `Synthetic Tabular Data Generation <https://developer.nvidia.com/blog/generating-synthetic-data-with-transformers-a-solution-for-enterprise-data-challenges/>`_
* Text-to-Speech Synthesis (TTS):
    * `Documentation <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/tts/intro.html#>`_
    * Mel-Spectrogram generators: FastPitch, SSL FastPitch, Mixer-TTS/Mixer-TTS-X, RAD-TTS, Tacotron2
    * Vocoders: HiFiGAN, UnivNet, WaveGlow
    * End-to-End Models: VITS
    * `Pre-trained Model Checkpoints in NVIDIA GPU Cloud (NGC) <https://ngc.nvidia.com/catalog/collections/nvidia:nemo_tts>`_
* `Tools <https://github.com/NVIDIA/NeMo/tree/stable/tools>`_
    * `Text Processing (text normalization and inverse text normalization) <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/text_normalization/intro.html>`_
    * `NeMo Forced Aligner <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/tools/nemo_forced_aligner.html>`_
    * `CTC-Segmentation tool <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/tools/ctc_segmentation.html>`_
    * `Speech Data Explorer <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/tools/speech_data_explorer.html>`_: a dash-based tool for interactive exploration of ASR/TTS datasets
    * `Speech Data Processor <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/tools/speech_data_processor.html>`_


Built for speed, NeMo can utilize NVIDIA's Tensor Cores and scale out training to multiple GPUs and multiple nodes.

Requirements
------------

1) Python 3.10 or above
2) Pytorch 1.13.1 or above
3) NVIDIA GPU for training

Documentation
-------------

.. |main| image:: https://readthedocs.com/projects/nvidia-nemo/badge/?version=main
  :alt: Documentation Status
  :scale: 100%
  :target: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/

.. |stable| image:: https://readthedocs.com/projects/nvidia-nemo/badge/?version=stable
  :alt: Documentation Status
  :scale: 100%
  :target:  https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/

+---------+-------------+------------------------------------------------------------------------------------------------------------------------------------------+
| Version | Status      | Description                                                                                                                              |
+=========+=============+==========================================================================================================================================+
| Latest  | |main|      | `Documentation of the latest (i.e. main) branch. <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/>`_                  |
+---------+-------------+------------------------------------------------------------------------------------------------------------------------------------------+
| Stable  | |stable|    | `Documentation of the stable (i.e. most recent release) branch. <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/>`_ |
+---------+-------------+------------------------------------------------------------------------------------------------------------------------------------------+

Tutorials
---------
A great way to start with NeMo is by checking `one of our tutorials <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html>`_.

Getting help with NeMo
----------------------
FAQ can be found on NeMo's `Discussions board <https://github.com/NVIDIA/NeMo/discussions>`_. You are welcome to ask questions or start discussions there.


Installation
------------
Please refer to the installation guidance in `INSTALLATION.md <./INSTALLATION.md>`_.

Examples
--------

Many examples can be found under the `"Examples" <https://github.com/NVIDIA/NeMo/tree/stable/examples>`_ folder.


Contributing
------------

We welcome community contributions! Please refer to `CONTRIBUTING.md <https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md>`_ for the process.

Publications
------------

We provide an ever growing list of publications that utilize the NeMo framework. Please refer to `PUBLICATIONS.md <https://github.com/NVIDIA/NeMo/tree/stable/PUBLICATIONS.md>`_. We welcome the addition of your own articles to this list !

License
-------
NeMo is under `Apache 2.0 license <https://github.com/NVIDIA/NeMo/blob/stable/LICENSE>`_.
