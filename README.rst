
|status| |documentation| |license| |lgtm_grade| |lgtm_alerts| |black|

.. |status| image:: http://www.repostatus.org/badges/latest/active.svg
  :target: http://www.repostatus.org/#active
  :alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.

.. |documentation| image:: https://readthedocs.com/projects/nvidia-nemo/badge/?version=main
  :alt: Documentation
  :target: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/

.. |license| image:: https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg
  :target: https://github.com/NVIDIA/NeMo/blob/master/LICENSE
  :alt: NeMo core license and license for collections in this repo

.. |lgtm_grade| image:: https://img.shields.io/lgtm/grade/python/g/NVIDIA/NeMo.svg?logo=lgtm&logoWidth=18
  :target: https://lgtm.com/projects/g/NVIDIA/NeMo/context:python
  :alt: Language grade: Python

.. |lgtm_alerts| image:: https://img.shields.io/lgtm/alerts/g/NVIDIA/NeMo.svg?logo=lgtm&logoWidth=18
  :target: https://lgtm.com/projects/g/NVIDIA/NeMo/alerts/
  :alt: Total alerts

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
  :target: https://github.com/psf/black
  :alt: Code style: black

.. _main-readme:

**NVIDIA NeMo**
===============

Introduction
------------

NVIDIA NeMo is a `conversational AI <https://developer.nvidia.com/conversational-ai#started>`_ toolkit built for researchers working on automatic speech recognition (ASR),
text-to-speech synthesis (TTS), large language models (LLMs), and 
natural language processing (NLP).

All NeMo models are trained with `Lightning <https://github.com/Lightning-AI/lightning>`_ and 
training is automatically scalable to 1000s of GPUs. 
Additionally, NeMo Megatron LLM models can be trained up to 1 trillion parameters using tensor and pipeline model parallelism.
NeMo models can be optimized for inference and deployed for production use-cases with `NVIDIA Riva <https://developer.nvidia.com/riva>`_.

Pretrained NeMo models are available on `HuggingFace Hub <https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia>`_ and
`NVIDIA NGC <https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC>`_.
These models can be used to transcribe audio, synthesize speech, translate text or perform various NLP tasks in a just a few lines of code.

We have have extensive `tutorials <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html>`_ that 
can all be run on `Google Colab <https://colab.research.google.com>`_.

For advanced users that want to train NeMo models from scratch or finetune existing NeMo models 
we have a full suite of `example scripts <https://github.com/NVIDIA/NeMo/tree/update_readme_into/examples>`_ that support multi-GPU/multi-node training.

Also see our `introductory video <https://www.youtube.com/embed/wBgpMf_KQVw>`_ for a high level overview of NeMo.

Key Features
------------

* Speech processing
    * `Automatic Speech Recognition (ASR) <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/intro.html>`_
        * Supported models: Jasper, QuartzNet, CitriNet, Conformer-CTC, Conformer-Transducer, Squeezeformer-CTC, Squeezeformer-Transducer, ContextNet, LSTM-Transducer (RNNT), LSTM-CTC, ...
        * Supports CTC and Transducer/RNNT losses/decoders
        * Beam Search decoding
        * `Language Modelling for ASR <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_language_modeling.html>`_: N-gram LM in fusion with Beam Search decoding, Neural Rescoring with Transformer
        * Streaming and Buffered ASR (CTC/Transducer) - `Chunked Inference Examples <https://github.com/NVIDIA/NeMo/tree/stable/examples/asr/asr_chunked_inference>`_
        * `Pretrained models on different languages. <https://ngc.nvidia.com/catalog/collections/nvidia:nemo_asr>`_: English, Spanish, German, Russian, Chinese, French, Italian, Polish, ...
    * `Speech Classification and Speech Command Recognition <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speech_classification/intro.html>`_: MatchboxNet (Command Recognition)
    * `Voice activity Detection (VAD) <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speech_classification/models.html#marblenet-vad>`_: MarbleNet
    * `Speaker Recognition <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_recognition/intro.html>`_: TitaNet, ECAPA_TDNN, SpeakerNet
    * `Speaker Diarization <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_diarization/intro.html>`_
        * Clustering Diarizer: TitaNet, ECAPA_TDNN, SpeakerNet
        * Neural Diarizer: MSDD (Multi-scale Diarization Decoder)
    * `Speech Intent Detection and Slot Filling <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speech_intent_slot/intro.html>`_: Conformer-Transformer
* `NeMo Megatron pre-training of Large Language Models (GPT, T5, BERT) <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/nemo_megatron/intro.html>`_
* Natural Language Processing
    * `Neural Machine Translation (NMT) <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/machine_translation/machine_translation.html>`_
    * `Punctuation and Capitalization <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/punctuation_and_capitalization.html>`_
    * `Token classification (named entity recognition) <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/token_classification.html>`_
    * `Text classification <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/text_classification.html>`_
    * `Joint Intent and Slot Classification <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/joint_intent_slot.html>`_    
    * `Question answering <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/question_answering.html>`_
    * `Entity Linking <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/entity_linking.html>`_
* `Speech synthesis (TTS) <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/tts/intro.html#>`_
    * Spectrogram generation: Tacotron2, GlowTTS, TalkNet, FastPitch, FastSpeech2, Mixer-TTS, Mixer-TTS-X
    * Vocoders: WaveGlow, SqueezeWave, UniGlow, MelGAN, HiFiGAN, UnivNet
    * End-to-end speech generation: FastPitch_HifiGan_E2E, FastSpeech2_HifiGan_E2E
* `Tools <https://github.com/NVIDIA/NeMo/tree/stable/tools>`_
    * `Text Processing (text normalization and inverse text normalization) <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/text_normalization/intro.html>`_
    * `CTC-Segmentation tool <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/tools/ctc_segmentation.html>`_
    * `Speech Data Explorer <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/tools/speech_data_explorer.html>`_: a dash-based tool for interactive exploration of ASR/TTS datasets

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


Installation
------------

https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/intro.html#installation


Examples
--------

Many examples can be found under `"Examples" <https://github.com/NVIDIA/NeMo/tree/stable/examples>`_ folder.


Getting help with NeMo
----------------------
FAQ can be found on NeMo's `Discussions board <https://github.com/NVIDIA/NeMo/discussions>`_. You are welcome to ask questions or start discussions there.


Contributing
------------

We welcome community contributions! Please refer to the  `CONTRIBUTING.md <https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md>`_ CONTRIBUTING.md for the process.

Publications
------------

We provide an ever growing list of publications that utilize the NeMo framework. Please refer to `PUBLICATIONS.md <https://github.com/NVIDIA/NeMo/tree/stable/PUBLICATIONS.md>`_. We welcome the addition of your own articles to this list !

Citation
--------

.. code-block:: bash

  @article{kuchaiev2019nemo,
    title={Nemo: a toolkit for building ai applications using neural modules},
    author={Kuchaiev, Oleksii and Li, Jason and Nguyen, Huyen and Hrinchuk, Oleksii and Leary, Ryan and Ginsburg, Boris and Kriman, Samuel and Beliaev, Stanislav and Lavrukhin, Vitaly and Cook, Jack and others},
    journal={arXiv preprint arXiv:1909.09577},
    year={2019}
  }

License
-------
NeMo is under `Apache 2.0 license <https://github.com/NVIDIA/NeMo/blob/stable/LICENSE>`_.
