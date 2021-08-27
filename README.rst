
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

NVIDIA NeMo is a conversational AI toolkit built for researchers working on automatic speech recognition (ASR), natural language processing (NLP), and text-to-speech synthesis (TTS).
The primary objective of NeMo is to help researchers from industry and academia to reuse prior work (code and pretrained models and make it easier to create new `conversational AI models <https://developer.nvidia.com/conversational-ai#started>`_.


`Introductory video. <https://www.youtube.com/embed/wBgpMf_KQVw>`_

Key Features
------------

* Speech processing
    * `Automatic Speech recognition (ASR) <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/intro.html>`_: Jasper, QuartzNet, CitriNet, Conformer
    * `Speech Classification <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speech_classification/intro.html>`_: MatchboxNet (command recognition), MarbleNet (voice activity detection)
    * `Speaker Recognition <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_recognition/intro.html>`_: SpeakerNet
    * `Speaker Diarization <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_diarization/intro.html>`_: MarbleNet + SpeakerNet
    * `NGC collection of pre-trained speech processing models. <https://ngc.nvidia.com/catalog/collections/nvidia:nemo_asr>`_
* Natural Language Processing
    * `Compatible with Hugging Face Transformers and NVIDIA Megatron <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/megatron_finetuning.html>`_
    * `Neural Machine Translation (NMT) <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/machine_translation.html>`_
    * `Punctuation and Capitalization <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/punctuation_and_capitalization.html>`_
    * `Token classification (named entity recognition) <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/token_classification.html>`_
    * `Text classification <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/text_classification.html>`_
    * `Joint Intent and Slot Classification <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/joint_intent_slot.html>`_
    * `BERT pre-training <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/bert_pretraining.html>`_
    * `Question answering <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/question_answering.html>`_
    * `GLUE benchmark <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/glue_benchmark.html>`_
    * `Information retrieval <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/information_retrieval.html>`_
    * `Entity Linking <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/entity_linking.html>`_
    * `Dialogue State Tracking <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/sgd_qa.html>`_
    * `Neural Duplex Text Normalization <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/text_normalization.html>`_
    * `NGC collection of pre-trained NLP models. <https://ngc.nvidia.com/catalog/collections/nvidia:nemo_nlp>`_
* `Speech synthesis (TTS) <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/tts/intro.html#>`_
    * Spectrogram generation: Tacotron2, GlowTTS, FastSpeech2, FastPitch, FastSpeech2
    * Vocoders: WaveGlow, SqueezeWave, UniGlow, MelGAN, HiFiGAN
    * End-to-end speech generation: FastPitch_HifiGan_E2E, FastSpeech2_HifiGan_E2E
    * `NGC collection of pre-trained TTS models. <https://ngc.nvidia.com/catalog/collections/nvidia:nemo_tts>`_


Built for speed, NeMo can utilize NVIDIA's Tensor Cores and scale out training to multiple GPUs and multiple nodes.

Requirements
------------

1) Python 3.6, 3.7 or 3.8
2) Pytorch 1.8.1 or above
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

Pip
~~~
Use this installation mode if you want the latest released version.

.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    pip install Cython
    pip install nemo_toolkit['all']

Pip from source
~~~~~~~~~~~~~~~
Use this installation mode if you want the a version from particular GitHub branch (e.g main).

.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    pip install Cython
    python -m pip install git+https://github.com/NVIDIA/NeMo.git@{BRANCH}#egg=nemo_toolkit[all]


From source
~~~~~~~~~~~
Use this installation mode if you are contributing to NeMo.

.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    git clone https://github.com/NVIDIA/NeMo
    cd NeMo
    ./reinstall.sh

RNNT
~~~~
Note that RNNT requires numba to be installed from conda.

.. code-block:: bash

  conda remove numba
  pip uninstall numba
  conda install -c numba numba

Docker containers:
~~~~~~~~~~~~~~~~~~

If you chose to work with main branch, we recommend using NVIDIA's PyTorch container version 21.05-py3 and then installing from GitHub.

.. code-block:: bash

    docker run --gpus all -it --rm -v <nemo_github_folder>:/NeMo --shm-size=8g \
    -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit \
    stack=67108864 --device=/dev/snd nvcr.io/nvidia/pytorch:21.05-py3

Examples
--------

Many example can be found under `"Examples" <https://github.com/NVIDIA/NeMo/tree/stable/examples>`_ folder.


Contributing
------------

We welcome community contributions! Please refer to the  `CONTRIBUTING.md <https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md>`_ CONTRIBUTING.md for the process.

License
-------
NeMo is under `Apache 2.0 license <https://github.com/NVIDIA/NeMo/blob/stable/LICENSE>`_.
