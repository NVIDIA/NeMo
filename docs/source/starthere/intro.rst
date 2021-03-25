Introduction
============

.. # define a hard line break for html
.. |br| raw:: html

    <br />

.. _dummy_header:

`NVIDIA NeMo <https://github.com/NVIDIA/NeMo>`_ is a toolkit for building new State-of-the-Art
Conversational AI models. NeMo has separate collections for Automatic Speech Recognition (ASR),
Natural Language Processing (NLP), and Text-to-Speech (TTS) models. Each collection consists of
prebuilt modules that include everything needed to train on your data.
Every module can easily be customized, extended, and composed to create new Conversational AI
model architectures.

Conversational AI architectures are typically large and require a lot of data  and compute
for training. NeMo uses PyTorch Lightning for easy and performant multi-GPU/multi-node
mixed-precision training.

.. raw:: html

    <div style="position: relative; padding-bottom: 3%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe width="560" height="315" src="https://www.youtube.com/embed/wBgpMf_KQVw" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </div>

Requirements
------------

1) Python 3.6, 3.7 or 3.8
2) Pytorch 1.7.1.  WARNING: This version currently does not support Pytorch 1.8.0
3) NVIDIA GPU for training.

Quick Start
-----------

We start by describing a simple prototype application which will take audio in one language and translate it into audio in another.
The fastest way to start is to go through `Getting Started Colab notebook. <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0rc1/tutorials/NeMo_Getting_Started.ipynb>`_

Below we is the exact same code-snippet for your reference.

.. code-block:: python

    # Import NeMo and it's ASR, NLP and TTS collections
    import nemo
    # Import Speech Recognition collection
    import nemo.collections.asr as nemo_asr
    # Import Natural Language Processing colleciton
    import nemo.collections.nlp as nemo_nlp
    # Import Speech Synthesis collection
    import nemo.collections.tts as nemo_tts

    # Next, we instantiate all the necessary models directly from NVIDIA NGC
    # Speech Recognition model - QuartzNet trained on Russian part of MCV 6.0
    quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="stt_ru_quartznet15x5").cuda()
    # Neural Machine Translation model
    nmt_model = nemo_nlp.models.MTEncDecModel.from_pretrained(model_name='nmt_ru_en_transformer6x6', strict=False).cuda()
    # Spectrogram generator which takes text as an input and produces spectrogram
    spectrogram_generator = nemo_tts.models.Tacotron2Model.from_pretrained(model_name="tts_en_tacotron2").cuda()
    # Vocoder model which takes spectrogram and produces actual audio
    vocoder = nemo_tts.models.WaveGlowModel.from_pretrained(model_name="tts_waveglow_88m").cuda()
    # Transcribe an audio file
    # IMPORTANT: The audio must be mono with 16Khz sampling rate
    # Get example from: https://nemo-public.s3.us-east-2.amazonaws.com/mcv-samples-ru/common_voice_ru_19034087.wav
    russian_text = quartznet.transcribe(['Path_to_audio_file'])
    print(russian_text)
    # You should see russian text here. Let's translate it to English
    english_text = nmt_model.translate(russian_text)
    print(english_text)
    # After this you should see English translation
    # Let's convert it into audio
    # A helper function which combines Tacotron2 and WaveGlow to go directly from
    # text to audio
    def text_to_audio(text):
      parsed = spectrogram_generator.parse(text)
      spectrogram = spectrogram_generator.generate_spectrogram(tokens=parsed)
      audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
      return audio.to('cpu').numpy()
    audio = text_to_audio(english_text[0])


Installation
------------

Pip
~~~
Use this installation mode if you want the latest released version.

.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    pip install Cython
    pip install nemo_toolkit[all]==1.0.0b3

Pip from source
~~~~~~~~~~~~~~~
Use this installation mode if you want the a version from particular GitHub branch (e.g main).

.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    pip install Cython
    python -m pip install git+https://github.com/NVIDIA/NeMo.git@{BRANCH}#egg=nemo_toolkit[all]
    # For r1.0.0rc1, replace {BRANCH} with r1.0.0rc1 like so:
    # python -m pip install git+https://github.com/NVIDIA/NeMo.git@r1.0.0rc1#egg=nemo_toolkit[all]

From source
~~~~~~~~~~~
Use this installation mode if you are contributing to NeMo.

.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    git clone https://github.com/NVIDIA/NeMo
    cd NeMo
    ./reinstall.sh

Docker containers
~~~~~~~~~~~~~~~~~
The easiest way to start training with NeMo is by using `NeMo's container <https://ngc.nvidia.com/catalog/containers/nvidia:nemo>`_.
It has all requirements and NeMo 1.0.0b3 already installed.

.. code-block:: bash

    docker run --gpus all -it --rm --shm-size=8g \
    -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit \
    stack=67108864 --device=/dev/snd nvcr.io/nvidia/nemo:1.0.0b3


If you chose to work with main branch, we recommend using NVIDIA's PyTorch container version 20.11-py3 and then installing from GitHub.

.. code-block:: bash

    docker run --gpus all -it --rm -v <nemo_github_folder>:/NeMo --shm-size=8g \
    -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit \
    stack=67108864 --device=/dev/snd nvcr.io/nvidia/pytorch:20.11-py3


FAQ
---
Have a look at our `Discussions board <https://github.com/NVIDIA/NeMo/discussions>`_ and feel free to post a question or start a discussion.


Contributing
------------

We welcome community contributions! Please refer to the  `CONTRIBUTING.md <https://github.com/NVIDIA/NeMo/blob/main/CONTRIBUTING.md>`_  for the process.

License
-------
NeMo is under `Apache 2.0 license <https://github.com/NVIDIA/NeMo/blob/main/LICENSE>`_.