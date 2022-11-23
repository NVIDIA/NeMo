Introduction
============

.. # define a hard line break for html
.. |br| raw:: html

    <br />

.. _dummy_header:

`NVIDIA NeMo <https://github.com/NVIDIA/NeMo>`_, part of the NVIDIA AI platform, is a toolkit for building new state-of-the-art
conversational AI models. NeMo has separate collections for Automatic Speech Recognition (ASR),
Natural Language Processing (NLP), and Text-to-Speech (TTS) synthesis models. Each collection consists of
prebuilt modules that include everything needed to train on your data.
Every module can easily be customized, extended, and composed to create new conversational AI
model architectures.

Conversational AI architectures are typically large and require a lot of data and compute
for training. NeMo uses `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ for easy and performant multi-GPU/multi-node
mixed-precision training.


.. raw:: html

    <div style="position: relative; padding-bottom: 3%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe width="560" height="315" src="https://www.youtube.com/embed/wBgpMf_KQVw" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </div>

For more information and questions, visit the `NVIDIA NeMo Discussion Board <https://github.com/NVIDIA/NeMo/discussions>`_.

Prerequisites
-------------

To use NeMo you'll need:

#. Python version 3.8 or above

#. Pytorch version 1.12.0 or above

#. NVIDIA GPU for training.

We test with Ubuntu Linux, typically using docker containers. You may be able to run on MacOS to try models.

Installation
------------
Conda
~~~~~

We recommend installing NeMo in a fresh Conda environment.

.. code-block:: bash

    conda create --name nemo python==3.8
    conda activate nemo

Install PyTorch using their `configurator <https://pytorch.org/get-started/locally/>`_.

.. code-block:: bash

    conda install pytorch torchvision torchaudio -c pytorch

.. note::

  The command used to install PyTorch may depend on your system.

Pip
~~~
Use this installation mode if you want the latest released version.

.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    pip install Cython
    pip install nemo_toolkit['all']

.. note::

    Depending on the shell used, you may need to use ``"nemo_toolkit[all]"`` instead in the above command.

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

.. note::

    If you only want the toolkit without additional conda-based dependencies, you may replace ``reinstall.sh``
    with ``pip install -e .`` when your PWD is the root of the NeMo repository.

RNNT
~~~~
Note that RNNT requires numba to be installed from conda.

.. code-block:: bash

  conda remove numba
  pip uninstall numba
  conda install -c conda-forge numba

NeMo Megatron
~~~~~~~~~~~~~
If you are using NeMo Megatron training container everything is already installed.
NeMo Megatron training requires NVIDIA Apex to be installed.
Install it manually if not using the NVIDIA PyTorch container.

.. code-block:: bash

    git clone https://github.com/ericharper/apex.git
    cd apex
    git checkout nm_v1.11.0
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_layer_norm" --global-option="--distributed_adam" --global-option="--deprecated_fused_adam" ./

Transformer Engine
~~~~~~~~~~~~~~~~~~
NeMo Megatron GPT has been integrated with `NVIDIA Transformer Engine <https://github.com/NVIDIA/TransformerEngine>`_
Transformer Engine enables FP8 training on NVIDIA Hopper GPUs.
`Install <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html>`_ it manually if not using the NVIDIA PyTorch container.

.. note::

  This is an experimental feature. Transformer Engine requires PyTorch to be built with CUDA 11.8.

NeMo Text Processing
~~~~~~~~~~~~~~~~~~~~
NeMo Text Processing, specifically (Inverse) Text Normalization, requires `Pynini <https://pypi.org/project/pynini/>`_ to be installed.

.. code-block:: bash

    bash NeMo/nemo_text_processing/install_pynini.sh

Docker containers:
~~~~~~~~~~~~~~~~~~
To build a nemo container with Dockerfile from a branch, please run

.. code-block:: bash

    DOCKER_BUILDKIT=1 docker build -f Dockerfile -t nemo:latest .


If you chose to work with main branch, we recommend using NVIDIA's PyTorch container version 22.09-py3 and then installing from GitHub.

.. code-block:: bash

    docker run --gpus all -it --rm -v <nemo_github_folder>:/NeMo --shm-size=8g \
    -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit \
    stack=67108864 --device=/dev/snd nvcr.io/nvidia/pytorch:22.09-py3

.. _quick_start_guide:

Quick Start Guide
-----------------

This NeMo Quick Start Guide is a starting point for users who want to try out NeMo; specifically, this guide enables users to quickly get started with the NeMo fundamentals by walking you through an example audio translator and voice swap.

If you're new to NeMo, the best way to get started is to take a look at the following tutorials:

* `Text Classification (Sentiment Analysis) <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/Text_Classification_Sentiment_Analysis>`__ - demonstrates the Text Classification model using the NeMo NLP collection.
* `NeMo Primer <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/00_NeMo_Primer.ipynb>`__ - introduces NeMo, PyTorch Lightning, and OmegaConf, and shows how to use, modify, save, and restore NeMo models.
* `NeMo Models <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/01_NeMo_Models.ipynb>`__ - explains the fundamental concepts of the NeMo model.
* `NeMo voice swap demo <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/NeMo_voice_swap_app.ipynb>`__ - demonstrates how to swap a voice in the audio fragment with a computer generated one using NeMo.

Below we is the code snippet of Audio Translator application.

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
    nmt_model = nemo_nlp.models.MTEncDecModel.from_pretrained(model_name='nmt_ru_en_transformer6x6').cuda()
    # Spectrogram generator which takes text as an input and produces spectrogram
    spectrogram_generator = nemo_tts.models.FastPitchModel.from_pretrained(model_name="tts_en_fastpitch").cuda()
    # Vocoder model which takes spectrogram and produces actual audio
    vocoder = nemo_tts.models.HifiGanModel.from_pretrained(model_name="tts_hifigan").cuda()
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
    # A helper function which combines FastPitch and HiFiGAN to go directly from
    # text to audio
    def text_to_audio(text):
      parsed = spectrogram_generator.parse(text)
      spectrogram = spectrogram_generator.generate_spectrogram(tokens=parsed)
      audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
      return audio.to('cpu').numpy()
    audio = text_to_audio(english_text[0])




`FAQ <https://github.com/NVIDIA/NeMo/discussions>`_
---------------------------------------------------
Have a look at our `discussions board <https://github.com/NVIDIA/NeMo/discussions>`_ and feel free to post a question or start a discussion.


Contributing
------------

We welcome community contributions! Refer to the `CONTRIBUTING.md <https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md>`_  file for the process.

License
-------

NeMo is under `Apache 2.0 license <https://github.com/NVIDIA/NeMo/blob/stable/LICENSE>`_.