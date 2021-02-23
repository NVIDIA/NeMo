NVIDIA NeMo Developer Guide
===========================

.. toctree::
   :hidden:
   :maxdepth: 8

   Introduction <self>
   core
   asr/intro
   cv/intro
   nlp/intro
   tts/intro
   common/intro
   api-docs/nemo


NeMo is a library for easy training, building and manipulating of AI models.
NeMo's current focus is providing great experience for Conversational AI.

NeMo models can be trained on multi-GPU and multi-node, with or without Mixed Precision
Many models in NeMo come with high-quality pre-trained checkpoints.

.. raw:: html

    <div style="position: relative; padding-bottom: 3%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe width="560" height="315" src="https://www.youtube.com/embed/wBgpMf_KQVw" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </div>

Requirements
------------

1) Python 3.6 or above
2) Pytorch 1.7.1 or above


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


From source
~~~~~~~~~~~
Use this installation mode if you are contributing to NeMo.

.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    git clone https://github.com/NVIDIA/NeMo
    cd NeMo
    ./reinstall.sh

Docker containers:
~~~~~~~~~~~~~~~~~~
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


Getting help with NeMo
----------------------
Have a look at our `Discussions board <https://github.com/NVIDIA/NeMo/discussions>`_ and feel free to post a question or start a discussion.

Tutorials
---------
The best way to get started with NeMo is to checkout one of our tutorials.

Most NeMo tutorials can be run on `Google's Colab <https://colab.research.google.com/notebooks/intro.ipynb>`_.

To run tutorials:

* Click on Colab link (see table below)
* Connect to an instance with a GPU (Runtime -> Change runtime type -> select "GPU" for hardware accelerator)

.. list-table:: *Tutorials*
   :widths: 15 25 25
   :header-rows: 1

   * - Domain
     - Title
     - GitHub URL
   * - NeMo
     - Simple Application with NeMo
     - `Voice swap app <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0b3/tutorials/NeMo_voice_swap_app.ipynb>`_
   * - NeMo
     - Exploring NeMo Fundamentals
     - `NeMo primer <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0b3/tutorials/00_NeMo_Primer.ipynb>`_
   * - NeMo Models
     - Exploring NeMo Model Construction
     - `NeMo models <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0b3/tutorials/01_NeMo_Models.ipynb>`_
   * - ASR
     - ASR with Subword Tokenization
     - `ASR with Subword Tokenization <https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/asr/08_ASR_with_Subword_Tokenization.ipynb>`_
   * - ASR
     - ASR with NeMo
     - `ASR with NeMo <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0b3/tutorials/asr/01_ASR_with_NeMo.ipynb>`_
   * - ASR
     - Speech Commands
     - `Speech commands <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0b3/tutorials/asr/03_Speech_Commands.ipynb>`_
   * - ASR
     - Speaker Recognition and Verification
     - `Speaker Recognition and Verification <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0b3/tutorials/speaker_recognition/Speaker_Recognition_Verification.ipynb>`_
   * - ASR
     - Online Noise Augmentation
     - `Online noise augmentation <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0b3/tutorials/asr/05_Online_Noise_Augmentation.ipynb>`_
   * - ASR
     - Beam Search and External Language Model Rescoring
     - `Beam search and external language model rescoring <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0b3/tutorials/asr/Offline_ASR.ipynb>`_
   * - NLP
     - Using Pretrained Language Models for Downstream Tasks
     - `Pretrained language models for downstream tasks <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0b3/tutorials/nlp/01_Pretrained_Language_Models_for_Downstream_Tasks.ipynb>`_
   * - NLP
     - Exploring NeMo NLP Tokenizers
     - `NLP tokenizers <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0b3/tutorials/nlp/02_NLP_Tokenizers.ipynb>`_
   * - NLP
     - Text Classification (Sentiment Analysis) with BERT
     - `Text Classification (Sentiment Analysis) <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0b3/tutorials/nlp/Text_Classification_Sentiment_Analysis.ipynb>`_
   * - NLP
     - Question answering with SQuAD
     - `Question answering Squad <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0b3/tutorials/nlp/Question_Answering_Squad.ipynb>`_
   * - NLP
     - Token Classification (Named Entity Recognition)
     - `Token classification: named entity recognition <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0b3/tutorials/nlp/Token_Classification_Named_Entity_Recognition.ipynb>`_
   * - NLP
     - Joint Intent Classification and Slot Filling
     - `Joint Intent and Slot Classification <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0b3/tutorials/nlp/Joint_Intent_and_Slot_Classification.ipynb>`_
   * - NLP
     - GLUE Benchmark
     - `GLUE benchmark <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0b3/tutorials/nlp/GLUE_Benchmark.ipynb>`_
   * - NLP
     - Punctuation and Capitialization
     - `Punctuation and capitalization <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0b3/tutorials/nlp/Punctuation_and_Capitalization.ipynb>`_
   * - NLP
     - Named Entity Recognition - BioMegatron
     - `Named Entity Recognition - BioMegatron <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0b3/tutorials/nlp/Token_Classification-BioMegatron.ipynb>`_
   * - NLP
     - Relation Extraction - BioMegatron
     - `Relation Extraction - BioMegatron <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0b3/tutorials/nlp/Relation_Extraction-BioMegatron.ipynb>`_
   * - TTS
     - Speech Synthesis
     - `TTS inference <https://colab.research.google.com/github/NVIDIA/NeMo/blob/v1.0.0b4/tutorials/tts/1_TTS_inference.ipynb>`_
   * - TTS
     - Speech Synthesis
     - `Tacotron2 training <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0b4/tutorials/tts/2_TTS_Tacotron2_Training.ipynb>`_
   * - Tools
     - CTC Segmentation
     - `CTC Segmentation <https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/tools/CTC_Segmentation_Tutorial.ipynb>`_
   * - Tools
     - Text Normalization for Text To Speech
     - `Text Normalization <https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/tools/Text_Normalization_Tutorial.ipynb>`_

Contributing
------------

We welcome community contributions! Please refer to the  `CONTRIBUTING.md <https://github.com/NVIDIA/NeMo/blob/main/CONTRIBUTING.md>`_ CONTRIBUTING.md for the process.

License
-------
NeMo is under `Apache 2.0 license <https://github.com/NVIDIA/NeMo/blob/main/LICENSE>`_.