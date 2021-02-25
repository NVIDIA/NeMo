
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

**NVIDIA NeMo**
===============

Introduction
------------

NeMo is a toolkit for creating `Conversational AI <https://developer.nvidia.com/conversational-ai#started>`_ applications.

`NeMo product page. <https://developer.nvidia.com/nvidia-nemo>`_

`Introductory video. <https://www.youtube.com/embed/wBgpMf_KQVw>`_

The toolkit comes with extendable collections of pre-built modules and ready-to-use models for:

* `Automatic Speech Recognition (ASR) <https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels>`_
* `Natural Language Processing (NLP) <https://ngc.nvidia.com/catalog/models/nvidia:nemonlpmodels>`_
* `Speech synthesis, or Text-To-Speech (TTS) <https://ngc.nvidia.com/catalog/models/nvidia:nemottsmodels>`_

Built for speed, NeMo can utilize NVIDIA's Tensor Cores and scale out training to multiple GPUs and multiple nodes.

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
It has all requirements and NeMo 1.0.0rc1 already installed.

.. code-block:: bash

    docker run --gpus all -it --rm --shm-size=8g \
    -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit \
    stack=67108864 --device=/dev/snd nvcr.io/nvidia/nemo:1.0.0rc1


If you chose to work with main branch, we recommend using NVIDIA's PyTorch container version 20.11-py3 and then installing from GitHub.

.. code-block:: bash

    docker run --gpus all -it --rm -v <nemo_github_folder>:/NeMo --shm-size=8g \
    -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit \
    stack=67108864 --device=/dev/snd nvcr.io/nvidia/pytorch:20.11-py3

Examples
--------

`Simplest application with NeMo. <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0rc1/tutorials/NeMo_voice_swap_app.ipynb>`_ (runs in Google Colab, no local installation necessary)

Lots of other examples in `"Examples" folder. <https://github.com/NVIDIA/NeMo/tree/main/examples>`_


Documentation
-------------

.. |main| image:: https://readthedocs.com/projects/nvidia-nemo/badge/?version=main
  :alt: Documentation Status
  :scale: 100%
  :target: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/

.. |v1.0.0b1| image:: https://readthedocs.com/projects/nvidia-nemo/badge/?version=v1.0.0b1
  :alt: Documentation Status
  :scale: 100%
  :target: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/v1.0.0b1/

+---------+------------+----------------------------------------------------------------------------------------------------------------------------------+
| Version | Status     | Description                                                                                                                      |
+=========+============+==================================================================================================================================+
| Latest  | |main|     | `Documentation of the latest (i.e. main) branch <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/>`_           |
+---------+------------+----------------------------------------------------------------------------------------------------------------------------------+
| Stable  | |v1.0.0b1| | `Documentation of the stable (i.e. v1.0.0b1) branch <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/v1.0.0b1/>`_   |
+---------+------------+----------------------------------------------------------------------------------------------------------------------------------+

Getting help with NeMo
----------------------
FAQ can be found on NeMo's `Discussions board <https://github.com/NVIDIA/NeMo/discussions>`_. You are welcome to ask questions or start discussions there.

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
     - `Voice swap app <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0rc1/tutorials/NeMo_voice_swap_app.ipynb>`_
   * - NeMo
     - Exploring NeMo Fundamentals
     - `NeMo primer <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0rc1/tutorials/00_NeMo_Primer.ipynb>`_
   * - NeMo Models
     - Exploring NeMo Model Construction
     - `NeMo models <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0rc1/tutorials/01_NeMo_Models.ipynb>`_
   * - ASR
     - ASR with NeMo
     - `ASR with NeMo <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0rc1/tutorials/asr/01_ASR_with_NeMo.ipynb>`_
   * - ASR
     - ASR with Subword Tokenization
     - `ASR with Subword Tokenization <https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/asr/08_ASR_with_Subword_Tokenization.ipynb>`_
   * - ASR
     - Speech Commands
     - `Speech commands <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0rc1/tutorials/asr/03_Speech_Commands.ipynb>`_
   * - ASR
     - Speaker Recognition and Verification
     - `Speaker Recognition and Verification <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0rc1/tutorials/speaker_recognition/Speaker_Recognition_Verification.ipynb>`_
   * - ASR
     - Online Noise Augmentation
     - `Online noise augmentation <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0rc1/tutorials/asr/05_Online_Noise_Augmentation.ipynb>`_
   * - ASR
     - Beam Search and External Language Model Rescoring
     - `Beam search and external language model rescoring <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0rc1/tutorials/asr/Offline_ASR.ipynb>`_
   * - NLP
     - Using Pretrained Language Models for Downstream Tasks
     - `Pretrained language models for downstream tasks <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0rc1/tutorials/nlp/01_Pretrained_Language_Models_for_Downstream_Tasks.ipynb>`_
   * - NLP
     - Exploring NeMo NLP Tokenizers
     - `NLP tokenizers <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0rc1/tutorials/nlp/02_NLP_Tokenizers.ipynb>`_
   * - NLP
     - Text Classification (Sentiment Analysis) with BERT
     - `Text Classification (Sentiment Analysis) <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0rc1/tutorials/nlp/Text_Classification_Sentiment_Analysis.ipynb>`_
   * - NLP
     - Question answering with SQuAD
     - `Question answering Squad <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0rc1/tutorials/nlp/Question_Answering_Squad.ipynb>`_
   * - NLP
     - Token Classification (Named Entity Recognition)
     - `Token classification: named entity recognition <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0rc1/tutorials/nlp/Token_Classification_Named_Entity_Recognition.ipynb>`_
   * - NLP
     - Joint Intent Classification and Slot Filling
     - `Joint Intent and Slot Classification <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0rc1/tutorials/nlp/Joint_Intent_and_Slot_Classification.ipynb>`_
   * - NLP
     - GLUE Benchmark
     - `GLUE benchmark <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0rc1/tutorials/nlp/GLUE_Benchmark.ipynb>`_
   * - NLP
     - Punctuation and Capitialization
     - `Punctuation and capitalization <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0rc1/tutorials/nlp/Punctuation_and_Capitalization.ipynb>`_
   * - NLP
     - Named Entity Recognition - BioMegatron
     - `Named Entity Recognition - BioMegatron <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0rc1/tutorials/nlp/Token_Classification-BioMegatron.ipynb>`_
   * - NLP
     - Relation Extraction - BioMegatron
     - `Relation Extraction - BioMegatron <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0rc1/tutorials/nlp/Relation_Extraction-BioMegatron.ipynb>`_
   * - TTS
     - Speech Synthesis
     - `TTS inference <https://colab.research.google.com/github/NVIDIA/NeMo/blob/v1.0.0b4/tutorials/tts/1_TTS_inference.ipynb>`_
   * - TTS
     - Speech Synthesis
     - `Tacotron2 training <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0b4/tutorials/tts/2_TTS_Tacotron2_Training.ipynb>`_
   * - Tools
     - CTC Segmentation
     - `CTC Segmentation <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0rc1/tutorials/tools/CTC_Segmentation_Tutorial.ipynb>`_
   * - Tools
     - Text Normalization for Text To Speech
     - `Text Normalization <https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/tools/Text_Normalization_Tutorial.ipynb>`_

Contributing
------------

We welcome community contributions! Please refer to the  `CONTRIBUTING.md <https://github.com/NVIDIA/NeMo/blob/main/CONTRIBUTING.md>`_ CONTRIBUTING.md for the process.

License
-------
NeMo is under `Apache 2.0 license <https://github.com/NVIDIA/NeMo/blob/main/LICENSE>`_.
