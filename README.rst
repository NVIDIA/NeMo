 
|status| |license| |lgtm_grade| |lgtm_alerts| |black|

.. |status| image:: http://www.repostatus.org/badges/latest/active.svg
  :target: http://www.repostatus.org/#active
  :alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.


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

NeMo toolkit makes it possible for researchers to easily compose complex neural network architectures for conversational AI using reusable components - Neural Modules.
**Neural Modules** are conceptual blocks of neural networks that take *typed* inputs and produce *typed* outputs. Such modules typically represent data layers, encoders, decoders, language models, loss functions, or methods of combining activations.


The toolkit comes with extendable collections of pre-built modules and ready-to-use models for:

* `Automatic Speech Recognition (ASR) <https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels>`_
* `Natural Language Processing (NLP) <https://ngc.nvidia.com/catalog/models/nvidia:nemonlpmodels>`_
* `Speech synthesis, or Text-To-Speech (TTS) <https://ngc.nvidia.com/catalog/models/nvidia:nemottsmodels>`_

Built for speed, NeMo can utilize NVIDIA's Tensor Cores and scale out training to multiple GPUs and multiple nodes.

`NeMo product page. <https://developer.nvidia.com/nvidia-nemo>`_

`Introductory video. <https://www.youtube.com/embed/wBgpMf_KQVw>`_

.. raw:: html

    <div style="position: relative; padding-bottom: 3%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe width="560" height="315" src="https://www.youtube.com/embed/wBgpMf_KQVw" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </div>


Requirements
------------

NeMo's works with:

1) Python 3.6 or 3.7
2) Pytorch 1.6 or above

Docker containers:
~~~~~~~~~~~~~~~~~~
The easiest way to start training with NeMo is by using `NeMo's container <https://ngc.nvidia.com/catalog/containers/nvidia:nemo>`_.

It has all requirements and NeMo 1.0.0b2 already installed.

.. code-block:: bash

    docker run --gpus all -it --rm -v <nemo_github_folder>:/NeMo --shm-size=8g \
    -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit \
    stack=67108864 --device=/dev/snd nvcr.io/nvidia/nemo:v1.0.0b3


If you chose to work with main branch, we recommend using NVIDIA's PyTorch container version 20.09-py3.

.. code-block:: bash

    docker run --gpus all -it --rm -v <nemo_github_folder>:/NeMo --shm-size=8g \
    -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit \
    stack=67108864 --device=/dev/snd nvcr.io/nvidia/pytorch:20.09-py3


Installation
~~~~~~~~~~~~
If you are not inside the NVIDIA docker container, please install Cython first. If you wish to either use the ASR or TTS collection, please install libsndfile1 and ffmpeg as well.

* ``pip install Cython``
* ``apt-get update && apt-get install -y libsndfile1 ffmpeg`` (If you want to install the TTS or ASR collections)

Once requirements are satisfied, simply install using pip:

* ``pip install nemo_toolkit[all]==1.0.0b2`` (latest version)

Or if you want the latest (or particular) version from GitHub:

* ``python -m pip install git+https://github.com/NVIDIA/NeMo.git@{BRANCH}#egg=nemo_toolkit[all]`` - where {BRANCH} should be replaced with the branch you want. This is recommended route if you are testing out the latest WIP version of NeMo.
* ``./reinstall.sh`` - from NeMo's git root. This will install the version from current branch in developement mode.

Examples
~~~~~~~~
``<nemo_github_folder>/examples/`` folder contains various example scripts. Many of them look very similar and have the same arguments because
we used `Facebook's Hydra <https://github.com/facebookresearch/hydra>`_ for configuration.

Here is an example command which trains ASR model (QuartzNet15x5) on LibriSpeech, using 4 GPUs and mixed precision training.
(It assumes you are inside the container with NeMo installed)

.. code-block:: bash

    root@987b39669a7e:/NeMo# python examples/asr/speech_to_text.py --config-name=quartznet_15x5 \
    model.train_ds.manifest_filepath=<PATH_TO_DATA>/librispeech-train-all.json \
    model.validation_ds.manifest_filepath=<PATH_TO_DATA>/librispeech-dev-other.json \
    trainer.gpus=4 trainer.max_epochs=128 model.train_ds.batch_size=64 \
    +trainer.precision=16 +trainer.amp_level=O1  \
    +model.validation_ds.num_workers=16  \
    +model.train_ds.num_workers=16 \
    +model.train_ds.pin_memory=True

    #(Optional) Tensorboard:
    tensorboard --bind_all --logdir nemo_experiments



Documentation
-------------

.. |main| image:: https://readthedocs.com/projects/nvidia-nemo/badge/?version=main
  :alt: Documentation Status
  :scale: 100%
  :target: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/

.. |latest| image:: https://readthedocs.com/projects/nvidia-nemo/badge/?version=main
  :alt: Documentation Status
  :scale: 100%
  :target: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/

.. |stable| image:: https://readthedocs.com/projects/nvidia-nemo/badge/?version=stable
  :alt: Documentation Status
  :scale: 100%
  :target: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/

.. |v0111| image:: https://readthedocs.com/projects/nvidia-nemo/badge/?version=v0.11.1
  :alt: Documentation Status
  :scale: 100%
  :target: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/v0.11.1/

.. |v0110| image:: https://readthedocs.com/projects/nvidia-nemo/badge/?version=v0.11.0
  :alt: Documentation Status
  :scale: 100%
  :target: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/v0.11.0/



+---------+----------+---------------------------------------------------------+
| Version | Status   | Description                                             |
+=========+==========+=========================================================+
| Latest  | |latest| | Documentation of the latest (i.e. `main`) branch        |
+---------+----------+---------------------------------------------------------+
| Stable  | |stable| | Documentation of the stable (i.e. `0.11.1`) branch      |
+---------+----------+---------------------------------------------------------+
| Main    | |main|   | Documentation of the `main` branch                      |
+---------+----------+---------------------------------------------------------+
| v0.11.1 | |v0111|  | Documentation of the v0.11.1 release                    |
+---------+----------+---------------------------------------------------------+
| v0.11.0 | |v0110|  | Documentation of the v0.11.0 release                    |
+---------+----------+---------------------------------------------------------+


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
     - `TTS inference <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0b3/tutorials/tts/1_TTS_inference.ipynb>`_

   * - Tools
     - CTC Segmentation
     - `CTC Segmentation <https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/tools/CTC_Segmentation_Tutorial.ipynb>`_
   * - Tools
     - Text Normalization for Text To Speech
     - `Text Normalization <https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/tools/Text_Normalization_Tutorial.ipynb>`_

Contributing
------------

We welcome community contributions! Please refer to the CONTRIBUTING.md for the process.

License
-------
NeMo is under Apache 2.0 license.
