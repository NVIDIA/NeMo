 
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


The toolkit comes with extendable collections of pre-built modules and ready-to-use models for automatic speech recognition (ASR), natural language processing (NLP) and text synthesis (TTS).
Built for speed, NeMo can utilize NVIDIA's Tensor Cores and scale out training to multiple GPUs and multiple nodes.


Requirements
------------

NeMo's works with:

1) Python 3.6 or 3.7
2) Pytorch 1.6 or above

Docker container:
~~~~~~~~~~~~~~~~~
We recommend using NVIDIA's PyTorch container version 20.06-py3 with NeMo's main branch.

.. code-block:: bash

    docker run --gpus all -it --rm -v <nemo_github_folder>:/NeMo --shm-size=8g \
    -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit \
    stack=67108864 nvcr.io/nvidia/pytorch:20.06-py3

Installation
~~~~~~~~~~~~
Once requirements are satisfied (or you are inside NVIDIA docker container), simply install using pip:

``pip install nemo_toolkit[all]==version``

* ``pip install nemo_toolkit[all]`` - latest released version (currently 0.11.0)
* ``python -m pip install git+https://github.com/NVIDIA/NeMo.git@{BRANCH}#egg=nemo_toolkit[nlp]`` - where {BRANCH} should be replaced with the branch you want. This is recommended route if you are testing out the latest WIP version of NeMo.


Documentation
-------------
* `Main <https://docs.nvidia.com/deeplearning/nemo/developer_guide/en/main/>`_
* `Latest released (0.11.0) <https://docs.nvidia.com/deeplearning/nemo/developer_guide/en/v0.11.0/>`_

Tutorials
---------
The best way to get started with NeMo is to checkout one of our tutorials.

Most NeMo tutorials can be run on `Google's Colab <https://colab.research.google.com/notebooks/intro.ipynb>`_.

To run tutorials:

1. Click on Colab link (see table below)
3. Connect to an instance with a GPU (Runtime -> Change runtime type -> select "GPU" for hardware accelerator)

.. list-table:: NeMo tutorials
   :widths: 15 25 25
   :header-rows: 1

   * - Domain
     - Title
     - GitHub URL
   * - ASR
     - ASR with NeMo
     - `01_ASR_with_NeMo.ipynb <https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/asr/01_ASR_with_NeMo.ipynb>`_
   * - ASR
     - Speech Commands
     - `02_Speech_Commands.ipynb <https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/asr/02_Speech_Commands.ipynb>`_
   * - ASR
     - Online Noise Augmentation
     - `05_Online_Noise_Augmentation.ipynb <https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/asr/05_Online_Noise_Augmentation.ipynb>`_
   * - NLP
     - Token Classification (Named Entity Recognition)
     - `Token_Classification_Named_Entity_Recognition_tutorial.ipynb <https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/nlp/Token_Classification_Named_Entity_Recognition_tutorial.ipynb>`_
   * - NLP
     - Punctuation and Capitialization
     - `Punctuation_and_Capitalization.ipynb <https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/nlp/Punctuation_and_Capitalization.ipynb>`_

Contributing
------------

We welcome community contributions! Please refer to the CONTRIBUTING.md for the process.

License
-------
NeMo is under Apache 2.0 license.