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

Requirements
------------

NeMo's main requirements are:

1) Python 3.6 or above
2) Pytorch 1.7.1 or above

Installation
~~~~~~~~~~~~
``apt-get update && apt-get install -y libsndfile1 ffmpeg && pip install Cython && pip install nemo_toolkit[all]==version``

We recommend using NVIDIA's PyTorch container

.. code-block:: bash

    docker run --gpus all -it --rm -v <nemo_github_folder>:/NeMo --shm-size=8g \
    -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit \
    stack=67108864 nvcr.io/nvidia/pytorch:20.11-py3


