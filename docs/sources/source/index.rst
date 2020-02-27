NVIDIA Neural Modules Developer Guide
=====================================

.. toctree::
   :hidden:
   :maxdepth: 2

   Introduction <self>
   tutorials/intro
   training
   asr/intro
   speech_command/intro
   nlp/intro
   tts/intro
   collections/modules
   api-docs/modules
   chinese/intro


Neural Modules (NeMo) is a framework-agnostic toolkit for building AI applications powered by Neural Modules. Current support is for PyTorch framework.

A "Neural Module" is a block of code that computes a set of outputs from a set of inputs.

Neural Modules’ inputs and outputs have Neural Type for semantic checking.

An application built with NeMo is a Directed Acyclic Graph (DAG) of connected modules enabling researchers to define and build new speech and nlp networks easily through API Compatible modules.


Introduction
------------

See this video for a walk-through.

.. raw:: html

    <div>
        <iframe src="https://drive.google.com/file/d/1CF-buP_Y1qCAefzoyvOUCXl_3v2vO5P-/preview" width="640" height="480"></iframe>
    </div>


Core Concepts and Features
--------------------------

* `NeuralModule` class - represents and implements a neural module.
* `NmTensor` - represents activations which flow between neural modules' ports.
* `NeuralType` - represents types of modules' ports and NmTensors.
* `NeuralFactory` - to create neural modules and manage training.
* `Lazy execution` - when describing activation flow between neural modules, nothing happens until an "action" (such as `optimizer.optimize(...)` is called.
* `Collections` - NeMo comes with collections - related group of modules such as `nemo_asr` (for Speech Recognition) and `nemo_nlp` for NLP


Requirements
------------

1) Python 3.6 or 3.7
2) PyTorch 1.4 or later with GPU support
3) (optional for best performance) NVIDIA APEX: https://github.com/NVIDIA/apex

.. _installation:

Getting started
---------------

You can use NVIDIA `NGC NeMo container <https://ngc.nvidia.com/catalog/containers/nvidia:nemo>`_ for the latest NeMo release and all dependencies.

.. code-block:: bash

    # Pull the docker
    docker pull nvcr.io/nvidia/nemo:v0.9

    # Do one of the two following commands
    # Run Docker for docker version <19.03
    nvidia-docker run -it --rm --shm-size=8g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/nemo:v0.9
    
    # Run Docker for docker version >=19.03
    docker run -it --rm --gpus all --shm-size=8g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/nemo:v0.9

and begin using NeMo immediately.

If you have all requirements installed (or are using `NGC PyTorch container <https://ngc.nvidia.com/catalog/containers/nvidia:pytorch>`_ ),
then you can simply use pip to install the latest released version (**currently 0.9.0**) of NeMo and its collections:

.. code-block:: bash

    pip install nemo_toolkit[all] # Installs NeMo Core and all collections including nemo_asr, nemo_nlp, nemo_tts

Tutorials
---------

* `Speech recognition <https://nvidia.github.io/NeMo/asr/intro.html>`_
* `Natural language processing <https://nvidia.github.io/NeMo/nlp/intro.html>`_
* `Speech synthesis <https://nvidia.github.io/NeMo/tts/intro.html>`_

Installing From Github
----------------------

If you prefer to use NeMo's latest development version (from GitHub) follow the steps below:

1) Clone the repository ``git clone https://github.com/NVIDIA/NeMo.git``
2) Go to NeMo folder and install the toolkit and collections:

.. code-block:: bash

    ./reinstall.sh

.. note::
    reinstall.sh install NeMo in development mode.

Unittests
---------

This command runs unittests:

.. code-block:: bash

    ./reinstall.sh
    python -m unittest tests/*.py

Building Docker Container
-------------------------

The NeMo Docker image requires Docker Buildx which is included in Docker 19.03 and layer. To build a custom NeMo Docker image, run

.. code-block:: bash

    docker buildx build --build-arg NEMO_VERSION=$(git describe --tags) -t nemo .

The ``NEMO_VERSION`` build arg is required. We recommend always setting to ``git describe --tags`` so that the build is traceable and replicable.
At runtime, the value of ``NEMO_VERSION`` specified at build time is exposed as an environment variable.

You may also specify a build arg ``BASE_IMAGE`` to override the underlying version of PyTorch used, though there are no compatability guarantees.

For development purposes, you can also create a Docker image containing only NeMo's dependencies, and map your local development branch into the
container at runtime.

.. code-block:: bash

    # build the development container
    docker buildx build --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:20.01-py3 --target nemo-deps -t nemo-devel .

    # launch the container, mapping local nemo into it
    cd <nemo_path>
    docker run -it --rm --gpus all -v $(pwd):/workspace/nemo --shm-size=8g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/nemo:v0.9

    # install in development mode
    ./reinstall.sh

