NVIDIA Neural Modules: NeMo
===========================

.. toctree::
   :hidden:
   :maxdepth: 2

   Introduction <self>
   installation
   tutorials/intro
   training
   asr/intro
   nlp/intro
   tts/intro
   collections/modules
   api-docs/modules



Neural Modules (NeMo) is a framework-agnostic toolkit for building AI applications powered by Neural Modules. Current support is for PyTorch framework.

A "Neural Module" is a block of code that computes a set of outputs from a set of inputs.

Neural Modules’ inputs and outputs have Neural Type for semantic checking.

An application built with NeMo application is a Directed Acyclic Graph (DAG) of connected modules enabling researchers to define and build new speech and nlp networks easily through API Compatible modules.


**Introduction**

See this video for a walk-through.

.. raw:: html

    <div>
        <iframe src="https://drive.google.com/file/d/1CF-buP_Y1qCAefzoyvOUCXl_3v2vO5P-/preview" width="640" height="480"></iframe>
    </div>


**Core Concepts and Features**

* `NeuralModule` class - represents and implements a neural module.
* `NmTensor` - represents activations which flow between neural modules' ports.
* `NeuralType` - represents types of modules' ports and NmTensors.
* `NeuralFactory` - to create neural modules and manage training.
* `Lazy execution` - when describing activation flow between neural modules, nothing happens until an "action" (such as `optimizer.optimize(...)` is called.
* `Collections` - NeMo comes with collections - related group of modules such as `nemo_asr` (for Speech Recognition) and `nemo_nlp` for NLP


**Requirements**

1) Python 3.6 or 3.7
2) PyTorch 1.2 with GPU support
3) NVIDIA APEX: https://github.com/NVIDIA/apex


**Getting started**

You can use NVIDIA `NGC PyTorch container <https://ngc.nvidia.com/catalog/containers/nvidia:pytorch>`_ which already includes all the requirements above.

.. code-block:: bash

    # Pull the docker
    docker pull nvcr.io/nvidia/pytorch:19.10-py3

    # Do one of the two following commands
    # Run Docker for docker version <19.03
    nvidia-docker run -it --rm -v <nemo_github_folder>:/NeMo --shm-size=1g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:19.10-py3
    
    # Run Docker for docker version >=19.03
    docker run --runtime=nvidia -it --rm -v <nemo_github_folder>:/NeMo --shm-size=1g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:19.10-py3

    cd /NeMo

and then continue with the following steps.

If you have all requirements installed (or are using `NGC PyTorch container <https://ngc.nvidia.com/catalog/containers/nvidia:pytorch>`_ ),
then you can simply use pip to install the latest released version (**currently 0.8.2**) of NeMo and its collections:

.. code-block:: bash

    pip install nemo-toolkit  # installs NeMo Core
    pip install nemo-asr # installs NeMo ASR collection
    pip install nemo-nlp # installs NeMo NLP collection
    pip install nemo-tts # installs NeMo TTS collection

**Tutorials**

* `Speech recognition <https://nvidia.github.io/NeMo/asr/intro.html>`_
* `Natural language processing <https://nvidia.github.io/NeMo/nlp/intro.html>`_
* `Speech synthesis <https://nvidia.github.io/NeMo/tts/intro.html>`_

**Installing From Github**

If you prefer to use NeMo's latest development version (from GitHub) follow the steps below:

*Note*: For step 2 and 3, if you want to use NeMo in development mode, use: ``pip install -e .`` instead of ``pip install .``

1) Clone the repository ``git clone https://github.com/NVIDIA/NeMo.git``
2) Go to NeMo folder and install the toolkit:

.. code-block:: bash

	cd NeMo/nemo
	pip install .

3) Install the collection(s) you want.

.. code-block:: bash

    # Install the ASR collection from collections/nemo_asr
    apt-get install libsndfile1
    cd NeMo/collections/nemo_asr
    pip install .

    # Install the NLP collection from collections/nemo_nlp
    cd NeMo/collections/nemo_nlp
    pip install .

    # Install the TTS collection from collections/nemo_tts
    cd NeMo/collections/nemo_tts
    pip install .

**Unittests**

This command runs unittests:

.. code-block:: bash

    ./reinstall.sh
    python -m unittest tests/*.py
