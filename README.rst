.. image:: http://www.repostatus.org/badges/latest/active.svg
	:target: http://www.repostatus.org/#active
	:alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.

.. image:: https://img.shields.io/badge/documentation-github.io-blue.svg
	:target: https://nvidia.github.io/NeMo/
	:alt: NeMo documentation on GitHub pages
   
.. image:: https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg
	:target: https://github.com/NVIDIA/NeMo/blob/master/LICENSE
	:alt: NeMo core license and license for collections in this repo


NVIDIA Neural Modules: NeMo
===========================

NeMo (**Ne**ural **Mo**dules) is a toolkit for creating AI applications built around **neural modules**, conceptual blocks of neural networks that take *typed* inputs and produce *typed* outputs. Such modules typically represent data layers, encoders, decoders, language models, loss functions, or methods of combining activations.

NeMo makes it easy to combine and re-use these building blocks while providing a level of semantic correctness checking via its neural type system. As long as two modules have compatible inputs and outputs, it is legal to chain them together. An application built with NeMo is a **Directed Acyclic Graph** (DAG) of connected modules.

NeMo's API is designed to be **framework-agnostic**, but currently only PyTorch is supported. We plan on supporting other frameworks in the near future.

NeMo follows a **lazy execution** model: no computation is done until an action (such as `optimizer.optimize(...)` is called.

The toolkit comes with extendable collections of pre-built modules for automatic speech recognition (ASR) and natural language processing (NLP). Furthermore, NeMo provides built-in support for **distributed training** and **mixed precision** on the latest NVIDIA GPUs.

NeMo consists of: 

* NeMo Core: fundamental building blocks for all neural models and type system.
* NeMo collections: pre-built neural modules for particular domains such as automatic speech recognition (nemo_asr) and natural language processing (nemo_nlp).


**Introduction**

See `this video <https://nvidia.github.io/NeMo/>`_ for a quick walk-through.


**Requirements**

1) Python 3.6 or 3.7
2) Pytorch 1.2 with GPU support
3) NVIDIA APEX. Install here: https://github.com/NVIDIA/apex


**Documentation**

`NeMo documentation <https://nvidia.github.io/NeMo/>`_

See `examples/start_here` to get started with the simplest example. The folder `examples` contains several examples to get you started with various tasks in NLP and ASR.


**Getting started**

You can use our `NGC PyTorch container <https://ngc.nvidia.com/catalog/containers/nvidia:pytorch>`_ which already includes all the requirements above.

* Pull the docker: ``docker pull nvcr.io/nvidia/pytorch:19.08-py3``
* Run: ``nvidia-docker run -it --rm -v <nemo_github_folder>:/NeMo --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:19.08-py3``
* ``cd /NeMo``

and then continue with the following steps.

** Note**

For step 2 and 3, if you want to use NeMo in development mode, use:

``pip install -e .``

instead of 

``pip install .``

1) Clone the repository ``git clone https://github.com/NVIDIA/NeMo.git``
2) Go to NeMo folder and install the toolkit:

.. code-block:: bash

	cd NeMo
	pip install .

3) Install the collection you want.
	
	* Install the ASR collection from `collections/nemo_asr`: 
        1. ``apt-get install libsndfile1``
        2. ``cd collections/nemo_asr``
        3. ``pip install .``
        
    * Install the NLP collection from `collections/nemo_nlp`:
    	1. ``apt-get install libsndfile1``
        2. ``cd collections/nemo_nlp``
        3. ``pip install .``


**Tutorials**

* `Speech recognition <https://nvidia.github.io/NeMo/asr/intro.html>`_
* `Natural language processing <https://nvidia.github.io/NeMo/nlp/intro.html>`_


**Unittests**

This command runs unittests:

.. code-block:: bash

    ./reinstall.sh
    python -m unittest tests/*.py

