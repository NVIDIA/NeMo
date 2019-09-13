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

Neural Modules (NeMo) is a framework-agnostic toolkit for building AI applications powered by Neural Modules. Current support is for PyTorch framework.

A "Neural Module" is a block of code that computes a set of outputs from a set of inputs.

Neural Modules’ inputs and outputs have Neural Type for semantic checking.

An application built with NeMo application is a Directed Acyclic Graph(DAG) of connected modules enabling researchers to define and build new speech and nlp networks easily through API Compatible modules.


**Introduction**

See this `video for a walk-through. <https://nvidia.github.io/NeMo/>`_


**Core Concepts and Features**

* `NeuralModule` class - represents and implements a neural module.
* `NmTensor` - represents activations which flow between neural modules' ports.
* `NeuralType` - represents types of modules' ports and NmTensors.
* `NeuralFactory` - to create neural modules and manage training.
* `Lazy execution` - when describing activation flow between neural modules, nothing happens until an "action" (such as `optimizer.optimize(...)` is called.
* `Collections` - NeMo comes with collections - related group of modules such as `nemo_asr` (for Speech Recognition) and `nemo_nlp` for NLP


**Requirements**

1) Python 3.6 or 3.7
2) Pytorch 1.2 with GPU support
3) NVIDIA APEX: https://github.com/NVIDIA/apex


**Documentation**
`NeMo documentation <https://nvidia.github.io/NeMo/>`_


**Getting started**

If desired, you can start with `NGC PyTorch container <https://ngc.nvidia.com/catalog/containers/nvidia:pytorch>`_ which already includes
requirements above.

* You can pull it like so: ``docker pull nvcr.io/nvidia/pytorch:19.08-py3``
* And then run: ``nvidia-docker run -it --rm -v <nemo_github_folder>:/NeMo --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:19.08-py3``
* ``cd /NeMo''

and then continue with the following steps:


1) Clone the repository
2) Go to nemo folder and then: ``python setup.py install``
3) Install collections:
    * ASR collection from `collections/nemo_asr`: 
        1. ``apt-get install libsndfile1``
        2. ``python setup.py install``
        
    * NLP collection from `collections/nemo_nlp`: ``python setup.py install``
4) For development you will need to: ``python setup.py develop`` instead of ``python setup.py install`` in Step (3.2) above
5) Go to `examples/start_here` to get started with few simple examples


**Tutorials**

* `Speech recognition <https://nvidia.github.io/NeMo/asr/intro.html>`_
* `Natural language processing <https://nvidia.github.io/NeMo/nlp/intro.html>`_


**Unittests**

This command runs unittests:

.. code-block:: bash

    ./reinstall.sh
    python -m unittest tests/*.py

