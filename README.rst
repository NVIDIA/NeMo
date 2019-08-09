NeMo: Neural Modules Toolkit
============================

.. image:: docs/_images/nemo-icon-256x256.png
    :width: 40
    :align: center
    :alt: NEMO

Neural Modules (NeMo) is a framework-agnostic toolkit for building AI applications powered by Neural Modules.
Current support is for PyTorch framework.

A "Neural Module" is a block of code that computes a set of outputs from a set of inputs.

Neural Modules’ inputs and outputs have Neural Type for semantic checking.

An application built with NeMo application is a Directed Acyclic Graph(DAG) of connected modules enabling researchers to define and build new speech and nlp networks easily through API Compatible modules.


**VIDEO**

`A Short VIDEO walk-through about using NEMO to experiment with ASR systems. <https://drive.google.com/file/d/1CF-buP_Y1qCAefzoyvOUCXl_3v2vO5P-/view?usp=sharing>`_


**Core Concepts and Features**

* `NeuralModule` class - represents and implements a neural module.
* `NmTensor` - represents activation flow between neural modules' ports.
* `NeuralType` - represents types of modules' ports and NmTensors.
* `NeuralFactory` - to create neural modules and manage training.
* **Lazy execution** - when describing activation flow between neural modules, nothing happens until an "action" (such as `optimizer.optimize(...)` is called.
* **Collections** - NeMo comes with collections - related group of modules such as `nemo_asr` (for Speech Recognition) and `nemo_nlp` for NLP


**Documentation**

Please refer to the HTML documentation in the `docs` folder


**Requirements**

1) Python 3.6 or 3.7
2) Pytorch >=1.0 with GPU support
3) NVIDIA APEX: https://github.com/NVIDIA/apex
4) (for `nemo_asr` do: `apt-get install libsndfile1`)


**Unittests**

Run this:

.. code-block:: bash

    ./reinstall.sh
    python -m unittest tests/*.py


**Getting started**

1) Clone the repository and run unittests
2) Go to `nemo` folder and do: `python setup.py install`
3) Install collections:
    a) ASR collection from `collections/nemo_asr` do: `python setup.py install`
    b) NLP collection coming soon ...

4) For development do: `python setup.py develop` instead of `python setup.py install` in Step (3) above
5) Go to `examples/start_here` to get started with few simple examples
6) To get started with speech recognition:
    a) head to the ASR tutorial in the documentation
    b) head to `examples/asr/ASR_made_simple.ipynb`
