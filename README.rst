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

NeMo (Neural Modules) is a toolkit for creating AI applications built around **neural modules**, conceptual blocks of neural networks that take *typed* inputs and produce *typed* outputs. Such modules typically represent data layers, encoders, decoders, language models, loss functions, or methods of combining activations.

NeMo makes it easy to combine and re-use these building blocks while providing a level of semantic correctness checking via its neural type system. As long as two modules have compatible inputs and outputs, it is legal to chain them together. An application built with NeMo is a **Directed Acyclic Graph** (DAG) of connected modules.

NeMo's API is designed to be **framework-agnostic**, but currently only PyTorch is supported.

NeMo follows a **lazy execution** model: no computation is done until an action (such as `optimizer.optimize(...)` is called.

The toolkit comes with extendable collections of pre-built modules for automatic speech recognition (ASR) and natural language processing (NLP). Furthermore, NeMo provides built-in support for **distributed training** and **mixed precision** on the latest NVIDIA GPUs.

NeMo consists of: 

* **NeMo Core**: fundamental building blocks for all neural models and type system.
* **NeMo collections**: pre-built neural modules for particular domains such as automatic speech recognition (nemo_asr) and natural language processing (nemo_nlp).


**Introduction**

See `this video <https://nvidia.github.io/NeMo/>`_ for a quick walk-through.


**Requirements**

1) Python 3.6 or 3.7
2) PyTorch 1.2 with GPU support
3) NVIDIA APEX. Install from here: https://github.com/NVIDIA/apex


**Documentation**

`NeMo documentation <https://nvidia.github.io/NeMo/>`_

See `examples/start_here` to get started with the simplest example. The folder `examples` contains several examples to get you started with various tasks in NLP and ASR.


**Getting started**

You can use our `NGC PyTorch container <https://ngc.nvidia.com/catalog/containers/nvidia:pytorch>`_ which already includes all the requirements above.

* Pull the docker: ``docker pull nvcr.io/nvidia/pytorch:19.10-py3``
* Run: ``nvidia-docker run -it --rm -v <nemo_github_folder>:/NeMo --shm-size=1g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:19.10-py3``
* (If your docker version is >=19.03) Run: ``docker run --runtime=nvidia -it --rm -v <nemo_github_folder>:/NeMo --shm-size=1g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:19.10-py3``
* ``cd /NeMo``

and then continue with the following steps.

If you have all requirements installed (or are using `NGC PyTorch container <https://ngc.nvidia.com/catalog/containers/nvidia:pytorch>`_ ),
then you can simply use pip to install the latest released version (**currently 0.8.1**) of NeMo and its collections:

.. code-block:: bash

    pip install nemo-toolkit  # install NeMo Core
    pip install nemo-asr # installs NeMo ASR collection
    pip install nemo-nlp # installs NeMo NLP collection
    
**Tutorials**

* `Speech recognition <https://nvidia.github.io/NeMo/asr/intro.html>`_
* `Natural language processing <https://nvidia.github.io/NeMo/nlp/intro.html>`_

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

**Unittests**

This command runs unittests:

.. code-block:: bash

    ./reinstall.sh
    python -m unittest tests/*.py


Citation
~~~~~~~~

If you are using NeMo please cite the following publication

@misc{nemo2019,
    title={NeMo: a toolkit for building AI applications using Neural Modules},
    author={Oleksii Kuchaiev and Jason Li and Huyen Nguyen and Oleksii Hrinchuk and Ryan Leary and Boris Ginsburg and Samuel Kriman and Stanislav Beliaev and Vitaly Lavrukhin and Jack Cook and Patrice Castonguay and Mariya Popova and Jocelyn Huang and Jonathan M. Cohen},
    year={2019},
    eprint={1909.09577},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
