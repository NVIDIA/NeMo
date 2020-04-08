.. image:: http://www.repostatus.org/badges/latest/active.svg
  :target: http://www.repostatus.org/#active
  :alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.

.. image:: https://img.shields.io/badge/documentation-github.io-blue.svg
  :target: https://nvidia.github.io/NeMo/
  :alt: NeMo documentation on GitHub pages

.. image:: https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg
  :target: https://github.com/NVIDIA/NeMo/blob/master/LICENSE
  :alt: NeMo core license and license for collections in this repo

.. image:: https://img.shields.io/lgtm/grade/python/g/NVIDIA/NeMo.svg?logo=lgtm&logoWidth=18
  :target: https://lgtm.com/projects/g/NVIDIA/NeMo/context:python
  :alt: Language grade: Python

.. image:: https://img.shields.io/lgtm/alerts/g/NVIDIA/NeMo.svg?logo=lgtm&logoWidth=18
  :target: https://lgtm.com/projects/g/NVIDIA/NeMo/alerts/
  :alt: Total alerts

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
  :target: https://github.com/psf/black
  :alt: Code style: black



NVIDIA Neural Modules: NeMo
===========================

NeMo is a toolkit for creating `Conversational AI <https://developer.nvidia.com/conversational-ai#started>`_ applications.

NeMo toolkit makes it possible for researchers to easily compose complex neural network architectures for conversational AI using reusable components - Neural Modules.
**Neural Modules** are conceptual blocks of neural networks that take *typed* inputs and produce *typed* outputs. Such modules typically represent data layers, encoders, decoders, language models, loss functions, or methods of combining activations.

The toolkit comes with extendable collections of pre-built modules for automatic speech recognition (ASR), natural language processing (NLP) and text synthesis (TTS).

Built for speed, NeMo can utilize NVIDIA's Tensor Cores and scale out training to multiple GPUs and multiple nodes.

**Introduction**

* Watch `this video <https://drive.google.com/a/nvidia.com/file/d/1AcOmtx4n1BAWvPoyhE0thcQXdloGWb6q/view?usp=sharing>`_ for a quick walk-through.

* `Documentation (latest released version) <https://nvidia.github.io/NeMo/>`_ and `Documentation (master branch) <http://nemo-master-docs.s3-website.us-east-2.amazonaws.com/>`_

* Read NVIDIA `Developer Blog for example applications <https://devblogs.nvidia.com/how-to-build-domain-specific-automatic-speech-recognition-models-on-gpus/>`_

* Read NVIDIA `Developer Blog for QuartzNet ASR model <https://devblogs.nvidia.com/develop-smaller-speech-recognition-models-with-nvidias-nemo-framework/>`_

* Recommended version to install is **0.10.0** via ``pip install nemo-toolkit[all]``

* Recommended NVIDIA `NGC NeMo Toolkit container <https://ngc.nvidia.com/catalog/containers/nvidia:nemo>`_

* Pretrained models are available on NVIDIA `NGC Model repository <https://ngc.nvidia.com/catalog/models?orderBy=modifiedDESC&query=nemo&quickFilter=models&filters=>`_


Getting started
~~~~~~~~~~~~~~~

THE LATEST STABLE VERSION OF NeMo is **0.10.0** (Available via PIP).

**Requirements**

1) Python 3.6 or 3.7
2) PyTorch 1.4.* with GPU support
3) (optional, for best performance) NVIDIA APEX. Install from here: https://github.com/NVIDIA/apex


Docker containers
~~~~~~~~~~~~~~~~~

**NeMo docker container**

You can use NeMo's docker container with all dependencies pre-installed

.. code-block:: bash

    docker run --runtime=nvidia -it --rm -v --shm-size=16g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/nemo:v0.10


If you are using the NVIDIA `NGC PyTorch container <https://ngc.nvidia.com/catalog/containers/nvidia:pytorch>`_ follow these instructions

* Pull the docker: ``docker pull nvcr.io/nvidia/pytorch:20.01-py3``
* Run:``docker run --gpus all -it --rm -v <nemo_github_folder>:/NeMo --shm-size=8g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:20.01-py3``
* ``apt-get update && apt-get install -y libsndfile1``
* ``pip install nemo_toolkit`` Installs NeMo core only.
* ``pip install nemo_toolkit[all]`` Installs NeMo core and ALL collections
* ``pip install nemo_toolkit[asr]`` Installs NeMo core and ASR (Speech Recognition) collection
* ``pip install nemo_toolkit[nlp]`` Installs NeMo core and NLP (Natural Language Processing) collection
* ``pip install nemo_toolkit[tts]`` Installs NeMo core and TTS (Speech Synthesis) collection

See `examples/start_here` to get started with the simplest example.

**Tutorials**

* `Speech recognition <https://nvidia.github.io/NeMo/asr/intro.html>`_
* `Natural language processing <https://nvidia.github.io/NeMo/nlp/intro.html>`_
* `Speech Synthesis <https://nvidia.github.io/NeMo/tts/intro.html>`_

Pre-trained models
~~~~~~~~~~~~~~~~~~

+------------+----------------------------------------------------------------------------------------------+-----------------------+
| Modality   | Model                                                                                        | Trained on            |
+============+==============================================================================================+=======================+
| ASR        | `QuartzNet15x5En <https://ngc.nvidia.com/catalog/models/nvidia:multidataset_quartznet15x5>`_ | LibriSpeech, WSJ,     |
|            |                                                                                              | Mozilla Common Voice  |
|            |                                                                                              | (en_1488_2019-12-10), |
|            |                                                                                              | Fisher, Switchboard   |
|            |                                                                                              | and Singapore English |
|            |                                                                                              | National Speech       |
|            |                                                                                              | Corpus                |
+------------+----------------------------------------------------------------------------------------------+-----------------------+
| ASR        | `QuartzNet15x5Zh <https://ngc.nvidia.com/catalog/models/nvidia:aishell2_quartznet15x5>`_     | AISHELL-2 Mandarin    |
|            |                                                                                              |                       |
|            |                                                                                              |                       |
|            |                                                                                              |                       |
+------------+----------------------------------------------------------------------------------------------+-----------------------+
| NLP        | `BERT base uncased <https://ngc.nvidia.com/catalog/models/nvidia:bertbaseuncasedfornemo>`_   |English Wikipedia and  |
|            |                                                                                              |BookCorpus dataset     |
|            |                                                                                              |seq len <= 512         |
|            |                                                                                              |                       |
+------------+----------------------------------------------------------------------------------------------+-----------------------+
| NLP        | `BERT large uncased <https://ngc.nvidia.com/catalog/models/nvidia:bertlargeuncasedfornemo>`_ |English Wikipedia and  |
|            |                                                                                              |BookCorpus dataset     |
|            |                                                                                              |seq len <= 512         |
|            |                                                                                              |                       |
+------------+----------------------------------------------------------------------------------------------+-----------------------+
| TTS        | `Tacotron2 <https://ngc.nvidia.com/catalog/models/nvidia:tacotron2_ljspeech>`_               |LJspeech               |
|            |                                                                                              |                       |
|            |                                                                                              |                       |
|            |                                                                                              |                       |
+------------+----------------------------------------------------------------------------------------------+-----------------------+
| TTS        | `WaveGlow <https://ngc.nvidia.com/catalog/models/nvidia:waveglow_ljspeech>`_                 |LJspeech               |
|            |                                                                                              |                       |
|            |                                                                                              |                       |
|            |                                                                                              |                       |
+------------+----------------------------------------------------------------------------------------------+-----------------------+


DEVELOPMENT
~~~~~~~~~~~
If you'd like to use master branch and/or develop NeMo you can run "reinstall.sh" script.

`Documentation (master branch) <http://nemo-master-docs.s3-website.us-east-2.amazonaws.com/>`_.

**Installing From Github**

If you prefer to use NeMo's latest development version (from GitHub) follow the steps below:

1) Clone the repository ``git clone https://github.com/NVIDIA/NeMo.git``
2) Go to NeMo folder and re-install the toolkit with collections:

.. code-block:: bash

    ./reinstall.sh

**Style tests**

.. code-block:: bash

    python setup.py style  # Checks overall project code style and output issues with diff.
    python setup.py style --fix  # Tries to fix error in-place.
    python setup.py style --scope=tests  # Operates within certain scope (dir of file).

**Unittests**

This command runs unittests:

.. code-block:: bash

    ./reinstall.sh
    pytest tests


Citation
~~~~~~~~

If you are using NeMo please cite the following publication

.. code-block:: tex

    @misc{nemo2019,
        title={NeMo: a toolkit for building AI applications using Neural Modules},
        author={Oleksii Kuchaiev and Jason Li and Huyen Nguyen and Oleksii Hrinchuk and Ryan Leary and Boris Ginsburg and Samuel Kriman and Stanislav Beliaev and Vitaly Lavrukhin and Jack Cook and Patrice Castonguay and Mariya Popova and Jocelyn Huang and Jonathan M. Cohen},
        year={2019},
        eprint={1909.09577},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }

