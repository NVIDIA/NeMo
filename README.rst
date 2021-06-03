
|status| |documentation| |license| |lgtm_grade| |lgtm_alerts| |black|

.. |status| image:: http://www.repostatus.org/badges/latest/active.svg
  :target: http://www.repostatus.org/#active
  :alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.

.. |documentation| image:: https://readthedocs.com/projects/nvidia-nemo/badge/?version=main
  :alt: Documentation
  :target: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/

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

.. _main-readme:
**NVIDIA NeMo**
===============

Introduction
------------

NeMo is a toolkit for creating `Conversational AI <https://developer.nvidia.com/conversational-ai#started>`_ applications.

`Introductory video. <https://www.youtube.com/embed/wBgpMf_KQVw>`_

The toolkit comes with extendable collections of pre-built modules and ready-to-use models for:

* `Automatic Speech Recognition (ASR) <https://ngc.nvidia.com/catalog/collections/nvidia:nemo_asr>`_
* `Natual Language Processing (NLP) <https://ngc.nvidia.com/catalog/collections/nvidia:nemo_nlp>`_
* `Text-to-Speech (TTS) <https://ngc.nvidia.com/catalog/collections/nvidia:nemo_tts>`_.

Built for speed, NeMo can utilize NVIDIA's Tensor Cores and scale out training to multiple GPUs and multiple nodes.

Requirements
------------

1) Python 3.6, 3.7 or 3.8
2) Pytorch 1.8.1 or above
3) NVIDIA GPU for training

Documentation
-------------

.. |main| image:: https://readthedocs.com/projects/nvidia-nemo/badge/?version=main
  :alt: Documentation Status
  :scale: 100%
  :target: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/

.. |v1.0.0| image:: https://readthedocs.com/projects/nvidia-nemo/badge/?version=v1.0.0
  :alt: Documentation Status
  :scale: 100%
  :target: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/v1.0.0/

.. |stable| image:: https://readthedocs.com/projects/nvidia-nemo/badge/?version=stable
  :alt: Documentation Status
  :scale: 100%
  :target:  https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/

+---------+-------------+----------------------------------------------------------------------------------------------------------------------------------+
| Version | Status      | Description                                                                                                                      |
+=========+=============+==================================================================================================================================+
| Latest  | |main|      | `Documentation of the latest (i.e. main) branch. <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/>`_          |
+---------+-------------+----------------------------------------------------------------------------------------------------------------------------------+
| Next    | |v1.0.0| | `Documentation of the most recent release: v1.0.0 <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/v1.0.0/>`_ |
+---------+-------------+----------------------------------------------------------------------------------------------------------------------------------+
| Stable  | |stable|    | `Documentation of the stable (i.e. stable) branch. <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/>`_      |
+---------+-------------+----------------------------------------------------------------------------------------------------------------------------------+

Tutorials
---------
A great way to start with NeMo is by checking `one of our tutorials <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/v1.0.0/starthere/tutorials.html>`_.

Getting help with NeMo
----------------------
FAQ can be found on NeMo's `Discussions board <https://github.com/NVIDIA/NeMo/discussions>`_. You are welcome to ask questions or start discussions there.


Installation
------------

Pip
~~~
Use this installation mode if you want the latest released version.

.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    pip install Cython
    pip install nemo_toolkit['all']

Pip from source
~~~~~~~~~~~~~~~
Use this installation mode if you want the a version from particular GitHub branch (e.g main).

.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    pip install Cython
    python -m pip install git+https://github.com/NVIDIA/NeMo.git@{BRANCH}#egg=nemo_toolkit[all]


From source
~~~~~~~~~~~
Use this installation mode if you are contributing to NeMo.

.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    git clone https://github.com/NVIDIA/NeMo
    cd NeMo
    ./reinstall.sh

Docker containers:
~~~~~~~~~~~~~~~~~~

If you chose to work with main branch, we recommend using NVIDIA's PyTorch container version 21.03-py3 and then installing from GitHub.

.. code-block:: bash

    docker run --gpus all -it --rm -v <nemo_github_folder>:/NeMo --shm-size=8g \
    -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit \
    stack=67108864 --device=/dev/snd nvcr.io/nvidia/pytorch:21.03-py3

Examples
--------

Many example can be found under `"Examples" <https://github.com/NVIDIA/NeMo/tree/main/examples>`_ folder.


Contributing
------------

We welcome community contributions! Please refer to the  `CONTRIBUTING.md <https://github.com/NVIDIA/NeMo/blob/main/CONTRIBUTING.md>`_ CONTRIBUTING.md for the process.

License
-------
NeMo is under `Apache 2.0 license <https://github.com/NVIDIA/NeMo/blob/main/LICENSE>`_.
