.. _installation:

Installation
============

**Requirements**

1) Python 3.6 or 3.7
2) `CUDA <https://developer.nvidia.com/cuda-downloads/>`_ >= 10.0
3) `cuDNN <https://developer.nvidia.com/cudnn/>`_ >= 7.6
4) `APEX <https://github.com/NVIDIA/apex/>`_
5) PyTorch >=1.1
6) (Recommended for distributed training) `NCCL <https://github.com/NVIDIA/nccl/>`_ >= 2.4


**Installing NEMO and Collections**

1) Clone the repository:

.. code-block:: bash

    git clone https://github.com/NVIDIA/nemo

2) Go to ``nemo`` folder and do: ``python setup.py install``

3) Run unittests to validate instalation:

.. code-block:: bash

    ./reinstall.sh
    python -m unittest tests/*.py

4) Install collections

   a) ASR collection from ``collections/nemo_asr`` do: ``sudo apt-get install libsndfile1 && python setup.py install``
   b) NLP collection from ``collections/nemo_nlp`` do: ``python setup.py install``
   c) LPR collection from ``collections/nemo_lpr`` do: ``python setup.py install`` 

 
For development do: ``python setup.py develop`` instead of ``python setup.py install``

5) Go to ``examples/start_here`` to get started with few simple examples



