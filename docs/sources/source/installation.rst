.. _installation:

Installation
============

**Requirements**

1) Python 3.6 or 3.7
2) `CUDA <https://developer.nvidia.com/cuda-downloads/>`_ >= 10.0
3) `cuDNN <https://developer.nvidia.com/cudnn/>`_ >= 7.6
4) `APEX <https://github.com/NVIDIA/apex/>`_
5) PyTorch >=1.2
6) (Recommended for distributed training) `NCCL <https://github.com/NVIDIA/nccl/>`_ >= 2.4

.. tip:: Instead of installing all requirements. They are all automatically included 
    in the `NVIDIA's PyTorch container <https://ngc.nvidia.com/catalog/containers/nvidia:pytorch>`_ .
    You can pull it like so: `docker pull nvcr.io/nvidia/pytorch:19.08-py3`

**Installing NeMo and Collections**

*Note*: For step 2 and 3, if you want to use NeMo in development mode, use: ``pip install -e .`` instead of ``pip install .``

1) Clone the repository:

.. code-block:: bash

    git clone https://github.com/NVIDIA/nemo

2) Go to ``nemo`` folder and install NeMo Core:

.. code-block:: bash

    cd nemo
    pip install .

3) Install collections

	a) To install the ASR collection from ``collections/nemo_asr``:
	
   	.. code-block:: bash

   		cd ../collections/nemo_asr
   		sudo apt-get install libsndfile1 && pip install .


    b) To install the NLP collection from ``collections/nemo_nlp``:

    .. code-block:: bash

   		cd ../nemo_nlp
   		pip install .

    c) To install the LPR collection from ``collections/nemo_simple_gan``:

    .. code-block:: bash

   		cd ../nemo_simple_gan
   		pip install .


4) Run unittests from the nemo directory to validate installation:

.. code-block:: bash

    python -m unittest tests/*.py
    
All tests should pass without errors.

5) Go to ``examples/start_here`` to get started with few simple examples

