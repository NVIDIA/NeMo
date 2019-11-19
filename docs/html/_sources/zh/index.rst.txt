NVIDIA Neural Modules: NeMo 中文文档
===========================

.. toctree::
   :hidden:
   :maxdepth: 2

   简介 <self>
   installation
   tutorials/intro
   training
   asr/intro
   nlp/intro
   collections/modules
   api-docs/modules




Neural Modules (NeMo) 是一个用神经模块来构建AI应用的框架无关的工具包。当前支持PyTorch框架。

一个“神经模块”指的是根据一系列的输入来计算一系列输出的代码块。

神经模块的输入和输出的神经类型会有语义检查。

用NeMo构建的应用是一个由连接在一起的模块构成的有向无环图，研究者们可以很容易地通过API兼容的模块定义以及构建新的语音语言神经网络。

**简介**

通过以下这个视频我们可以有个概览：

.. raw:: html

    <div>
        <iframe src="https://drive.google.com/file/d/1CF-buP_Y1qCAefzoyvOUCXl_3v2vO5P-/preview" width="640" height="480"></iframe>
    </div>


**核心概念和特性**

* `NeuralModule` class - represents and implements a neural module.
* `NmTensor` - represents activations which flow between neural modules' ports.
* `NeuralType` - represents types of modules' ports and NmTensors.
* `NeuralFactory` - to create neural modules and manage training.
* `Lazy execution` - when describing activation flow between neural modules, nothing happens until an "action" (such as `optimizer.optimize(...)` is called.
* `Collections` - NeMo comes with collections - related group of modules such as `nemo_asr` (for Speech Recognition) and `nemo_nlp` for NLP


**安装要求**

1) Python 3.6 or 3.7
2) PyTorch 1.2 with GPU support
3) NVIDIA APEX: https://github.com/NVIDIA/apex


**开始吧**

如果需要的话， 你可以从这个docker容器开始 `NGC PyTorch container <https://ngc.nvidia.com/catalog/containers/nvidia:pytorch>`_ 这里面已经包含了上面所需要的环境。

你可以直接运行``docker pull nvcr.io/nvidia/pytorch:19.08-py3``

接着就按照下面的步骤：

1) Clone the repository
2) Go to nemo folder and then: ``python setup.py install``
3) Install collections:
    * ASR collection from `collections/nemo_asr`:
        1. ``apt-get install libsndfile1``
        2. ``python setup.py install``

    * NLP collection from `collections/nemo_nlp`: ``python setup.py install``
4) For development you will need to: ``python setup.py develop`` instead of ``python setup.py install`` in Step (3.2) above
5) Go to `examples/start_here` to get started with few simple examples


**单元测试**

下面这个命令会运行单元测试:

.. code-block:: bash

    ./reinstall.sh
    python -m unittest tests/*.py
