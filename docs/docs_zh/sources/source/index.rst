NVIDIA Neural Modules: NeMo 中文文档
=====================================

.. toctree::
   :hidden:
   :maxdepth: 3

   简介 <self>
   installation
   tutorials/intro
   training
   asr/intro
   nlp/intro
   tts/intro
   collections/modules
   api-docs/modules




Neural Modules (NeMo) 是一个用神经模块来构建 AI 应用的工具包，它与具体的框架无关。当前支持 PyTorch 框架。

一个“神经模块”指的是，根据一系列的输入来计算一系列输出的代码块。

神经模块的输入和输出的神经类型会有语义检查。

用 NeMo 构建的应用，是一个由连接在一起的模块构成的有向无环图，研究者们可以很容易地通过 API 兼容的模块，定义和构建新的语音或语言神经网络。

**简介**

我们可以通过以下这个视频有个概览：

.. raw:: html

    <div>
        <iframe src="https://drive.google.com/file/d/1CF-buP_Y1qCAefzoyvOUCXl_3v2vO5P-/preview" width="640" height="480"></iframe>
    </div>


**核心概念和特性**

* `NeuralModule` 类 - 表示以及执行一个神经模块。
* `NmTensor` - 表示的是神经模块端口之间流动的激活元。
* `NeuralType` - 表示模块端口类型和 NmTensors。
* `NeuralFactory` - 创建神经模块并且管理训练流程。
* `Lazy execution` - 当描述神经模块之间的激活流时，在一个“action”(比如 `optimizer.optimize(...)` 没有触发前，什么都不会发生。
* `Collections` - NeMo中附带的模块集合 -  与其相关的模块集合，例如， `nemo_asr` (语音识别) 以及 `nemo_nlp` (自然语言处理)


**安装依赖**

1) Python 3.6 or 3.7
2) PyTorch 1.2 with GPU support
3) NVIDIA APEX: https://github.com/NVIDIA/apex


**开始吧**

如果需要的话，你可以从这个 docker 容器开始 `NGC PyTorch容器 <https://ngc.nvidia.com/catalog/containers/nvidia:pytorch>`_ 这里面已经包含了上面所需要的环境。

你可以直接运行 ``docker pull nvcr.io/nvidia/pytorch:19.08-py3``

接着按照下面的步骤：

1) 克隆这个仓库
2) 切到 nemo 文件夹下，运行: ``python setup.py install``
3) 安装 collections:

    * ASR collections `collections/nemo_asr`:

        1) ``apt-get install libsndfile1``
        2) ``python setup.py install``

    * NLP collections `collections/nemo_nlp`: ``python setup.py install``
    * TTS collections `collections/nemo_tts`: ``python setup.py install``

4) 如果要开发，你需要运行: ``python setup.py develop`` 而不是之前步骤3.2中的 ``python setup.py install``
5) 到 `examples/start_here` 下，从这几个简单的例子开始吧


**单元测试**

下面这个命令会运行单元测试:

.. code-block:: bash

    ./reinstall.sh
    python -m unittest tests/*.py
