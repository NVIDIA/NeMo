NVIDIA Neural Modules: NeMo 中文文档
=====================================

.. toctree::
   :hidden:
   :maxdepth: 2

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

你可以从这个 docker 容器开始 `NGC PyTorch容器 <https://ngc.nvidia.com/catalog/containers/nvidia:pytorch>`_ 这里面已经包含了上面所需要的环境。

.. code-block:: bash

    # pull相应的 docker 容器
    docker pull nvcr.io/nvidia/pytorch:19.10-py3

    # 运行下面两个命令之一
    # 如果你的 docker 版本 <19.03
    nvidia-docker run -it --rm -v <nemo_github_folder>:/NeMo --shm-size=1g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:19.10-py3
    
    # 如果你的 docker 版本 >=19.03
    docker run --runtime=nvidia -it --rm -v <nemo_github_folder>:/NeMo --shm-size=1g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:19.10-py3

    cd /NeMo


接着运行下面的步骤

如果你已经安装了所有依赖（或者使用了 `NGC PyTorch容器 <https://ngc.nvidia.com/catalog/containers/nvidia:pytorch>`_ ）
那么你只要简单的用 pip来安装最新的 NeMo 和 NeMo 集合即可

.. code-block:: bash

    pip install nemo-toolkit  # 安装 NeMo Core
    pip install nemo-asr # 安装 NeMo asr 集合
    pip install nemo-nlp # 安装 NeMo nlp 集合
    pip install nemo-tts # 安装 NeMo tts 集合

**教程**

* `语音识别 <https://nvidia.github.io/NeMo/asr/zh/intro.html>`_
* `自然语言处理 <https://nvidia.github.io/NeMo/zh/nlp/intro.html>`_
* `语音合成 <https://nvidia.github.io/NeMo/tts/zh/intro.html>`_


**从github上安装**

如果你更想用 NeMo 最新的开发版本（从 github上 获取），请按照下面的步骤：

*Note*: 对于下面的步骤2和3，如果你想在开发模式下用 NeMo，用: ``pip install -e .`` 而不是 ``pip install .``

1) 克隆这个仓库 ``git clone https://github.com/NVIDIA/NeMo.git``
2) 切到 nemo 文件夹下，安装工具包: 

.. code-block:: bash

    cd NeMo/nemo
    pip install .

3) 安装 collections:

.. code-block:: bash

    # 从 collections/nemo_asr 下安装 ASR 集合
    apt-get install libsndfile1
    cd NeMo/collections/nemo_asr
    pip install .

    # 从 collections/nemo_nlp 下安装 NLP 集合
    cd NeMo/collections/nemo_nlp
    pip install .

    # 从 collections/nemo_tts 下安装 TTS 集合
    cd NeMo/collections/nemo_tts
    pip install .

**单元测试**

下面这个命令会运行单元测试:

.. code-block:: bash

    ./reinstall.sh
    python -m unittest tests/*.py
