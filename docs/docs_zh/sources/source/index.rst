NVIDIA Neural Modules 开发者指南(中文版)
==========================================

.. toctree::
   :hidden:
   :maxdepth: 2

   简介 <self>
   tutorials/intro
   training
   asr/intro
   speech_command/intro
   nlp/intro
   tts/intro
   collections/modules
   api-docs/modules



Neural Modules (NeMo) 是一个用神经模块来构建 AI 应用的工具包，它与具体的框架无关。当前支持 PyTorch 框架。

一个“神经模块”指的是，根据一系列的输入来计算一系列输出的代码块。

神经模块的输入和输出的神经类型会有语义检查。

用 NeMo 构建的应用，是一个由连接在一起的模块构成的有向无环图，研究者们可以很容易地通过 API 兼容的模块，定义和构建新的语音或语言神经网络。


简介
-----

我们可以通过以下这个视频有个概览：

.. raw:: html

    <div>
        <iframe src="https://drive.google.com/file/d/1CF-buP_Y1qCAefzoyvOUCXl_3v2vO5P-/preview" width="640" height="480"></iframe>
    </div>


核心概念和特性
------------------

* `NeuralModule` 类 - 表示以及执行一个神经模块。
* `NmTensor` - 表示的是神经模块端口之间流动的激活元。
* `NeuralType` - 表示模块端口和 NmTensors 的类型。
* `NeuralFactory` - 创建神经模块并且管理训练流程。
* `Lazy execution` - NeMo 描述神经模块之间的激活流时，运算并没有发生。只有在运行某个 “action”，例如 `optimizer.optimize(...)` 才会真正触发计算。
* `Collections` - NeMo中附带的模块集合 -  与其相关的模块集合，例如， `nemo_asr` (语音识别) 以及 `nemo_nlp` (自然语言处理)


安装依赖
----------

1) Python 3.6 or 3.7
2) PyTorch >= 1.4 带GPU支持
3) （可选）NVIDIA APEX: https://github.com/NVIDIA/apex

.. _installation:

开始吧
------

你可以从这个 docker 容器开始 `NGC NeMo 容器 <https://ngc.nvidia.com/catalog/containers/nvidia:nemo>`_ 这里面已经包含了最新版的 NeMo 和上面所需要的环境。

.. code-block:: bash

    # pull相应的 docker 容器
    docker pull nvcr.io/nvidia/nemo:v0.9

    # 运行下面两个命令之一
    # 如果你的 docker 版本 <19.03
    nvidia-docker run -it --rm --shm-size=8g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/nemo:v0.9

    # 如果你的 docker 版本 >=19.03
    docker run -it --rm --gpus all --shm-size=8g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/nemo:v0.9

马上开始用 NeMo 吧。

如果你已经安装了所有依赖（或者使用了 `NGC PyTorch容器 <https://ngc.nvidia.com/catalog/containers/nvidia:pytorch>`_ ）
那么你只要简单的用 pip来安装最新的 NeMo 和 NeMo 集合即可

.. code-block:: bash

    pip install nemo-toolkit[all] # 安装 NeMo 核心和所有集合(nemo_asr, nemo_nlp, nemo_tts)

教程
------

* `语音识别 <asr-docs>`_
* `自然语言处理 <nlp-docs>`_
* `语音合成 <tts-docs>`_

从github上安装
--------------

如果你更想用 NeMo 最新的开发版本（从 github上 获取），请按照下面的步骤：

1) 克隆这个仓库 ``git clone https://github.com/NVIDIA/NeMo.git``
2) 切到 nemo 文件夹下，安装工具包和各个集合: 

.. code-block:: bash

    ./reinstall.sh

.. note::
    reinstall.sh 是在开发者模式下安装 NeMo

单元测试
---------

下面这个命令会运行单元测试:

.. code-block:: bash

    ./reinstall.sh
    python -m unittest tests/*.py

构建 Docker 容器
-------------------------

NeMo Docker 镜像需要 Docker Buildx (包含在 Docker 19.03)。 想要构建一个自定义的 NeMo Docker 镜像, 运行

.. code-block:: bash

    docker buildx build --build-arg NEMO_VERSION=$(git describe --tags) -t nemo .

``NEMO_VERSION``参数是必须的。我们推荐设置为 ``git describe --tags`` 这样构建就可追溯和可复现的。
在运行时候, ``NEMO_VERSION`` 在构建时候指定的值会变成环境变量。

你也可以指定参数 ``BASE_IMAGE`` 来重载底层版本的 Pytorch，不过，不保证兼容性。

为了开发需要, 你也可以构建一个只包含 NeMo 依赖的Docker镜像。
在运行时，把你本地的开发分支映射到容器中。

.. code-block:: bash

    # 构建开发容器
    docker buildx build --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:20.01-py3 --target nemo-deps -t nemo-devel .

    # 运行容器, 把本地 nemo 映射进去
    cd <nemo_path>
    docker run -it --rm --gpus all -v $(pwd):/workspace/nemo --shm-size=8g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/nemo:v0.9

    # 在开发模式下安装
    ./reinstall.sh

