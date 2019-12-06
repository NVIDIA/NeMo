.. _installation:

如何安装
========

**依赖组件**

1) Python 3.6 or 3.7
2) `CUDA <https://developer.nvidia.com/cuda-downloads/>`_ >= 10.0
3) `cuDNN <https://developer.nvidia.com/cudnn/>`_ >= 7.6
4) `APEX <https://github.com/NVIDIA/apex/>`_
5) PyTorch >=1.2
6) 对于多 GPU 或者分布式训练，推荐安装： `NCCL <https://github.com/NVIDIA/nccl/>`_ >= 2.4

.. tip::
    您还可以不安装这些依赖，直接使用 `英伟达的 PyTorch 镜像 <https://ngc.nvidia.com/catalog/containers/nvidia:pytorch>`_ .
    在该镜像中，所有的依赖都已提前为您安装好。方便您直接使用。

**安装 NeMo 及其 Collections **

1) 克隆代码库：

.. code-block:: bash

    git clone https://github.com/NVIDIA/nemo

2) 切换到 ``nemo`` 文件夹并运行： ``python setup.py install``

3) 安装 collections 

   a) ASR: 进入 ``collections/nemo_asr`` 文件夹并运行 ``sudo apt-get install libsndfile1 && python setup.py install``
   b) NLP: 进入 ``collections/nemo_nlp`` 文件夹并运行 ``python setup.py install``
   c) LPR: 进入 ``collections/nemo_simple_gan`` 文件夹并运行 ``python setup.py install``

对于开发模型，请运行 ``python setup.py develop``

4) 运行单元测试以验证是否安装成功：

.. code-block:: bash

    python -m unittest tests/*.py

5) 切换到 ``examples/start_here`` 文件夹运行一些范例。

