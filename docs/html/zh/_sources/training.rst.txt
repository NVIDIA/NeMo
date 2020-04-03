快速训练
========

训练较大的模型，特别是从头开始训练，需要巨大的算力。NeMo 支持分布式训练和混合精度训练以加速训练。NeMo 借助 `英伟达的 APEX 库 <https://github.com/NVIDIA/apex>`_ 在英伟达 GPU 上达到最佳的性能。另外，配备了多块 GPU 的系统（例如 DGX Station, DGX-1 & DGX-2 等），可以进一步地使用 *NVLINK* 加速 GPU 间的通信，从而最大限度地发挥 GPU 的性能。


混合精度训练
~~~~~~~~~~~~
在英伟达最新的 Volta 和 Turning 架构中，GPU 配备了 Tensor Cores 计算单元，能够大幅加速半精度浮点数的矩阵乘法运算。
想要在 NeMo 中使用混合精度训练，你可以设置 `nemo.core.NeuralModuleFactory` 类的 ``optimization_level`` 选项为 ``nemo.core.Optimization.mxprO1`` 。

.. code-block:: python

    nf = nemo.core.NeuralModuleFactory(
           optimization_level=nemo.core.Optimization.mxprO1)

.. important::
    混合精度训练需要 Tensor Cores 的硬件支持，因此当前只在英伟达的 Volta 或者 Turing GPU 上有支持。

多 GPU 训练
~~~~~~~~~~~

进行多 GPU 训练需要进行如下设置：

在你的 python 脚本中添加命令行选项 ``local_rank``: ``parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', None), type=int)``

.. code-block:: python

    nf = nemo.core.NeuralModuleFactory(
           local_rank=args.local_rank)


利用 PyTorch 中的 `torch.distributed.launch` 包运行脚本(假设8块GPU)：

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=8 <nemo_repo>/examples/asr/jasper.py ...

范例
~~~~

一个比较完整的利用 NeMo 训练 ASR 模型的范例，请参阅这个文件： `<nemo_git_repo_root>/examples/asr/jasper.py` 。 
这个例子会创建一个训练有向无环图和三个验证集上的有向无环图，以便在不同的数据集上对模型进行验证。

在一台配备了多块 Volta GPU 的系统上，你可以用如下的命令来开始训练：

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=8 <nemo_git_repo_root>/examples/asr/jasper.py --batch_size=64 --num_epochs=100 --lr=0.015 --warmup_steps=8000 --weight_decay=0.001 --train_manifest=/manifests/librivox-train-all.json --val_manifest1=/manifests/librivox-dev-clean.json --val_manifest2=/manifests/librivox-dev-other.json --model_config=<nemo_git_repo_root>/nemo/examples/asr/configs/jasper15x5SEP.yaml --exp_name=MyLARGE-ASR-EXPERIMENT

这条命令会触发8卡并行和混合精度训练，在上面的命令中，不同的列表文件（.json）指的是不同的数据集。你可以用自己的数据集来代替它们。

.. tip::
    你可以在选项中同时传入多个数据集，使用逗号隔开，例如：``--train_manifest=/manifests/librivox-train-all.json,/manifests/librivox-train-all-sp10pcnt.json,/manifests/cv/validated.json``

这个例子会在三个数据集上进行训练，LibriSpeech, Mozzila Common Voice 和 Librispeech 做了速度扰动后的数据集。

多节点训练
~~~~~~~~~~
我们强烈建议在进行多节点训练前，先阅读 pytorch 的分布式文档。这里是一个使用 TCP 初始化进行多节点训练的方法。
假设我们有两台机子，每台4张卡。
我们把机子1当主节点(master)。我们需要主节点的 IP 地址，以及它上面的一个空闲端口。
在机子1上，我们运行:

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=<MASTER_IP_ADDRESS> --master_port=<FREE_PORT> jasper.py ...

在机子2上，运行:

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=<MASTER_IP_ADDRESS> --master_port=<FREE_PORT> jasper.py ...

.. tip::
    设置环境变量 NCCL_DEBUG 为 INFO 来发现启动时候的问题

.. tip::
    我们推荐阅读下面的 pytorch 文档 
    https://pytorch.org/docs/stable/distributed.html#launch-utility
    https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py

.. tip::
    关于多进程, neural_factory 包含了两个属性 ``local_rank`` 和 ``global_rank``。
    ``local_rank`` 指的是当前机子上的 rank, 而 ``global_rank`` 指的是所有机子上的 rank。
    比如, 假设你有2台机子，每台4张GPU。 global_rank 0 指的是 local_rank 0 并且是第一台机子的
    第一张GPU, 而 global_rank 5 可以是 local_rank 0 并且是第二台机子的第一张卡。换句话说
    local_rank == 0 并且 global_rank == 0 确保了它占有主节点上的第一张卡； local_rank == 0
    且 global_rank != 0 确保它占有奴隶节点上的第一张卡。
