快速训练
========

训练较大的模型，特别是从头开始训练，需要巨大的算力。NeMo 支持分布式训练和混合精度训练以加速训练。NeMo 借助 `英伟达的 APEX 库 <https://github.com/NVIDIA/apex>`_ 在英伟达 GPU 上达到最佳的性能。另外，配备了多块 GPU 的系统（例如 DGX Station, DGX-1 & DGX-2 等），可以进一步地使用 *NVLINK* 加速 GPU 间的通信，从而最大限度地发挥 GPU 的性能。

混合精度训练
~~~~~~~~~~~~

在英伟达最新的 Volta 和 Turning 架构中，GPU 配备了 Tensor Cores 计算单元，能够大幅加速半精度浮点数的矩阵乘法运算。想要在 NeMo 中使用混合精度训练，你可以设置 `nemo.core.NeuralModuleFactory` 类的 ``optimization_level`` 选项为 ``nemo.core.Optimization.mxprO1`` 。

.. code-block:: python

    nf = nemo.core.NeuralModuleFactory(
           optimization_level=nemo.core.Optimization.mxprO1)

.. important::
    混合精度训练需要 Tensor Cores 的硬件支持，因此当前只在英伟达的 Volta 或者 Turing GPU 上有支持。

多 GPU 训练
~~~~~~~~~~~

进行多 GPU 训练需要进行如下设置：

(1) 在 ``NeuralModuleFactory`` 类中设置选项 ``placement`` 为 ``nemo.core.DeviceType.AllGpu``
(2) 在你的 python 脚本中添加命令行选项 ``local_rank``: ``parser.add_argument("--local_rank", default=None, type=int)``

.. code-block:: python

    nf = nemo.core.NeuralModuleFactory(
           placement=nemo.core.DeviceType.AllGpu,
           local_rank=args.local_rank)


利用 PyTorch 中的 `torch.distributed.launch` 包来启动训练：

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=8 <nemo_repo>/examples/asr/jasper.py --num_gpus=8 ...


范例
~~~~

一个比较完整的利用 NeMo 训练 ASR 模型的范例，请参阅这个文件： `<nemo_git_repo_root>/examples/asr/jasper.py` 。 这个例子会创建一个训练有向无环图和三个验证有向无环图，以便在不同的数据集上对模型精度进行验证。

在一台配备了多块 Volta GPU 的系统上，你可以用如下的命令来开始训练：

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=8 <nemo_git_repo_root>/examples/asr/jasper.py --batch_size=64 --num_gpus=8 --num_epochs=100 --lr=0.015 --warmup_steps=8000 --weight_decay=0.001 --train_manifest=/manifests/librivox-train-all.json --val_manifest1=/manifests/librivox-dev-clean.json --val_manifest2=/manifests/librivox-dev-other.json --model_config=<nemo_git_repo_root>/nemo/examples/asr/configs/jasper15x5SEP.yaml --exp_name=MyLARGE-ASR-EXPERIMENT

这条命令会进行8卡并行和混合精度训练，在上面的命令中，不同的列表文件（.json）指的是不同的数据集。你可以用自己的数据集来代替它们。

.. tip::
    你可以在选项中同时传入多个数据集，使用逗号隔开，例如：
    ``--train_manifest=/manifests/librivox-train-all.json,/manifests/librivox-train-all-sp10pcnt.json,/manifests/cv/validated.json``

这个例子会在三个数据集上进行训练，LibriSpeech, Mozzila Common Voice 和 Librispeech做了速度扰动后的数据集。
