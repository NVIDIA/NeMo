快速训练
========

训练较大的模型，特别是从头开始训练，需要巨大的算力。NeMo支持分布式训练和混合精度训练以加速训练。NeMo借助 `英伟达的APEX库 <https://github.com/NVIDIA/apex>`_ 在英伟达GPU上达到最佳的性能。另外，配备了多块GPU的系统，例如DGX Station, DGX-1 & DGX-2等，可以进一步的加速GPU间的通信，从而最大限度的发挥GPU的性能。

混合精度训练
~~~~~~~~~~~~

英伟达最新的Volta架构和Turning架构GPU配备了Tensor Cores计算单元，能够大幅加速半精度数据的矩阵乘法运算。想要在NeMo中使用混合精度训练，你可以设置 `nemo.core.NeuralModuleFactory` 类的 `optimization_level` 选项为 `nemo.core.Optimization.mxprO1` 。

.. code-block:: python

    nf = nemo.core.NeuralModuleFactory(
           optimization_level=nemo.core.Optimization.mxprO1)

.. important::
    混合精度训练需要Tensor Cores的硬件支持，因此当前只在英伟达的Volta或者Turing GPU上有支持。

多GPU训练
~~~~~~~~~

进行多GPU训练需要进行如下设置：

(1) 在 `NeuralModuleFactory` 类中设置选项 `placement` 为 `nemo.core.DeviceType.AllGpu`
(2) 在你的python脚本中添加命令行选项 'local_rank': `parser.add_argument("--local_rank", default=None, type=int)`

.. code-block:: python

    nf = nemo.core.NeuralModuleFactory(
           placement=nemo.core.DeviceType.AllGpu,
           local_rank=args.local_rank)


利用PyTorch中的 `torch.distributed.launch` 包来启动你的训练：

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=8 <nemo_repo>/examples/asr/jasper.py --num_gpus=8 ...

范例
~~~~

一个比较完整的利用NeMo训练ASR模型的范例请参阅这个文件： `<nemo_repo>/examples/asr/jasper.py`. 这个例子会创建一个训练集和三个验证集以便在不同的数据集上对模型精度进行验证。

在一台配备了多块Volta GPU的系统上，你可以用如下的命令来开始训练：

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=8 <nemo_git_repo_root>/examples/asr/jasper.py --batch_size=64 --num_gpus=8 --num_epochs=100 --lr=0.015 --warmup_steps=8000 --weight_decay=0.001 --train_manifest=/manifests/librivox-train-all.json --val_manifest1=/manifests/librivox-dev-clean.json --val_manifest2=/manifests/librivox-dev-other.json --model_config=<nemo_git_repo_root>/nemo/examples/asr/configs/jasper15x5SEP.yaml --exp_name=MyLARGE-ASR-EXPERIMENT

这条命令会进行8卡并行和混合精度训练，并且会在多个数据集上进行验证。

.. tip::
    你可以在选项中同时传入多个数据集，他们之间用逗号隔开，例如：
    `--train_manifest=/manifests/librivox-train-all.json,/manifests/librivox-train-all-sp10pcnt.json,/manifests/cv/validated.json`.
