教程
========

确保你已经安装了 ``nemo`` 和 ``nemo_asr``
参考 :ref:`installation` 部分。

.. note::
    在这个教程中你只需要用到 `nemo` 和 `nemo_asr`。

简介
-------------
这个教程中我们使用 Jasper :cite:`asr-tut-li2019jasper` 模型。Jasper 是一个基于 CTC :cite:`asr-tut-graves2006` 的端到端的语音识别模型。这个模型之所以被称之为“端到端”是因为它在不需要额外的对齐信息下就可以把输入的音频样本转到对应的文本上。
CTC 可以在音频和文本中找到对齐方式。基于 CTC 的语音识别管道包含了下面的这些模块：

1. 音频预处理（特征提取）：信号正则化，窗口化，（log）频谱（梅尔谱或者 MFCC）
2. 神经网络声学模型（在给定的每个时间步上的输入特征下，预测词表中字符c的概率分布 P_t(c)）
3. CTC 损失函数

    .. image:: ctc_asr.png
        :align: center
        :alt: CTC-based ASR



获取数据
--------
我们会使用 LibriSpeech :cite:`asr-tut-panayotov2015librispeech` 数据集。下面这些脚本会下载并且把 Librispeech 转成 `nemo_asr` 需要的数据格式：

.. code-block:: bash

    mkdir data
    # 我们需要安装 sox
    # 在 ubuntu 上安装 sox, 只需要：sudo apt-get install sox
    # 接着：pip install sox
    # get_librispeech_data.py script 位于 <nemo_git_repo_root>/scripts 目录下
    python get_librispeech_data.py --data_root=data --data_set=dev_clean,train_clean_100
    # 如果想获取所有的 Librispeech 数据:
    # python get_librispeech_data.py --data_root=data --data_set=ALL

.. note::
    如果用 ``--data_set=dev_clean,train_clean_100`` ，你的磁盘空间至少需要 26GB。如果用 ``--data_set=ALL`` ，你的磁盘空间至少需要 110GB。下载和处理都需要一段时间，所以休息一下下吧。



下载和转换后, 你的 `data` 文件夹应该包含两个 Json 文件：

* dev_clean.json
* train_clean_100.json

在这个教程中我们会使用 `train_clean_100.json` 做训练，以及 `dev_clean.json` 做评估。
Json 文件中的每一行都指的是一个训练样本 `audio_filepath` 包含了 wav 文件的路径，`duration` 为该文件的音频时长（秒），`text` 是音频对应的文本：

.. code-block:: json

    {"audio_filepath": "<absolute_path_to>/1355-39947-0000.wav", "duration": 11.3, "text": "psychotherapy and the community both the physician and the patient find their place in the community the life interests of which are superior to the interests of the individual"}
    {"audio_filepath": "<absolute_path_to>/1355-39947-0001.wav", "duration": 15.905, "text": "it is an unavoidable question how far from the higher point of view of the social mind the psychotherapeutic efforts should be encouraged or suppressed are there any conditions which suggest suspicion of or direct opposition to such curative work"}



训练
---------

我们会在 Jasper 家族 :cite:`asr-tut-li2019jasper` 中训练一个小模型。
Jasper （"Just Another SPeech Recognizer"） 是一个深度时延网络 （TDNN） 包含了一维卷积层的块（blocks）。 
Jasper 家族的模型的结构可以这样表示 Jasper_[BxR] 其中 B 是块的个数, R 表示的是一个块中卷积子块的个数。每个子块包含了一个一维卷积层，一层 batch normalization，一个 ReLU 激活函数，和一个 dropout 层：

    .. image:: jasper.png
        :align: center
        :alt: japer model

在这个教程中我们会使用 [12x1] 的模型结构并且会用分开的卷积。
下面脚本的训练（on `train_clean_100.json` ）和评估（on `dev_clean.json` ）都是在一块GPU上：

    .. tip::
        运行 Jupyter notebook，一步一步跟着这个脚本运行一遍。


**训练脚本**

.. code-block:: python

    # NeMo's "core" package
    import nemo
    # NeMo's ASR collection
    import nemo_asr

    # 创建 Neural Factory
    # 它会为我们创建日志文件和 tensorboard 记录器
    nf = nemo.core.NeuralModuleFactory(
        log_dir='jasper12x1SEP',
        create_tb_writer=True)
    tb_writer = nf.tb_writer
    logger = nf.logger

    # 到训练列表文件的路径
    train_dataset = "<path_to_where_you_put_data>/train_clean_100.json"

    # 到验证集列表文件的路径
    eval_datasets = "<path_to_where_you_put_data>/dev_clean.json"

    # Jasper 模型定义
    from ruamel.yaml import YAML

    # 这里我们用可分离卷积
    # with 12 blocks (k=12 repeated once r=1 from the picture above)
    yaml = YAML(typ="safe")
    with open("<nemo_git_repo_root>/examples/asr/configs/jasper12x1SEP.yaml") as f:
        jasper_model_definition = yaml.load(f)
    labels = jasper_model_definition['labels']

    # 初始化神经模块
    data_layer = nemo_asr.AudioToTextDataLayer(
        manifest_filepath=train_dataset,
        labels=labels, batch_size=32)
    data_layer_val = nemo_asr.AudioToTextDataLayer(
        manifest_filepath=eval_datasets,
        labels=labels, batch_size=32, shuffle=False)

    data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor()
    spec_augment = nemo_asr.SpectrogramAugmentation(rect_masks=5)

    jasper_encoder = nemo_asr.JasperEncoder(
        feat_in=64,
        **jasper_model_definition['JasperEncoder'])
    jasper_decoder = nemo_asr.JasperDecoderForCTC(
        feat_in=1024, num_classes=len(labels))
    ctc_loss = nemo_asr.CTCLossNM(num_classes=len(labels))
    greedy_decoder = nemo_asr.GreedyCTCDecoder()

    # 训练有向无环图 DAG （模型）
    audio_signal, audio_signal_len, transcript, transcript_len = data_layer()
    processed_signal, processed_signal_len = data_preprocessor(
        input_signal=audio_signal, length=audio_signal_len)
    aug_signal = spec_augment(input_spec=processed_signal)
    encoded, encoded_len = jasper_encoder(
        audio_signal=aug_signal, length=processed_signal_len)
    log_probs = jasper_decoder(encoder_output=encoded)
    predictions = greedy_decoder(log_probs=log_probs)
    loss = ctc_loss(
        log_probs=log_probs, targets=transcript,
        input_length=encoded_len, target_length=transcript_len)

    # 验证有向无环图 DAG （模型）
    # 我们需要为验证集初始化额外的数据层的神经模块
    audio_signal_v, audio_signal_len_v, transcript_v, transcript_len_v = data_layer_val()
    processed_signal_v, processed_signal_len_v = data_preprocessor(
        input_signal=audio_signal_v, length=audio_signal_len_v)
    # 注意我们再验证 DAG 的时候不会用数据增强
    encoded_v, encoded_len_v = jasper_encoder(
        audio_signal=processed_signal_v, length=processed_signal_len_v)
    log_probs_v = jasper_decoder(encoder_output=encoded_v)
    predictions_v = greedy_decoder(log_probs=log_probs_v)
    loss_v = ctc_loss(
        log_probs=log_probs_v, targets=transcript_v,
        input_length=encoded_len_v, target_length=transcript_len_v)

    # 这些帮助函数对于打印和计算不同的指标很重要
    # 比如计算错词率和把它们记录到 tensorboard
    # 这些函数是领域特殊性的，由 NeMo 的不同 collections 提供（nemo_asr，nemo_nlp）
    from nemo_asr.helpers import monitor_asr_train_progress, \
        process_evaluation_batch, process_evaluation_epoch

    from functools import partial
    # 回调追踪损失值，打印训练中的预测结果
    train_callback = nemo.core.SimpleLossLoggerCallback(
        tb_writer=tb_writer,
        # 定义让 SimpleLossLoggerCallback 回调打印的张量
        # 这里我们想打印损失值，和我们的错词率
        # 错词率是预测值，文本和文本长度的函数
        tensors=[loss, predictions, transcript, transcript_len],
        # 为了能把日志打印到屏幕，定义一个 print_func 函数
        print_func=partial(
            monitor_asr_train_progress,
            labels=labels,
            logger=logger
        ))

    saver_callback = nemo.core.CheckpointCallback(
        folder="./",
        # 设置多少个步数保存一次 checkpoint
        step_freq=100)

    # PRO TIP: 虽然你只能有一个有向无环图，但是你可以有任意个验证有向无环图和回调函数
    # 如果你想在多个验证集上做监测，这非常重要
    # (比如说LibriSpeech的dev clean和dev other两个数据集)
    eval_callback = nemo.core.EvaluatorCallback(
        eval_tensors=[loss_v, predictions_v, transcript_v, transcript_len_v],
        # 如何处理验证集的每个 batch - 例如，计算 WER
        user_iter_callback=partial(
            process_evaluation_batch,
            labels=labels
            ),
        # 如何把每个 batch 的验证集统计指标（比如WER）合并起来
        user_epochs_done_callback=partial(
            process_evaluation_epoch, tag="DEV-CLEAN", logger=logger
            ),
        eval_step=500,
        tb_writer=tb_writer)

    # 用你的 Neural Factory 跑训练
    # 一旦这个“操作”开始调用，数据开始在训练和验证的有向无环图上流动
    # 计算就开始了
    nf.train(
        # 指定需要优化的损失函数
        tensors_to_optimize=[loss],
        # 定义你想跑多少个回调
        callbacks=[train_callback, eval_callback, saver_callback],
        # 定义想用哪个优化器
        optimizer="novograd",
        # 定义优化器的参数，训练轮数和学习率
        optimization_params={
            "num_epochs": 50, "lr": 0.02, "weight_decay": 1e-4
            }
        )

.. note::
    这个脚本在 GTX1080 上完成 50 轮训练需要大约 7 小时

.. tip::
    进一步提升 WER:
        (1) 训练的更久
        (2) 训更多的数据
        (3) 用更大的模型
        (4) 在多 GPU 上训练并且使用混精度训练（NVIDIA Volta 和 Turing 架构的GPU）
        (5) 从预训练好的 checkpoints 上开始训练


混精度训练
-------------------------
NeMo 中的混精度和分布式训练上基于 `NVIDIA's APEX library <https://github.com/NVIDIA/apex>`_ 。
确保它已经安装了。

训混精度训练你只需要在 `nemo.core.NeuralModuleFactory` 中设置 `optimization_level` 参数为 `nemo.core.Optimization.mxprO1` 。例如：

.. code-block:: python

    nf = nemo.core.NeuralModuleFactory(
        backend=nemo.core.Backend.PyTorch,
        local_rank=args.local_rank,
        optimization_level=nemo.core.Optimization.mxprO1,
        placement=nemo.core.DeviceType.AllGpu,
        cudnn_benchmark=True)

.. note::
    因为混精度训练需要 Tensor Cores, 因此它当前只能在 NVIDIA Volta 和 Turing 架构的 GPU 上运行。

多 GPU 训练
-------------------

在 NeMo 中开启多 GPU 训练很容易：

   (1) 首先把 NeuralModuleFactory 中的 `placement` 设置成 `nemo.core.DeviceType.AllGpu`
   (2) 让你的脚本能够接受 'local_rank' 参数，你无需手动指定该参数值，只需要在代码中添加: `parser.add_argument("--local_rank", default=None, type=int)`
   (3) 用 `torch.distributed.launch` 包来运行你的脚本（把 <num_gpus> 改成 GPU 的数量）

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=<num_gpus> <nemo_git_repo_root>/examples/asr/jasper.py ...


大量训练样本例子
~~~~~~~~~~~~~~~~~~~~~~

请参考 `<nemo_git_repo_root>/examples/asr/jasper.py` , 该实例做一个更全面的理解。它构建了一个训练的有向无环图，在不同的验证集上构建了多达三个有向无环图。

假设你能够使用基于 Volta 架构的的 DGX 服务器，你可以这样运行：

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=<num_gpus> <nemo_git_repo_root>/examples/asr/jasper.py --batch_size=64 --num_epochs=100 --lr=0.015 --warmup_steps=8000 --weight_decay=0.001 --train_dataset=/manifests/librivox-train-all.json --eval_datasets /manifests/librivox-dev-clean.json /manifests/librivox-dev-other.json --model_config=<nemo_git_repo_root>/nemo/examples/asr/configs/quartznet15x5.yaml --exp_name=MyLARGE-ASR-EXPERIMENT

上面的命令行应该会出发一个8卡的混精度训练。其中不同的列表文件（.json）文件是不同的数据集。你可以用你的数据来替代它们。

.. tip::
    你可以用逗号分隔不同的数据集：`--train_manifest=/manifests/librivox-train-all.json,/manifests/librivox-train-all-sp10pcnt.json,/manifests/cv/validated.json`。
    这里使用了3个数据集 LibriSpeech，Mozilla Common Voice 和 LibriSpeech音频速度进行干扰后的数据集。


微调
-----------
如果我们从一个好的预训练模型开始训练，训练时间会大大的减小：

    (1) 从`这里 <https://ngc.nvidia.com/catalog/models/nvidia:quartznet15x5>`_获取预训练模型 （jasper_encoder，jasper_decoder 和 configuration files）。
    (2) 在你初始化好 jasper_encoder 和 jasper_decoder 后，可以这样加载权重：

.. code-block:: python

    jasper_encoder.restore_from("<path_to_checkpoints>/15x5SEP/JasperEncoder-STEP-247400.pt")
    jasper_decoder.restore_from("<path_to_checkpoints>/15x5SEP/JasperDecoderForCTC-STEP-247400.pt")
    # 防止是分布式训练加入 args.local_rank
    jasper_decoder.restore_from("<path_to_checkpoints>/15x5SEP/JasperDecoderForCTC-STEP-247400.pt", args.local_rank)

.. tip::
    微调的时候，用小一点的学习率。


推理
---------

首先下载预训练模型（jasper_encoder, jasper_decoder and configuration files） 请从 `这里 <https://ngc.nvidia.com/catalog/models/nvidia:quartznet15x5>`_ 下载并放置到 `<path_to_checkpoints>` 。 我们会用这个预训练模型在 LibriSpeech dev-clean 数据集上测试 WER。

.. code-block:: bash

    python <nemo_git_repo_root>/examples/asr/jasper_infer.py --model_config=<nemo_git_repo_root>/examples/asr/configs/quartznet15x5.yaml --eval_datasets "<path_to_data>/dev_clean.json" --load_dir=<directory_containing_checkpoints>


用语言模型推理
-----------------------------

用KenLM构建的语言模型
~~~~~~~~~~~~~~~~~~~~~~
我们会使用 `Baidu's CTC 带语言模型的解码器 <https://github.com/PaddlePaddle/DeepSpeech>`_ .

请按照下面的步骤：

    * 到 scripts 目录下 ``cd <nemo_git_repo_root>/scripts``
    * 安装百度 CTC 解码器（如果在 docker 容器中不需要用 sudo）：
        * ``sudo apt-get update && sudo apt-get install swig``
        * ``sudo apt-get install pkg-config libflac-dev libogg-dev libvorbis-dev libboost-dev``
        * ``sudo apt-get install libsndfile1-dev python-setuptools libboost-all-dev python-dev``
        * ``./install_decoders.sh``
    * 在 Librispeech 上构建一个 6-gram KenLM 的语言模型  ``./build_6-gram_OpenSLR_lm.sh``
    * 运行 jasper_infer.py 带上 --lm_path 来指定语言模型的路径

    .. code-block:: bash

        python <nemo_git_repo_root>/examples/asr/jasper_infer.py --model_config=<nemo_git_repo_root>/examples/asr/configs/quartznet15x5.yaml --eval_datasets "<path_to_data>/dev_clean.json" --load_dir=<directory_containing_checkpoints> --lm_path=<path_to_6gram.binary>


参考
----------

.. bibliography:: asr_all.bib
    :style: plain
    :labelprefix: ASR-TUT
    :keyprefix: asr-tut-
