教程
========

确保你已经安装了 ``nemo`` 和 ``nemo_asr``
参考 :ref:`installation` 部分.

.. note::
    在这个教程中你只需要用到 `nemo` 和 `nemo_asr`

简介
-------------
这个教程中我们使用Jasper :cite:`li2019jasper` 模型。Jasper是一个基于CTC :cite:`graves2006` 的端到端的语音识别模型。这个模型之所以被称之为“端到端”是因为它在不需要额外的对齐信息下就可以把输入的音频样本转到对应的抄本上。
CTC可以在音频和文本中找到对齐方式。基于CTC的语音识别管道包含了下面的这些模块：

1. 音频预处理(特征提取)： 信号正则化，窗口化，(log)频谱(梅尔谱或者MFCC)
2. 神经网络声学模型(在给定的每个时间步上的输入特征下，预测词表中字符c的概率分布P_t(c))
3. CTC损失函数

    .. image:: ctc_asr.png
        :align: center
        :alt: CTC-based ASR



获取数据
--------
我们会使用LibriSpeech :cite:`panayotov2015librispeech` 数据集. 下面这些脚本会下载并且把Librispeech转成 `nemo_asr` 需要的数据格式 :

.. code-block:: bash

    mkdir data
    # 我们需要安装sox
    # 在ubuntu上安装sox, 只需要: sudo apt-get install sox
    # 接着: pip install sox
    # get_librispeech_data.py script is located under <nemo_git_repo_root>/scripts
    python get_librispeech_data.py --data_root=data --data_set=dev_clean,train_clean_100
    # 如果想获取所有的Librispeech数据:
    # python get_librispeech_data.py --data_root=data --data_set=ALL

.. note::
    你的磁盘空间至少需要26GB，如果用 ``--data_set=dev_clean,train_clean_100``; 至少用100GB， 如果用 ``--data_set=ALL``. 下载和处理都需要一段时间.


下载和转换后, 你的 `data` 文件夹应该包含两个json文件:

* dev_clean.json
* train_clean_100.json

在这个教程中我们会使用 `train_clean_100.json` 做训练，以及 `dev_clean.json` 做评估.
json文件中的每一行都指的是一个训练样本 - `audio_filepath` 包含了wav文件的路径, `duration` 该文件的音频时长(秒), `text` 是抄本:

.. code-block:: json

    {"audio_filepath": "<absolute_path_to>/1355-39947-0000.wav", "duration": 11.3, "text": "psychotherapy and the community both the physician and the patient find their place in the community the life interests of which are superior to the interests of the individual"}
    {"audio_filepath": "<absolute_path_to>/1355-39947-0001.wav", "duration": 15.905, "text": "it is an unavoidable question how far from the higher point of view of the social mind the psychotherapeutic efforts should be encouraged or suppressed are there any conditions which suggest suspicion of or direct opposition to such curative work"}



训练
---------

我们会在Jasper家族 :cite:`li2019jasper` 中训练一个小模型。
Jasper ("Just Another SPeech Recognizer") 是一个深度时延网络 (TDNN) 包含了一维卷积层的块(blocks)。 
Jasper家族的模型的结构可以这样表示 Jasper_[BxR] 其中B是块的个数, R表示的是一个块中卷积子块的个数。每个子块包含了一个一维卷积层，一层batch normalization,一个ReLU激活函数, 和一个dropout层:

    .. image:: jasper.png
        :align: center
        :alt: japer model
jiaoben

在这个教程中我们会使用 [12x1] 的模型结构并且会用分开的卷积.
下面脚本的训练(on `train_clean_100.json`)和评估(on `dev_clean.json`)都是在一块GPU上:

    .. tip::
        运行Jupyter notebook，一步一步跟着这个脚本走一遍


**训练脚本**

.. code-block:: python

    # NeMo's "core" package
    import nemo
    # NeMo's ASR collection
    import nemo_asr

    # Create a Neural Factory
    # It creates log files and tensorboard writers for us among other functions
    nf = nemo.core.NeuralModuleFactory(
        log_dir='jasper12x1SEP',
        create_tb_writer=True)
    tb_writer = nf.tb_writer
    logger = nf.logger

    # Path to our training manifest
    train_dataset = "<path_to_where_you_put_data>/train_clean_100.json"

    # Path to our validation manifest
    eval_datasets = "<path_to_where_you_put_data>/dev_clean.json"

    # Jasper Model definition
    from ruamel.yaml import YAML

    # Here we will be using separable convolutions
    # with 12 blocks (k=12 repeated once r=1 from the picture above)
    yaml = YAML(typ="safe")
    with open("<nemo_git_repo_root>/examples/asr/configs/jasper12x1SEP.yaml") as f:
        jasper_model_definition = yaml.load(f)
    labels = jasper_model_definition['labels']

    # Instantiate neural modules
    data_layer = nemo_asr.AudioToTextDataLayer(
        manifest_filepath=train_dataset,
        labels=labels, batch_size=32)
    data_layer_val = nemo_asr.AudioToTextDataLayer(
        manifest_filepath=eval_datasets,
        labels=labels, batch_size=32, shuffle=False)

    data_preprocessor = nemo_asr.AudioPreprocessing()
    spec_augment = nemo_asr.SpectrogramAugmentation(rect_masks=5)

    jasper_encoder = nemo_asr.JasperEncoder(
        feat_in=64,
        **jasper_model_definition['JasperEncoder'])
    jasper_decoder = nemo_asr.JasperDecoderForCTC(
        feat_in=1024, num_classes=len(labels))
    ctc_loss = nemo_asr.CTCLossNM(num_classes=len(labels))
    greedy_decoder = nemo_asr.GreedyCTCDecoder()

    # Training DAG (Model)
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

    # Validation DAG (Model)
    # We need to instantiate additional data layer neural module
    # for validation data
    audio_signal_v, audio_signal_len_v, transcript_v, transcript_len_v = data_layer_val()
    processed_signal_v, processed_signal_len_v = data_preprocessor(
        input_signal=audio_signal_v, length=audio_signal_len_v)
    # Note that we are not using data-augmentation in validation DAG
    encoded_v, encoded_len_v = jasper_encoder(
        audio_signal=processed_signal_v, length=processed_signal_len_v)
    log_probs_v = jasper_decoder(encoder_output=encoded_v)
    predictions_v = greedy_decoder(log_probs=log_probs_v)
    loss_v = ctc_loss(
        log_probs=log_probs_v, targets=transcript_v,
        input_length=encoded_len_v, target_length=transcript_len_v)

    # These helper functions are needed to print and compute various metrics
    # such as word error rate and log them into tensorboard
    # they are domain-specific and are provided by NeMo's collections
    from nemo_asr.helpers import monitor_asr_train_progress, \
        process_evaluation_batch, process_evaluation_epoch

    from functools import partial
    # Callback to track loss and print predictions during training
    train_callback = nemo.core.SimpleLossLoggerCallback(
        tb_writer=tb_writer,
        # Define the tensors that you want SimpleLossLoggerCallback to
        # operate on
        # Here we want to print our loss, and our word error rate which
        # is a function of our predictions, transcript, and transcript_len
        tensors=[loss, predictions, transcript, transcript_len],
        # To print logs to screen, define a print_func
        print_func=partial(
            monitor_asr_train_progress,
            labels=labels,
            logger=logger
        ))

    saver_callback = nemo.core.CheckpointCallback(
        folder="./",
        # Set how often we want to save checkpoints
        step_freq=100)

    # PRO TIP: while you can only have 1 train DAG, you can have as many
    # val DAGs and callbacks as you want. This is useful if you want to monitor
    # progress on more than one val dataset at once (say LibriSpeech dev clean
    # and dev other)
    eval_callback = nemo.core.EvaluatorCallback(
        eval_tensors=[loss_v, predictions_v, transcript_v, transcript_len_v],
        # how to process evaluation batch - e.g. compute WER
        user_iter_callback=partial(
            process_evaluation_batch,
            labels=labels
            ),
        # how to aggregate statistics (e.g. WER) for the evaluation epoch
        user_epochs_done_callback=partial(
            process_evaluation_epoch, tag="DEV-CLEAN", logger=logger
            ),
        eval_step=500,
        tb_writer=tb_writer)

    # Run training using your Neural Factory
    # Once this "action" is called data starts flowing along train and eval DAGs
    # and computations start to happen
    nf.train(
        # Specify the loss to optimize for
        tensors_to_optimize=[loss],
        # Specify which callbacks you want to run
        callbacks=[train_callback, eval_callback, saver_callback],
        # Specify what optimizer to use
        optimizer="novograd",
        # Specify optimizer parameters such as num_epochs and lr
        optimization_params={
            "num_epochs": 50, "lr": 0.02, "weight_decay": 1e-4
            }
        )

.. note::
    这个脚本在GTX1080上完成50轮训练需要大约7小时

.. tip::
    进一步提升WER:
        (1) 训练的更久
        (2) 训更多的数据
        (3) 用更大的模型
        (4) 在多GPU上训练并且使用混精度训练(on NVIDIA Volta and Turing GPUs)
        (5) 从预训练好的checkpoints上开始训练


混精度训练
-------------------------
NeMo中的混精度和分布式训练上基于 `NVIDIA's APEX library <https://github.com/NVIDIA/apex>`_.
确保它已经安装了。

训混精度训练你只需要设置在 `nemo.core.NeuralModuleFactory` 中设置 `optimization_level` 参数为 `nemo.core.Optimization.mxprO1`. For example:

.. code-block:: python

    nf = nemo.core.NeuralModuleFactory(
        backend=nemo.core.Backend.PyTorch,
        local_rank=args.local_rank,
        optimization_level=nemo.core.Optimization.mxprO1,
        placement=nemo.core.DeviceType.AllGpu,
        cudnn_benchmark=True)

.. note::
    Because mixed precision requires Tensor Cores it only works on NVIDIA Volta and Turing based GPUs
    因为混精度训练需要Tensor Cores, 因此它只能在NVIDIA Volta和Turing架构的GPU上运行。

多GPU训练
-------------------

在NeMo中开启多GPU训练很容易:

   (1) 首先把NeuralModuleFactory中的 `placement` 设置成 `nemo.core.DeviceType.AllGpu`
   (2) 让你的脚本能够接受 'local_rank' 参数， 你自己不要去设置这个参数， 只需要在代码中添加: `parser.add_argument("--local_rank", default=None, type=int)`
   (3) 用 `torch.distributed.launch` 包来运行你的脚本 (把<num_gpus>改成GPU的数量):

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=<num_gpus> <nemo_git_repo_root>/examples/asr/jasper.py ...


大量训练样本例子
~~~~~~~~~~~~~~~~~~~~~~

请参考 `<nemo_git_repo_root>/examples/asr/jasper.py` 为例，做一个更全面的理解。它构建了一个训练的有向无环图，在不同的验证集上构建了多达三个有向无环图。

假设你用的上基于Volta的DGX, 你可以这么运行:

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=<num_gpus> <nemo_git_repo_root>/examples/asr/jasper.py --batch_size=64 --num_epochs=100 --lr=0.015 --warmup_steps=8000 --weight_decay=0.001 --train_dataset=/manifests/librivox-train-all.json --eval_datasets /manifests/librivox-dev-clean.json /manifests/librivox-dev-other.json --model_config=<nemo_git_repo_root>/nemo/examples/asr/configs/quartznet15x5.yaml --exp_name=MyLARGE-ASR-EXPERIMENT

.. tip::
    你可以用逗号分割不同的数据集: `--train_manifest=/manifests/librivox-train-all.json,/manifests/librivox-train-all-sp10pcnt.json,/manifests/cv/validated.json`. Here it combines 3 data sets: LibriSpeech, Mozilla Common Voice and LibriSpeech speed perturbed.


微调（Fine-tuning）
-----------
如果我们从一个好的预训练模型开始训练，训练时间会大大的减小:

    (1) 获取预训练模型 (jasper_encoder, jasper_decoder and configuration files) `from here <https://ngc.nvidia.com/catalog/models/nvidia:quartznet15x5>`_.
    (2) 在你初始化好jasper_encoder和jasper_decoder后，可以这样加载权重:

.. code-block:: python

    jasper_encoder.restore_from("<path_to_checkpoints>/15x5SEP/JasperEncoder-STEP-247400.pt")
    jasper_decoder.restore_from("<path_to_checkpoints>/15x5SEP/JasperDecoderForCTC-STEP-247400.pt")
    # in case of distributed training add args.local_rank
    jasper_decoder.restore_from("<path_to_checkpoints>/15x5SEP/JasperDecoderForCTC-STEP-247400.pt", args.local_rank)

.. tip::
    微调的时候，用小一点的学习率


推理
---------

首先下载预训练模型(jasper_encoder, jasper_decoder and configuration files) `戳这里 <https://ngc.nvidia.com/catalog/models/nvidia:quartznet15x5>`_ 放到 `<path_to_checkpoints>`. 我们会用这个预训练模型在LibriSpeech dev-clean数据集上测试WER.

.. code-block:: bash

    python <nemo_git_repo_root>/examples/asr/jasper_infer.py --model_config=<nemo_git_repo_root>/examples/asr/configs/quartznet15x5.yaml --eval_datasets "<path_to_data>/dev_clean.json" --load_dir=<directory_containing_checkpoints>


用语言模型推理
-----------------------------

用KenLM构建的语言模型
~~~~~~~~~~~
我们会使用 `Baidu's CTC decoder with LM implementation. <https://github.com/PaddlePaddle/DeepSpeech>`_.

请按照下面的步骤:

    * 到scripts目录下 ``cd <nemo_git_repo_root>/scripts``
    * 安装百度CTC解码器 (如果在docker容器中不需要用sudo):
        * ``sudo apt-get update && sudo apt-get install swig``
        * ``sudo apt-get install pkg-config libflac-dev libogg-dev libvorbis-dev libboost-dev``
        * ``sudo apt-get install libsndfile1-dev python-setuptools libboost-all-dev python-dev``
        * ``./install_decoders.sh``
    * 在Librispeech上构建一个6-gram KenLM的语言模型  ``./build_6-gram_OpenSLR_lm.sh``
    * 运行 jasper_infer.py 带上 --lm_path来指定语言模型的路径

    .. code-block:: bash

        python <nemo_git_repo_root>/examples/asr/jasper_infer.py --model_config=<nemo_git_repo_root>/examples/asr/configs/quartznet15x5.yaml --eval_datasets "<path_to_data>/dev_clean.json" --load_dir=<directory_containing_checkpoints> --lm_path=<path_to_6gram.binary>


参考
----------

.. bibliography:: Jasperbib.bib
    :style: plain
