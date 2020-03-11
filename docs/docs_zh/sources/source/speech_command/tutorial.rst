教程
====

请首先安装 ``nemo`` 和 ``nemo_asr`` 集合。
具体的安装步骤请参阅 :ref:`installation` 章节。
另外，本教程还使用 python 包 `torchaudio` 进行语音特征提取。


入门
----

本教程基于 QuartzNet :cite:`speech-recognition-tut-kriman2019quartznet` 模型。其中的解码器部分做了些许修改以适配分类任务。

1. 音频处理（特征提取）：包括信号归一化，滑窗处理，频谱转换（或者是梅尔频谱 MFCC ）
2. 使用 SpecAugment :cite:`speech-recognition-tut-park2019` 进行数据增强，同时这一方法也能增加数据量。
3. 创建一个小型的神经网络模型进行训练。

数据准备
--------

我们使用开源的谷歌语音指令数据集 Google Speech Commands Dataset 。目前，我们使用的是第一版数据集。如果想使用第二版，还需要一些简单的修改。下面的命令可以下载数据并进行相应的格式转换。

.. code-block:: bash

    mkdir data
    # process_speech_commands_data.py script is located under <nemo_git_repo_root>/scripts
    # The `--rebalance` flag will duplicate elements in the train set so that all classes
    # have the same number of elements. It is not mandatory to add this flag.
    python process_speech_commands_data.py --data_root=data --data_version=1 --rebalance

.. note::
    如果使用第一版数据集 ``--data_version=1`` ，至少需要 4GB 的硬盘空间。如果使用第二版 ``--data_version=2`` 至少需要 16 GB 的硬盘空间。另外，下载和处理的过程均需要一些时间。

下载和处理完成后，你会得到一个名为 `data` 文件夹，其中包含另一个文件夹名称为 `google_speech_recognition_v{1/2}` 。
在这个文件夹中会有多个子目录包含很多 wav 文件和三个 json 文件，分别是

* `train_manifest.json`
* `validation_manifest.json`
* `test_manifest.json`

json文件的每一行代表一条训练数据。其中， `audio_filepath` 属性是音频文件的地址， `duration` 是音频时间长度， `command` 是音频的标注。

.. code-block:: json

    {"audio_filepath": "<absolute path to dataset>/two/8aa35b0c_nohash_0.wav", "duration": 1.0, "command": "two"}
    {"audio_filepath": "<absolute path to dataset>/two/ec5ab5d5_nohash_2.wav", "duration": 1.0, "command": "two"}


训练
----

我们使用的是 QuartzNet 模型 :cite:`speech-recognition-tut-kriman2019quartznet` 。
相比于 Jasper 模型, QuartzNet 模型中使用了可分离卷积 (Separable Convolutions) ，大幅减少了参数数量。

QuartzNet 模型使用一种固定的模型定义模式： QuartzNet-[BxR], 其中 B 是模块的数量，R 是卷积子模块的数量。每个子模块包含一个 1D 掩码卷积，批归一化， ReLU 激活和 dropout 。

    .. image:: quartz_vertical.png
        :align: center
        :alt: quartznet model

本教程中我们使用的是 QuartzNet [3x1] 模型。
接下来的脚本会在一个 GPU 上进行训练和评估。

    .. tip::
        借助 Jupyter 笔记本一步一步地运行这个脚本。

**训练脚本**

.. code-block:: python

    # Import some utility functions
    import argparse
    import copy
    import math
    import os
    import glob
    from functools import partial
    from datetime import datetime
    from ruamel.yaml import YAML

    # NeMo's "core" package
    import nemo
    # NeMo's ASR collection
    import nemo.collections.asr as nemo_asr
    # NeMo's learning rate policy
    from nemo.utils.lr_policies import CosineAnnealing
    from nemo.collections.asr.helpers import (
        monitor_classification_training_progress,
        process_classification_evaluation_batch,
        process_classification_evaluation_epoch,
    )

    logging = nemo.logging

    # Lets define some hyper parameters
    lr = 0.05
    num_epochs = 100
    batch_size = 128
    weight_decay = 0.001

    # Create a Neural Factory
    # It creates log files and tensorboard writers for us among other functions
    neural_factory = nemo.core.NeuralModuleFactory(
        log_dir='./quartznet-3x1-v1',
        create_tb_writer=True)
    tb_writer = neural_factory.tb_writer

    # Path to our training manifest
    train_dataset = "<path_to_where_you_put_data>/train_manifest.json"

    # Path to our validation manifest
    eval_datasets = "<path_to_where_you_put_data>/test_manifest.json"

    # Here we will be using separable convolutions
    # with 3 blocks (k=3 repeated once r=1 from the picture above)
    yaml = YAML(typ="safe")
    with open("<nemo_git_repo_root>/examples/asr/configs/quartznet_speech_commands_3x1_v1.yaml") as f:
        jasper_params = yaml.load(f)

    # Pre-define a set of labels that this model must learn to predict
    labels = jasper_params['labels']

    # Get the sampling rate of the data
    sample_rate = jasper_params['sample_rate']

    # Check if data augmentation such as white noise and time shift augmentation should be used
    audio_augmentor = jasper_params.get('AudioAugmentor', None)

    # Build the input data layer and the preprocessing layers for the train set
    train_data_layer = nemo_asr.AudioToSpeechLabelDataLayer(
        manifest_filepath=train_dataset,
        labels=labels,
        sample_rate=sample_rate,
        batch_size=batch_size,
        num_workers=os.cpu_count(),
        augmentor=audio_augmentor,
        shuffle=True
    )

     # Build the input data layer and the preprocessing layers for the test set
    eval_data_layer = nemo_asr.AudioToSpeechLabelDataLayer(
        manifest_filepath=eval_datasets,
        sample_rate=sample_rate,
        labels=labels,
        batch_size=args.eval_batch_size,
        num_workers=os.cpu_count(),
        shuffle=False,
    )

    # We will convert the raw audio data into MFCC Features to feed as input to our model
    data_preprocessor = nemo_asr.AudioToMFCCPreprocessor(
        sample_rate=sample_rate, **jasper_params["AudioToMFCCPreprocessor"],
    )

    # Compute the total number of samples and the number of training steps per epoch
    N = len(train_data_layer)
    steps_per_epoch = math.ceil(N / float(args.batch_size))

    logging.info("Steps per epoch : {0}".format(steps_per_epoch))
    logging.info('Have {0} examples to train on.'.format(N))

    # Here we begin defining all of the augmentations we want
    # We will pad the preprocessed spectrogram image to have a certain number of timesteps
    # This centers the generated spectrogram and adds black boundaries to either side
    # of the padded image.
    crop_pad_augmentation = nemo_asr.CropOrPadSpectrogramAugmentation(audio_length=128)

    # We also optionally add `SpecAugment` augmentations based on the config file
    # SpecAugment has various possible augmentations to the generated spectrogram
    # 1) Frequency band masking
    # 2) Time band masking
    # 3) Rectangular cutout
    spectr_augment_config = jasper_params.get('SpectrogramAugmentation', None)
    if spectr_augment_config:
        data_spectr_augmentation = nemo_asr.SpectrogramAugmentation(**spectr_augment_config)

    # Build the QuartzNet Encoder model
    # The config defines the layers as a list of dictionaries
    # The first and last two blocks are not considered when we say QuartzNet-[BxR]
    # B is counted as the number of blocks after the first layer and before the penultimate layer.
    # R is defined as the number of repetitions of each block in B.
    # Note: We can scale the convolution kernels size by the float parameter `kernel_size_factor`
    jasper_encoder = nemo_asr.JasperEncoder(**jasper_params["JasperEncoder"])

    # We then define the QuartzNet decoder.
    # This decoder head is specialized for the task for classification, such that it
    # accepts a set of `N-feat` per timestep of the model, and averages these features
    # over all the timesteps, before passing a Linear classification layer on those features.
    jasper_decoder = nemo_asr.JasperDecoderForClassification(
        feat_in=jasper_params["JasperEncoder"]["jasper"][-1]["filters"],
        num_classes=len(labels),
        **jasper_params['JasperDecoderForClassification'],
    )

    # We can easily apply cross entropy loss to train this model
    ce_loss = nemo_asr.CrossEntropyLossNM()

    # Lets print out the number of parameters of this model
    logging.info('================================')
    logging.info(f"Number of parameters in encoder: {jasper_encoder.num_weights}")
    logging.info(f"Number of parameters in decoder: {jasper_decoder.num_weights}")
    logging.info(
        f"Total number of parameters in model: " f"{jasper_decoder.num_weights + jasper_encoder.num_weights}"
    )
    logging.info('================================')

    # Now we have all of the components that are required to build the NeMo execution graph!
    ## Build the training data loaders and preprocessors first
    audio_signal, audio_signal_len, commands, command_len = train_data_layer()
    processed_signal, processed_signal_len = data_preprocessor(input_signal=audio_signal, length=audio_signal_len)
    processed_signal, processed_signal_len = crop_pad_augmentation(
        input_signal=processed_signal,
        length=audio_signal_len
    )

    ## Augment the dataset for training
    if spectr_augment_config:
        processed_signal = data_spectr_augmentation(input_spec=processed_signal)

    ## Define the model
    encoded, encoded_len = jasper_encoder(audio_signal=processed_signal, length=processed_signal_len)
    decoded = jasper_decoder(encoder_output=encoded)

    ## Obtain the train loss
    train_loss = ce_loss(logits=decoded, labels=commands)

    # Now we build the test graph in a similar way, reusing the above components
    ## Build the test data loader and preprocess same way as train graph
    ## But note, we do not add the spectrogram augmentation to the test graph !
    test_audio_signal, test_audio_signal_len, test_commands, test_command_len = eval_data_layer()
    test_processed_signal, test_processed_signal_len = data_preprocessor(
        input_signal=test_audio_signal, length=test_audio_signal_len
    )
    test_processed_signal, test_processed_signal_len = crop_pad_augmentation(
        input_signal=test_processed_signal, length=test_processed_signal_len
    )

    # Pass the test data through the model encoder and decoder
    test_encoded, test_encoded_len = jasper_encoder(
        audio_signal=test_processed_signal, length=test_processed_signal_len
    )
    test_decoded = jasper_decoder(encoder_output=test_encoded)

    # Compute test loss for visualization
    test_loss = ce_loss(logits=test_decoded, labels=test_commands)

    # Now that we have our training and evaluation graphs built,
    # we can focus on a few callbacks to help us save the model checkpoints
    # during training, as well as display train and test metrics

    # Callbacks needed to print train info to console and Tensorboard
    train_callback = nemo.core.SimpleLossLoggerCallback(
        # Notice that we pass in loss, predictions, and the labels.
        # Of course we would like to see our training loss, but we need the
        # other arguments to calculate the accuracy.
        tensors=[train_loss, decoded, commands],
        # The print_func defines what gets printed.
        print_func=partial(monitor_classification_training_progress, eval_metric=None),
        get_tb_values=lambda x: [("loss", x[0])],
        tb_writer=neural_factory.tb_writer,
    )

    # Callbacks needed to print test info to console and Tensorboard
    tagname = 'TestSet'
    eval_callback = nemo.core.EvaluatorCallback(
        eval_tensors=[test_loss, test_decoded, test_commands],
        user_iter_callback=partial(process_classification_evaluation_batch, top_k=1),
        user_epochs_done_callback=partial(process_classification_evaluation_epoch, eval_metric=1, tag=tagname),
        eval_step=200,  # How often we evaluate the model on the test set
        tb_writer=neural_factory.tb_writer,
    )

    # Callback to save model checkpoints
    chpt_callback = nemo.core.CheckpointCallback(
        folder=neural_factory.checkpoint_dir,
        step_freq=1000,
    )

    # Prepare a list of checkpoints to pass to the engine
    callbacks = [train_callback, eval_callback, chpt_callback]

    # Now we have all the components required to train the model
    # Lets define a learning rate schedule

    # Define a learning rate schedule
    lr_policy = CosineAnnealing(
        total_steps=num_epochs * steps_per_epoch,
        warmup_ratio=0.05,
        min_lr=0.001,
    )

    logging.info(f"Using `{lr_policy}` Learning Rate Scheduler")

    # Finally, lets train this model !
    neural_factory.train(
        tensors_to_optimize=[train_loss],
        callbacks=callbacks,
        lr_policy=lr_policy,
        optimizer="novograd",
        optimization_params={
            "num_epochs": num_epochs,
            "max_steps": None,
            "lr": lr,
            "momentum": 0.95,
            "betas": (0.98, 0.5),
            "weight_decay": weight_decay,
            "grad_norm_clip": None,
        },
        batches_per_step=1,
    )

.. note::
    整个训练过程大概需要 100 个 epoch ，在 GTX 1080 GPU 上大概需要 4-5 小时。

.. tip::
    想要进一步提升准确率，可以尝试下列方法：
        (1) 更长时间的训练 (200-300 epochs)
        (2) 使用更多的数据
        (3) 选择更大的模型
        (4) 使用多个 GPU 或者使用混合精度训练
        (5) 使用一个预训练的模型

混合精度训练
------------

可以借助英伟达的 `APEX 工具包 <https://github.com/NVIDIA/apex>`_ 进行混合精度训练和分布式训练。
要进行混合精度训练，你只需要设置 `optimization_level` 选项为 `nemo.core.Optimization.mxprO1` 。例如：

.. code-block:: python

    nf = nemo.core.NeuralModuleFactory(
        backend=nemo.core.Backend.PyTorch,
        local_rank=args.local_rank,
        optimization_level=nemo.core.Optimization.mxprO1,
        placement=nemo.core.DeviceType.AllGpu,
        cudnn_benchmark=True)


多 GPU 训练
-----------

在 NeMo 中进行多 GPU 训练也非常容易：

   (1) 将 `NeuralModuleFactory` 类的 `placement` 选项设置为 `nemo.core.DeviceType.AllGpu`
   (2) 添加命令行选项 `local_rank` : `parser.add_argument("--local_rank", default=None, type=int)`
   (3) 导入 `torch.distributed.launch` 包并且使用如下的方式运行脚本：

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=<num_gpus> <nemo_git_repo_root>/examples/asr/quartznet_speech_commands.py ...

.. note::
    混合精度训练依赖于 Tensor Cores 硬件单元，所以当前只支持英伟达 Volta 和 Turing 架构 GPU


完整的训练案例
~~~~~~~~~~~~~~

更详细的一个训练案例请参阅文件 `<nemo_git_repo_root>/examples/asr/quartznet_speech_commands.py` 。
在这个案例中，我们分别构建了训练，评估和测试的计算图。
下面的这条命令会启动8个 GPU 并进行混合精度训练。其中的 json 文件指定了数据集信息。

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=<num_gpus> <nemo_git_repo_root>/examples/asr/quartznet_speech_commands.py --model_config "<nemo_git_repo_root>/examples/asr/configs/quartznet_speech_commands_3x1_v1.yaml" \
      --train_dataset="<absolute path to dataset>/train_manifest.json" --eval_datasets "<absolute path to dataset>/validation_manifest.json" "<absolute path to dataset>/test_manifest.json" \
      --num_epochs=200 --batch_size=128 --eval_batch_size=128 --eval_freq=200 --lr=0.05 --min_lr=0.001 \
      --optimizer="novograd" --weight_decay=0.001 --amp_opt_level="O1" --warmup_ratio=0.05 --hold_ratio=0.45 \
      --checkpoint_dir="./checkpoints/quartznet_speech_commands_checkpoints_3x1_v1/" \
      --exp_name="./results/quartznet_speech_classification-quartznet-3x1_v1/"

.. tip::
    你还可以同时输入多个 json 文件，以便在多个数据集上进行训练。例如： `--train_manifest=/manifests/<first dataset>.json,/manifests/<second dataset>.json`


微调 (Fine-tuning)
------------------

如果使用一个预训练好的模型，那么训练时间可以被大大缩短：
1. 准备一个预训练模型，包含 jasper_encoder, jasper_decoder 和配置文件。
2. 载入模型权重，使用类似于下面这样的代码：

.. code-block:: python

    jasper_encoder.restore_from("<path_to_checkpoints>/JasperEncoder-STEP-89000.pt")
    jasper_decoder.restore_from("<path_to_checkpoints>/JasperDecoderForClassification-STEP-89000.pt")
    # in case of distributed training add args.local_rank
    jasper_decoder.restore_from("<path_to_checkpoints>/JasperDecoderForClassification-STEP-89000.pt", args.local_rank)

.. tip::
    微调的时候，最好降低学习率。


评估
----

我们可以下载预训练模型，并用它在谷歌语音指令数据集上检验分类准确率。

.. note::
    如果你想亲自听一下数据集中的音频，你可以在 notebook 里运行下面的这份代码。

.. code-block:: python

    # Lets add some generic imports.
    # Please note that you will need to install `librosa` for this code
    # To install librosa : Run `!pip install librosa` from the notebook itself.
    import glob
    import os
    import json
    import re
    import numpy as np
    import torch
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import IPython.display as ipd
    from ruamel.yaml import YAML

    # Import nemo and asr collections
    import nemo
    import nemo.collections.asr as nemo_asr

    logging = nemo.logging

    # We add some
    data_dir = '<path to the data directory>'
    data_version = 1
    config_path = '<path to the config file for this model>'
    model_path = '<path to the checkpoint directory for this model>'

    test_manifest = os.path.join(data_dir, "test_manifest.json")

    # Parse the config file provided to us
    # Parse config and pass to model building function
    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)
        logging.info("******\nLoaded config file.\n******")

    labels = params['labels']  # Vocab of tokens
    sample_rate = params['sample_rate']
    batch_size = 128

    # Build the evaluation graph
    # Create our NeuralModuleFactory, which will oversee the neural modules.
    neural_factory = nemo.core.NeuralModuleFactory(
        log_dir=f'v{data_version}/eval_results/')

    logger = neural_factory.logger

    test_data_layer = nemo_asr.AudioToSpeechLabelDataLayer(
        manifest_filepath=test_manifest,
        labels=labels,
        sample_rate=sample_rate,
        shuffle=False,
        batch_size=batch_size,
    )
    crop_pad_augmentation = nemo_asr.CropOrPadSpectrogramAugmentation(
        audio_length=128
    )
    data_preprocessor = nemo_asr.AudioToMFCCPreprocessor(
        sample_rate=sample_rate,
        **params['AudioToMFCCPreprocessor']
    )

    # Create the Jasper_3x1 encoder as specified, and a classification decoder
    encoder = nemo_asr.JasperEncoder(**params['JasperEncoder'])
    decoder = nemo_asr.JasperDecoderForClassification(
        feat_in=params['JasperEncoder']['jasper'][-1]['filters'],
        num_classes=len(labels),
        **params['JasperDecoderForClassification']
    )

    ce_loss = nemo_asr.CrossEntropyLossNM()

    # Assemble the DAG components
    test_audio_signal, test_audio_signal_len, test_commands, test_command_len = test_data_layer()

    test_processed_signal, test_processed_signal_len = data_preprocessor(
        input_signal=test_audio_signal,
        length=test_audio_signal_len
    )

    # --- Crop And Pad Augment --- #
    test_processed_signal, test_processed_signal_len = crop_pad_augmentation(
        input_signal=test_processed_signal,
        length=test_processed_signal_len
    )

    test_encoded, test_encoded_len = encoder(
        audio_signal=test_processed_signal,
        length=test_processed_signal_len
    )

    test_decoded = decoder(
        encoder_output=test_encoded
    )

    test_loss = ce_loss(
        logits=test_decoded,
        labels=test_commands
    )

    # We import the classification accuracy metric to compute Top-1 accuracy
    from nemo.collections.asr.metrics import classification_accuracy
    from functools import partial

    # --- Inference Only --- #
    # We've already built the inference DAG above, so all we need is to call infer().
    evaluated_tensors = neural_factory.infer(
        # These are the tensors we want to get from the model.
        tensors=[test_loss, test_decoded, test_commands],
        # checkpoint_dir specifies where the model params are loaded from.
        checkpoint_dir=model_path
        )

    # Let us count the total number of incorrect classifications by this model
    correct_count = 0
    total_count = 0

    for batch_idx, (logits, labels) in enumerate(zip(evaluated_tensors[1], evaluated_tensors[2])):
        acc = classification_accuracy(
            logits=logits,
            targets=labels,
            top_k=[1]
        )

        # Select top 1 accuracy only
        acc = acc[0]

        # Since accuracy here is "per batch", we simply denormalize it by multiplying
        # by batch size to recover the count of correct samples.
        correct_count += int(acc * logits.size(0))
        total_count += logits.size(0)

    logging.info(f"Total correct / Total count : {correct_count} / {total_count}")
    logging.info(f"Final accuracy : {correct_count / float(total_count)}")

    # Let us now filter out the incorrectly labeled samples from the total set of samples in the test set

    # First lets create a utility class to remap the integer class labels to actual string label
    class ReverseMapLabel:
        def __init__(self, data_layer: nemo_asr.AudioToSpeechLabelDataLayer):
            self.label2id = dict(data_layer._dataset.label2id)
            self.id2label = dict(data_layer._dataset.id2label)

        def __call__(self, pred_idx, label_idx):
            return self.id2label[pred_idx], self.id2label[label_idx]

    # Next, lets get the indices of all the incorrectly labeled samples
    sample_idx = 0
    incorrect_preds = []
    rev_map = ReverseMapLabel(test_data_layer)

    for batch_idx, (logits, labels) in enumerate(zip(evaluated_tensors[1], evaluated_tensors[2])):
        probs = torch.softmax(logits, dim=-1)
        probas, preds = torch.max(probs, dim=-1)

        incorrect_ids = (preds != labels).nonzero()
        for idx in incorrect_ids:
            proba = float(probas[idx][0])
            pred = int(preds[idx][0])
            label = int(labels[idx][0])
            idx = int(idx[0]) + sample_idx

            incorrect_preds.append((idx, *rev_map(pred, label), proba))

        sample_idx += labels.size(0)

    logging.info(f"Num test samples : {total_count}")
    logging.info(f"Num errors : {len(incorrect_preds)}")

    # First lets sort by confidence of prediction
    incorrect_preds = sorted(incorrect_preds, key=lambda x: x[-1], reverse=False)

    # Lets print out the (test id, predicted label, ground truth label, confidence)
    # tuple of first 20 incorrectly labeled samples
    for incorrect_sample in incorrect_preds[:20]:
        logging.info(str(incorrect_sample))

    # Lets define a threshold below which we designate a model's prediction as "low confidence"
    # and then filter out how many such samples exist
    low_confidence_threshold = 0.25
    count_low_confidence = len(list(filter(lambda x: x[-1] <= low_confidence_threshold, incorrect_preds)))
    logging.info(f"Number of low confidence predictions : {count_low_confidence}")

    # One interesting observation is to actually listen to these samples whose predicted labels were incorrect
    # Note: The following requires the use of a Notebook environment

    # First lets create a helper function to parse the manifest files
    def parse_manifest(manifest):
        data = []
        for line in manifest:
            line = json.loads(line)
            data.append(line)

        return data

    # Now lets load the test manifest into memory
    test_samples = []
    with open(test_manifest, 'r') as test_f:
        test_samples = test_f.readlines()

    test_samples = parse_manifest(test_samples)

    # Next, lets create a helper function to actually listen to certain samples
    def listen_to_file(sample_id, pred=None, label=None, proba=None):
        # Load the audio waveform using librosa
        filepath = test_samples[sample_id]['audio_filepath']
        audio, sample_rate = librosa.load(filepath)

        if pred is not None and label is not None and proba is not None:
            logging.info(f"Sample : {sample_id} Prediction : {pred} Label : {label} Confidence = {proba: 0.4f}")
        else:
            logging.info(f"Sample : {sample_id}")

        return ipd.Audio(audio, rate=sample_rate)

    # Finally, lets listen to all the audio samples where the model made a mistake
    # Note: This list of incorrect samples may be quite large, so you may choose to subsample `incorrect_preds`
    for sample_id, pred, label, proba in incorrect_preds:
        ipd.display(listen_to_file(sample_id, pred=pred, label=label, proba=proba))  # Needs to be run in a notebook environment

参考
----

.. bibliography:: speech_recognition_all.bib
    :style: plain
    :labelprefix: SPEECH-RECOGNITION-ALL-TUT
    :keyprefix: speech-recognition-tut-
