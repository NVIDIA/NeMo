数据集
======

.. _GoogleSpeechCommands_dataset:


谷歌语音指令数据集
------------------

准确地识别语音指令在很多场景都有应用。为了这一目的，谷歌发布了语音指令数据集 :cite:`speech-recognition-dataset-warden2018speech` 。
这一数据集涵盖一些指令的短语音，例如，stop, go, up, down 等等。这些语音来自很多不同的说话人。为了推广这个数据集，谷歌还组织了一次 Kaggle 竞赛。在这项竞赛中，最终获胜的队伍取得了91%的分类准确率。

我们借助 NeMo 中已有的 ASR 模型进行了测试，并发现效果很好。再加上数据增强技术，准确率可以被进一步提升。

版本和预处理
------------

截至目前，谷歌共发布了两个版本的数据集。第一版共包含30个类别共6万5千条数据。第二版包含35个类别共11万条数据。当前我们主要使用第一版数据，以便与其他方法进行比较。

脚本 `process_speech_commands_data.py` 可以被用来对数据集进行处理，以便将其转换为合适的格式。
这个文件位于 `scripts` 文件夹中。你可以设定选项 `--data_root` 来指定数据集的位置，选项 `--data_version` 来指定版本。

还有一个选项 `--rebalance` 可以被用来重新平衡数据集。

.. code-block:: bash

    python process_speech_commands_data.py --data_root=<data directory> --data_version=<1 or 2> {--rebalance}

运行之后，你会得到三个文件： `train_manifest.json` ， `validation_manifest.json` 和 `test_manifest.json`
在文件夹 `{data_root}/google_speech_recognition_v{1/2}` 中。

参考
----

.. bibliography:: speech_recognition_all.bib
    :style: plain
    :labelprefix: SPEECH-RECOGNITION-DATASET
    :keyprefix: speech-recognition-dataset-
