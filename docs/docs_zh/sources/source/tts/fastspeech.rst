.. _fastspeech:

Fast Speech
===========

模型
----
这个模型基于
`Fast Speech 模型 <https://www.microsoft.com/en-us/research/blog/fastspeech-new-text-to-speech-model-improves-on-speed-accuracy-and-controllability>`_
(另可见 `此文献 <https://arxiv.org/abs/1905.09263>`_)。

Fast Speech 包含两个不同的阶段：持续时间（durations）抽取 和 实际训练。

持续时间抽取
++++++++++++

第一个阶段是持续时间的抽取，首先，对于每一个输入数据集中的字符，你应该获得一个表示其持续时长的整型数值，该数值对应着音频样本中该字符持续的时间步数量。
对此，NeMo 使用从 Tacotron 2 推理时提取的输入字符与梅尔谱的对齐映射矩阵（alignment map）来指导训练。
对每一个时间步，我们都将该时间步在对齐映射矩阵中最强信号值对应的字符的持续时间增加一个单位。

想要完成以上步骤，请运行位于 NeMo/examples/tts 的 fastspeech_alignments.py 文件，并指定以下参数（提供存储 durations 的路径）：

.. code-block:: bash

    python fastspeech_durations.py --spec_model=tacotron2 --spec_model_config=configs/tacotron2.yaml --spec_model_load_dir=<directory_with_tacotron2_checkopints> --eval_dataset=<data_root>/ljspeech_train.json --durations_dir=<data_root>/durs

Fast Speech 训练
++++++++++++++++

第二个阶段是实际模型的训练。 NeMo 将 fast speech 中所有 梅尔谱合成 以及 持续时间计算 的逻辑都打包在一个对应名称的神经模块中。
FastSpeechLoss 会根据其输出计算损失值。

要使用上一步骤中抽取的 librispeech 数据的持续时间来开始训练，请执行以下命令：

.. code-block:: bash

    python fastspeech.py --model_config=configs/fastspeech.yaml --train_dataset=<data_root>/ljspeech_train.json --durations_dir=<data_root>/durs
