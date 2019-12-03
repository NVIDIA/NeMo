WaveGlow
========

`WaveGlow <https://arxiv.org/abs/1811.00002>`_ 是一个通用的基于标准化流来合成音频的神经网络声码器。
它能够将梅尔频谱转化为音频波形。

NeMo 当前并未将 WaveGlow 分解为多个神经模块的组合，而是将整个 WaveGlow 模型作为一个神经模块。

小建议
~~~~~~~
我们提供的预训练 Waveglow 模型应该能作为绝大部分语言的声码器。
当然，你也可以通过运行位于 waveglow.py 脚本来训练自己的 WaveGlow 声码器。
该脚本的运行方式如下所示：

.. code-block:: bash

    python waveglow.py --train_dataset=<data_root>/ljspeech_train.json --eval_datasets <data_root>/ljspeech_eval.json --model_config=configs/waveglow.yaml --num_epochs=1500

请注意，训练 WaveGlow 相对于训练 Tacotron 2 需要花费更长时间。
