QuartzNet
---------

QuartzNet 模型相当于使用了可分离卷积和更大卷积核的 Jasper 模型 :cite:`asr-models-li2019jasper` 。两个模型都能达到相似的准确率，但是 QuatzNet 模型的参数量要少一个数量级。
与 Jasper 模型类似，QuartzNet 模型规格使用 QuartzNet_[BxR] 来表示，其中 B 表示模块的数量，R 表示卷积子模块的数量。

我们使用这些模型在谷歌语音指令数据集上进行训练。

.. image:: quartz_vertical.png
    :align: center
    :alt: quartznet model

关于 QuartzNet 模型的详细信息可以参阅 `QuartzNet <https://arxiv.org/abs/1910.10261>`_ 。

我们使用2个 GPU 进行了200 epochs 的混合精度训练，其中，batch size 设为128。整个训练大概需要1个小时。

=============================== ===================== ============
Network                         Dataset               Results
=============================== ===================== ============
QuartzNet3x1 (77k params)       Speech Commands V1    97.46% Test

QuartzNet3x2 (93k params)       Speech Commands V2    97.35% Test
=============================== ===================== ============


参考
----

.. bibliography:: speech_recognition_all.bib
    :style: plain
    :labelprefix: SPEECH-RECOGNITION-MODELS
    :keyprefix: speech-recognition-models-
