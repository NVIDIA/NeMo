QuartzNet
---------

QuartzNet 是 Jasper模型 :cite:`asr-models-li2019jasper` 的一个版本, 它有可分离的卷积和更大的过滤器。它可以获得和 Jasper
相似的效果，但是在参数上是远远小于 Jasper 的。
类似于 Jasper, QuartzNet 模型的家族可以用 QuartzNet_[BxR] 结构来表示，其中 B 是块的个数，R 表示一个块中卷积子块的个数。每个子块包含了一个一维的 *可分离* 的卷积层，批归一化层，ReLU和dropout层：

    .. image:: quartz_vertical.png
        :align: center
        :alt: quartznet model
   
    .. note:: 这个checkpoint是在LibriSpeech上训练的，完全在EN Mozilla Common Voice的部分数据集上做的“验证”

我们正在写 QuartzNet 的论文，不久就会发布。

预训练的模型在 `这里 <https://ngc.nvidia.com/catalog/models/nvidia:quartznet15x5>`_ 。
