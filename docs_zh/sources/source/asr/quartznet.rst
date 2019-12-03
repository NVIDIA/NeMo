QuartzNet
---------

QuartzNet 是Jasper的一个版本 :cite:`li2019jasper` 它有可分离的卷积和更大的过滤器。它可以获得和Jasper
相似的效果，但是在参数上是远远小于Jasper的。
类似于Jasper, QuartzNet模型的家族可以用 QuartzNet_[BxR] 结构来表示，其中B是块的个数，R表示一个块中卷积子块的个数. 每个子块包含了一个一维的 *可分离* 的卷积层，批归一化层，ReLU和dropout层:

    .. image:: quartz_vertical.png
        :align: center
        :alt: quartznet model
   
    .. note:: 这个checkpoint是在LibriSpeech上训练的，完全在EN Mozilla Common Voice的部分数据集上做的“验证”

我们正在写QuartzNet的论文，不就就会发布。

预训练的模型在 `这里 <https://ngc.nvidia.com/catalog/models/nvidia:quartznet15x5>`_.
