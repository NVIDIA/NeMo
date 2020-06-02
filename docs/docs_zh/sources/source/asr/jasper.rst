.. _Jasper_model:

Jasper
------

Jasper（Just Another SPeech Recognizer) :cite:`asr-models-li2019jasper` 是一个深度时延网络（TDNN），由包含一维卷积层的块组成。
Jasper 家族的模型可以用 Jasper_[BxR] 来表示，其中 B 是块的个数, R 是一个块中卷积子块的个数。每个卷积子块中包含一个一维卷积，批归一化层，ReLU 和 dropout 层：

    .. image:: jasper_vertical.png
        :align: center
        :alt: japer model

预训练的模型如下：

============= ======================= =================================================================================
Network       Dataset                 Download Link 
============= ======================= =================================================================================
Jasper10x5dr  Librispeech             `这里 <https://ngc.nvidia.com/catalog/models/nvidia:jaspernet10x5dr>`__
Jasper10x5dr  | Librispeech,          `这里 <https://ngc.nvidia.com/catalog/models/nvidia:multidataset_jasper10x5dr>`__
              | Mozilla Common Voice,
              | WSJ,
              | Fisher,
              | Switchboard
Jasper15x5SEP Aishell2                `这里 <https://ngc.nvidia.com/catalog/models/nvidia:aishell2_jasper10x5dr>`__
============= ======================= =================================================================================
