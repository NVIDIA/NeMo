.. _Jasper_model:

Jasper
------

Jasper ("Just Another SPeech Recognizer") :cite:`asr-models-li2019jasper`  is a deep time delay neural network (TDNN) comprising of blocks of 1D-convolutional layers.
Jasper family of models are denoted as Jasper_[BxR] where B is the number of blocks, and R - the number of convolutional sub-blocks within a block. Each sub-block contains a 1-D convolution, batch normalization, ReLU, and dropout:

    .. image:: jasper_vertical.png
        :align: center
        :alt: japer model

Pretrained models can be found at the following links:

============= ======================= =================================================================================
Network       Dataset                 Download Link 
============= ======================= =================================================================================
Jasper10x5dr  Librispeech             `here <https://ngc.nvidia.com/catalog/models/nvidia:jaspernet10x5dr>`__
Jasper10x5dr  | Librispeech,          `here <https://ngc.nvidia.com/catalog/models/nvidia:multidataset_jasper10x5dr>`__
              | Mozilla Common Voice,
              | WSJ,
              | Fisher,
              | Switchboard
Jasper15x5SEP Aishell2                `here <https://ngc.nvidia.com/catalog/models/nvidia:aishell2_jasper10x5dr>`__
============= ======================= =================================================================================
