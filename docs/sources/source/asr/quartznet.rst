.. _Quartznet_model:

QuartzNet
---------

QuartzNet :cite:`qtz-models-kriman2019quartznet` is a version of Jasper :cite:`qtz-models-li2019jasper` model with separable convolutions and larger filters. It can achieve performance
similar to Jasper but with an order of magnitude less parameters.
Similarly to Jasper, QuartzNet family of models are denoted as QuartzNet_[BxR] where B is the number of blocks, and R - the number of convolutional sub-blocks within a block. Each sub-block contains a 1-D *separable* convolution, batch normalization, ReLU, and dropout:

    .. image:: quartz_vertical.png
        :align: center
        :alt: quartznet model
   
`QuartzNet paper <https://arxiv.org/abs/1910.10261>`_.

Pretrained models can be found at the following links:

============= ===================== ==============================================================================
Network       Dataset               Download Link 
============= ===================== ==============================================================================
QuartzNet15x5 Librispeech,          `here <https://ngc.nvidia.com/catalog/models/nvidia:quartznet15x5>`__
              Mozilla Common Voice
QuartzNet15x5 Aishell2              `here <https://ngc.nvidia.com/catalog/models/nvidia:aishell2_quartznet15x5>`__
============= ===================== ==============================================================================

References
^^^^^^^^^^

.. bibliography:: asr_all.bib
    :style: plain
    :labelprefix: QTZ-MODELS
    :keyprefix: qtz-models-

