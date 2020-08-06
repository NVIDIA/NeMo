Models
======

Currently, NeMo's ASR collection supports the following models:

.. _Jasper_model:

Jasper
------

Jasper ("Just Another SPeech Recognizer") :cite:`asr-models-li2019jasper`  is a deep time delay neural network (TDNN) comprising of blocks of 1D-convolutional layers.
Jasper family of models are denoted as Jasper_[BxR] where B is the number of blocks, and R - the number of convolutional sub-blocks within a block. Each sub-block contains a 1-D convolution, batch normalization, ReLU, and dropout:

    .. image:: jasper_vertical.png
        :align: center
        :alt: japer model


QuartzNet
---------

QuartzNet :cite:`asr-models-kriman2019quartznet` is a version of Jasper :cite:`asr-models-li2019jasper` model with separable convolutions and larger filters. It can achieve performance
similar to Jasper but with an order of magnitude less parameters.
Similarly to Jasper, QuartzNet family of models are denoted as QuartzNet_[BxR] where B is the number of blocks, and R - the number of convolutional sub-blocks within a block. Each sub-block contains a 1-D *separable* convolution, batch normalization, ReLU, and dropout:

    .. image:: quartz_vertical.png
        :align: center
        :alt: quartznet model


Jasper and QuartzNet models can be instantiated using :class:`EncDecCTCModel<nemo.collections.asr.models.EncDecCTCModel>` class.



References
----------

.. bibliography:: asr_all.bib
    :style: plain
    :labelprefix: ASR-MODELS
    :keyprefix: asr-models-
