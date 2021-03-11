Models
======

This page gives a brief overview of the models that NeMo's ASR collection currently supports.

Each of these models can be used with the example ASR scripts (in the ``<NeMo_git_root>/examples/asr`` directory) by
specifying the model architecture in the config file used.
Examples of config files for each model can be found in the ``<NeMo_git_root>/examples/asr/conf`` directory.

For more information about the config files and how they should be structured, see the :doc:`./configs` page.

Pretrained checkpoints for all of these models, as well as instructions on how to load them, can be found on the :doc:`./results` page.
You can use the available checkpoints for immediate inference, or fine-tune them on your own datasets.
The Checkpoints page also contains benchmark results for the available ASR models.

.. _Jasper_model:

Jasper
------

Jasper ("Just Another SPeech Recognizer") :cite:`asr-models-li2019jasper`  is a deep time delay neural network (TDNN) comprising of blocks of 1D-convolutional layers.
The Jasper family of models are denoted as Jasper_[BxR] where B is the number of blocks, and R is the number of convolutional sub-blocks within a block. Each sub-block contains a 1-D convolution, batch normalization, ReLU, and dropout:

    .. image:: images/jasper_vertical.png
        :align: center
        :alt: japer model
        :scale: 50%

Jasper models can be instantiated using the :class:`EncDecCTCModel<nemo.collections.asr.models.EncDecCTCModel>` class.


QuartzNet
---------

QuartzNet :cite:`asr-models-kriman2019quartznet` is a version of Jasper :cite:`asr-models-li2019jasper` model with separable convolutions and larger filters. It can achieve performance
similar to Jasper but with an order of magnitude fewer parameters.
Similarly to Jasper, the QuartzNet family of models are denoted as QuartzNet_[BxR] where B is the number of blocks, and R is the number of convolutional sub-blocks within a block. Each sub-block contains a 1-D *separable* convolution, batch normalization, ReLU, and dropout:

    .. image:: images/quartz_vertical.png
        :align: center
        :alt: quartznet model
        :scale: 40%


QuartzNet models can be instantiated using the :class:`EncDecCTCModel<nemo.collections.asr.models.EncDecCTCModel>` class.


Citrinet
--------

Citrinet is a version of QuartzNet :cite:`asr-models-kriman2019quartznet` that extends ContextNet :cite:`asr-models-han2020contextnet`,
utilizing subword encoding (via Word Piece tokenization) and Squeeze-and-Excitation mechanism :cite:`asr-models-hu2018squeeze` to
obtain highly accurate audio transcripts while utilizing a non-autoregressive CTC based decoding scheme for efficient inference.

    .. image:: images/citrinet_vertical.png
        :align: center
        :alt: citrinet model
        :scale: 50%

Citrinet models can be instantiated using the :class:`EncDecCTCModelBPE<nemo.collections.asr.models.EncDecCTCModelBPE>` class.

References
----------

.. bibliography:: asr_all.bib
    :style: plain
    :labelprefix: ASR-MODELS
    :keyprefix: asr-models-
