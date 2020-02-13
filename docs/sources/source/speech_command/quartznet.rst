QuartzNet
---------

QuartzNet is a version of Jasper :cite:`asr-models-li2019jasper` model with separable convolutions and larger filters. It can achieve performance
similar to Jasper but with an order of magnitude less parameters.
Similarly to Jasper, QuartzNet family of models are denoted as QuartzNet_[BxR] where B is the number of blocks, and R - the number of convolutional sub-blocks within a block. Each sub-block contains a 1-D *separable* convolution, batch normalization, ReLU, and dropout:

These models are trained on Google Speech Commands dataset (V1 - all 30 classes).

    .. image:: quartz_vertical.png
        :align: center
        :alt: quartznet model
   
    .. note:: This checkpoint was trained on LibriSpeech :cite:`panayotov2015librispeech` and full "validated" part of En Mozilla Common Voice :cite:`ardila2019common`

`QuartzNet paper <https://arxiv.org/abs/1910.10261>`_.

=============================== ===================== ============
Network                         Dataset               Results
=============================== ===================== ============
QuartzNet5x1 (1M params)        Speech Commands V1    98.52% Test

QuartzNet5x1 (0.48M params)     Speech Commands V1    98.41% Test

QuartzNet3x1 (0.077M params)    Speech Commands V1    97.65% Test
=============================== ===================== ============

Squeeze-Excitation QuartzNet
----------------------------

SE-QuartzNet is a variant of the original QuartzNet which adds Squeeze and Excitation :cite:`speech-recognition-models-hu2018squeeze`
sub-modules. It has been shown in the paper that "Squeeze and Excitation" sub-modules can improve performance at the cost
of adding more parameters.

The Temporal Squeeze and Excitation sub-module can be described as below :cite:`speech-recognition-models-karim2019multivariate`:


    .. image:: temporal_se.png
        :align: center
        :alt: temporal squeeze excitation sub-module


=============================== ===================== ============
Network                         Dataset               Results
=============================== ===================== ============
SE-QuartzNet5x1 (1M params)     Speech Commands V1    98.27% Test

SE-QuartzNet5x1 (0.51M params)  Speech Commands V1    98.26% Test

SE-QuartzNet3x1 (0.08M params)  Speech Commands V1    97.88% Test
=============================== ===================== ============


References
----------

.. bibliography:: speech_recognition_all.bib
    :style: plain
    :labelprefix: SPEECH-RECOGNITION-MODELS
    :keyprefix: speech-recognition-models-