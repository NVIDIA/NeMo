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

These QuartzNet models were trained for 200 epochs using mixed precision on 2 GPUs with a batch size of 128 over 200 epochs.
On 2 Quadro GV100 GPUs, training time is approximately 1 hour.

=============================== ===================== ============
Network                         Dataset               Results
=============================== ===================== ============
QuartzNet3x1 (0.077M params)    Speech Commands V1    97.46% Test

QuartzNet3x2 (0.093M params)    Speech Commands V2    97.35% Test
=============================== ===================== ============


References
----------

.. bibliography:: speech_recognition_all.bib
    :style: plain
    :labelprefix: SPEECH-RECOGNITION-MODELS
    :keyprefix: speech-recognition-models-