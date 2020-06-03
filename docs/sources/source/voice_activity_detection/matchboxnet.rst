MatchboxNet
-----------

MatchboxNet :cite:`vad-models-majumdar2020` is a version of QuartzNet :cite:`vad-models-kriman2019quartznet`. QuartzNet can achieve performance
similar to Jasper but with an order of magnitude less parameters.
Similarly to Jasper, QuartzNet family of models are denoted as QuartzNet_[BxR] where B is the number of blocks, and R - the number of convolutional sub-blocks within a block. Each sub-block contains a 1-D *separable* convolution, batch normalization, ReLU, and dropout:

These models are trained on Google Speech Commands dataset (V2 - all 35 classes) and freesound background data.

    .. image:: quartz_vertical.png
        :align: center
        :alt: quartznet model

Th MatchboxNet models were trained for 200 epochs using mixed precision on 2 GPUs with a batch size of 128 over 200 epochs.
On 2 Quadro GV100 GPUs, training time is approximately 1 hour.

============================ ============================== ============ ============
Network                      Dataset                        Accuracy     F1 score
============================ ============================== ============ ============
MatchboxNet3x1 (73k params)  Speech Commands V2 + Freesound  99.71% Test 99.75% Test
============================ ============================== ============ ============


References
^^^^^^^^^^

.. bibliography:: vad_all.bib
    :style: plain
    :labelprefix: VAD-MODELS
    :keyprefix: vad-models-