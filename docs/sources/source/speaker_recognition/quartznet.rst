QuartzNet
---------

QuartzNet is a version of Jasper model with separable convolutions and larger filters. It can achieve performance
similar to Jasper but with an order of magnitude less parameters.
Similarly to Jasper, QuartzNet family of models are denoted as QuartzNet_[BxR] where B is the number of blocks, and R - the number of convolutional sub-blocks within a block. Each sub-block contains a 
1-D *separable* convolution, batch normalization, ReLU, and dropout:

We use a Quarznet encoder of 3x2 size with narrow filters. This encoder is connected to decoder by using statspooling layer. 
We experimented with various layers like a gram layer, a x-vector pooling layer, or super vector layer which is combination of the gram and x-vector layers.
xvector stratergy is based on Mean and variance based statistic pooling, it is faster to train and very stable.

    .. image:: ../asr/quartz_vertical.png
        :align: center
        :alt: quartznet model

`QuartzNet paper <https://arxiv.org/abs/1910.10261>`_.

on average for 417 hrs of data should finish 25 epochs in under 8 hours on single Quadro GV100. 

============== ================= ===================== ====================== ==========
Network            Trained             Evaluated           cosine similarity     PLDA
                    Dataset             trial-set              EER               EER
============== ================= ===================== ====================== ==========
QuartzNet3x2        hi-mia                hi-mia               8.72%             6.32%
QuartzNet3x2        voxceleb1             ffsvc-dev            14.22%	         7.12%
                    hi-mia
                    aishell
                    voxceleb2
============== ================= ===================== ====================== ==========


References
----------

    .. bibliography:: speaker.bib
        :style: plain
        :labelprefix: SPEAKER-TUT
        :keyprefix: speaker-tut-
