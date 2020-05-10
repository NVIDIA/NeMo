QuartzNet
---------

QuartzNet is a version of Jasper model with separable convolutions and larger filters. It can achieve performance
similar to Jasper but with an order of magnitude less parameters.
Similarly to Jasper, QuartzNet family of models are denoted as QuartzNet_[BxR] where B is the number of blocks, and R - the number of convolutional sub-blocks within a block. Each sub-block contains a 
1-D *separable* convolution, batch normalization, ReLU, and dropout:

We use Quarznet encoder of 3x2 size with narrow filters. These encoder is connected to decoder by using statspooling layer. 
We experimented with various layers like gram layer, x-vector pooling, super vector which is combination of gram and x-vector. 
Mean and variance based pooling is faster to train and very stable.

    .. image:: quartz_vertical.png
        :align: center
        :alt: quartznet model

`QuartzNet paper <https://arxiv.org/abs/1910.10261>`_.

on average for 417 hrs of data should finish 25 epochs in about 7-8 hours on single Quadro GV100. For larger datasets 
on 8 GPUS with  mixed precision it takes about 17hrs to train.

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
