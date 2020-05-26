QuartzNet
---------

QuartzNet is a version of Jasper that utilizes separable convolutions and larger filters. It can achieve performance
similar to Jasper but with an order of magnitude less parameters.
Similar to Jasper, QuartzNet family of models are denoted as QuartzNet_[BxR] where B is the number of blocks, and R -
the number of convolutional sub-blocks within a block. Each sub-block contains a 1-D *separable* convolution, batch
normalization, ReLU, and dropout:

We use a Quartznet 3x2 model with narrow filters. This encoder is connected to the decoder by using a statistics pooling layer.
We experimented with various statistics pooling layers including a gram layer, a x-vector pooling layer, or a super vector layer which is a combination of the gram and x-vector layers.
The xvector layer is based on mean and variance based statistics pooling, it is faster to train and very stable.

    .. image:: ../asr/quartz_vertical.png
        :align: center
        :alt: quartznet model

`QuartzNet paper <https://arxiv.org/abs/1910.10261>`_.

For a dataset with ~400 hours, this model should finish 25 epochs in under 8 hours on single Quadro GV100.

============== ================= ===================== ====================== ==========
Network            Trained             Evaluated           cosine similarity     PLDA
                    Dataset             trial-set              EER               EER
============== ================= ===================== ====================== ==========
QuartzNet3x2        hi-mia                hi-mia               8.72%             6.32%
QuartzNet3x2        voxceleb1             ffsvc-dev            14.22%            7.12%
                    hi-mia
                    aishell
                    voxceleb2
============== ================= ===================== ====================== ==========

