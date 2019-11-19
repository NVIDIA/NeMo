QuartzNet
---------

QuartzNet is a version of Jasper :cite:`li2019jasper` model with separable convolutions and larger filters. It can achieve performance
similar to Jasper but with an order of magnitude less parameters.
Similarly to Jasper, QuartzNet family of models are denoted as QuartzNet_[BxR] where B is the number of blocks, and R - the number of convolutional sub-blocks within a block. Each sub-block contains a 1-D *separable* convolution, batch normalization, ReLU, and dropout:

    .. image:: quartz_vertical.png
        :align: center
        :alt: quartznet model
   
    .. note:: This checkpoint was trained on LibriSpeech and full "validated" part of En Mozilla Common Voice

We are working on a QuartzNet paper and will release it soon.

Pretrained models can be found, `here <https://ngc.nvidia.com/catalog/models/nvidia:quartznet15x5>`_.
