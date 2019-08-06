Jasper
------

Jasper ("Just Another SPeech Recognizer") :cite:`li2019jasper`  is a deep time delay neural network (TDNN) comprising of blocks of 1D-convolutional layers.
Jasper family of models are denoted as Jasper_[BxR] where B is the number of blocks, and R - the number of convolutional sub-blocks within a block. Each sub-block contains a 1-D convolution, batch normalization, ReLU, and dropout:

    .. image:: jasper.png
        :align: center
        :alt: japer model


References
-------------

.. bibliography:: Jasperbib.bib
    :style: plain

