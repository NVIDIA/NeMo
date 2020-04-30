Installation
============

Neural Modules and their corresponding collections have certain requirements that can be optionally installed to
improve performance of operations.

Torch Audio
-----------

The `torchaudio` library is used for certain audio pre-processing Neural Modules. Primarily,

 - AudioToMFCCPreprocessor
 - TimeStretchAugmentation

Official installation directions are provided at the `torchaudio github page <https://github.com/pytorch/audio>`_. It is recommended to follow
the conda installation procedure and install the latest version of the library available on conda.

Numba
-----

The `numba` library is used for optimized execution of certain data augmentation procedures that can be used during
data pre-processing. It can substantially reduce execution time during training, and is a recommended installation for
Neural Modules.

Official installation directions are provided at the `numba github page <https://github.com/numba/numba>`_. It is recommended to follow
the conda installation procedure and install the latest version of the library available on conda.

.. code-block:: bash

    conda install numba
