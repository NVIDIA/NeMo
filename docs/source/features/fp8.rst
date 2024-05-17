FP8 usage
=========

Overview
^^^^^^^^

NVIDIA's H100 GPU introduced support for a new datatype, FP8 (8-bit floating point), enabling higher throughput of matrix multiplies and convolutions. NeMo uses NVIDIA's `TransformerEngine <https://github.com/NVIDIA/TransformerEngine>`_ (TE) in order to leverage speedups from FP8.
