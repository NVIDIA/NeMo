.. _wfst_tn:

Text (Inverse) Normalization
============================

The `nemo_text_processing` Python package :cite:`textprocessing-norm-zhang2021nemo` is based on WFST grammars :cite:`textprocessing-norm-mohri2005weighted` and supports:

1. Text Normalization (TN) converts text from written form into its verbalized form. It is used as a preprocessing step before Text to Speech (TTS). 
2. Inverse text normalization (ITN) is a part of the Automatic Speech Recognition (ASR) post-processing pipeline and can be used to convert normalized ASR model outputs into written form to improve text readability.
3. Audio-based TN uses an extended set of non-deterministic normalization grammars to normalize ASR training data for better ASR accuracy.


Installation
------------

`nemo_text_processing` is automatically installed with `nemo_toolkit`.
See :doc:`NeMo Introduction <../../starthere/intro>` for installation details of `nemo_toolkit`.

Quick Start Guide
-----------------

.. note::

    Walk through `NeMo/tutorials/text_processing/Text_Normalization.ipynb <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/text_processing/Text_Normalization.ipynb>`__ in `Google's Colab <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/text_processing/Text_Normalization.ipynb>`_.


Language Support
-------------------
Languages are listed in decreasing order of grammar complexity.

Deterministic TN: 

* English
* Spanish
* German

Deterministic ITN: 

* English
* Spanish
* German
* French
* Russian
* Vietnamese

Non-deterministic TN:

* English
* Spanish
* German
* Russian


Grammar customization
---------------------

.. note::

    In-depth walk through `NeMo/tutorials/text_processing/WFST_tutorial.ipynb <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/text_processing/WFST_Tutorial.ipynb>`__ in `Google's Colab <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/text_processing/WFST_Tutorial.ipynb>`_.


Deploy to C++
-----------------
See :doc:`Text Procesing Deployment <wfst_text_processing_deployment>` for details.



References
----------

.. bibliography:: ../tn_itn_all.bib
    :style: plain
    :labelprefix: TEXTPROCESSING-NORM
    :keyprefix: textprocessing-norm-