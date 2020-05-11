Tutorial
========

Make sure you have installed ``nemo`` and the ``nemo_asr`` collection.
See the :ref:`installation` section.

.. note::
    
    You need to have ``nemo`` and the ``nemo_asr`` collection for this tutorial.
    It is also necessary to install `torchaudio` in order to use MFCC preprocessing.


Introduction
------------

Speaker Recognition (SR) is a broad research area which solves two major tasks: speaker identification (who is speaking?) and 
speaker verification (is the speaker who she claims to be?). In this work, we focus on the far-field, 
text-independent speaker recognition when the identity of the speaker is based on how speech is spoken, 
not necessarily in what is being said. Typically such SR systems operate on unconstrained speech utterances, 
which are converted into vector of fixed length, called speaker embedding. Speaker embeddings are also  used in 
automatic speech recognition (ASR) and speech synthesis. 

As the goal of most speaker related systems is to get good speaker level embeddings that could help distinguish from other speakers, we shall first train these embeddings in end-to-end
manner optimizing the QuatzNet based :cite:`speaker-tut-kriman2019quartznet` encoder model on cross-entropy loss. 
We modeify the decoder to get these fixed size embeddings irrespective of the length of input audio. We employ mean and variance 
based statistics pooling method to grab these embeddings.

In this tutorial we shall first train these embeddings on speaker related datasets and then get speaker embeddings from a 
pretrained network for a new dataset, then followed by scoring them using cosine similarity method or optionally with PLDA backend. 


Jupyter Notebooks containing all the steps to download the dataset, train a model and evaluate its results 
is available at : `Speaker Recognition an4 example <https://github.com/NVIDIA/NeMo/blob/master/examples/speaker_recognition/notebooks/Speaker_Recognition_an4.ipynb>`_ 
also for advanced hi-mia dataset at: `Speaker Recognition hi-mia example <https://github.com/NVIDIA/NeMo/blob/master/examples/speaker_recognition/notebooks/Speaker_Recognition_hi-mia.ipynb>`_ 


References
----------

.. bibliography:: speaker.bib
    :style: plain
    :labelprefix: SPEAKER-TUT
    :keyprefix: speaker-tut-
