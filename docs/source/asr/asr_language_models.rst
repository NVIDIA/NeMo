ASR Language Modeling
=====================

Language models have shown to help the accuracy of ASR models. NeMo support the following two approaches to incorporate language models into the ASR models:
    + N-gram Language modelling
    + Neural Rescoring


.. _ngram_modelling:

N-gram Language Modelling
-------------------------

In this approach, an N-gram LM is trained on some text data, then it is used in fusion with beam search decoding to find the best candidates.
The beam search decoders in NeMo supports language models trained by KenLM library <TODO:give ref>.
The script to train a KenLM model an be found at <TODO: give ref>.
The trained N-gram model can be used with beam search decoders on top of the ASR models to produce more accurate candidates.
An script to evaluate an ASR model with N-gram models can be found at <TODO:give ref>

.. _neural_rescoring:

Neural Rescoring
----------------

In this approach a neural network is used which can gives scores to a candidate.
The top K candidates produced by the beam search decoding (beam width of K) are given to the neural language model to rank them.
Ranking can be done by a language model which gives a score to each candidate.
This score is usually combined with the scores from the beam search decoding to produce the final scores and rankings.
An example script to train such a model with Transformer language models can be found at <TODO: give ref>.

References
----------

.. bibliography:: asr_all.bib
    :style: plain
    :labelprefix: ASR-LM_MODELS
    :keyprefix: asr-lm-models-
