ASR Language Modeling
=====================

Language models have shown to help the accuracy of ASR models. NeMo support the following two approaches to incorporate language models into the ASR models:
    + N-gram Language modelling
    + Neural Rescoring

It is possible to use both apprpaches on the same ASR model.


.. _ngram_modelling:

N-gram Language Modelling
-------------------------

In this approach, an N-gram LM is trained on text data, then it is used in fusion with beam search decoding to find the best candidates.
The beam search decoders in NeMo supports language models trained by KenLM library 'https://github.com/kpu/kenlm'.
The beam search decoders and KenLM library are not installed by default in NeMo.
You may need to install them first by the script at 'scripts/asr_language_modelling/ngram_lm/install_beamsearch_decoders.py'.

The script to train a KenLM model can be found at 'scripts/asr_language_modelling/ngram_lm/train_kenlm.py'.
The trained N-gram model can be used with beam search decoders on top of the ASR models to produce more accurate candidates.
The beam search decoder would incorporate the scores produced by the N-gram LM into its score calculations as the following:

.. code::

    final_score = acoustic_score + beam_alpha*lm_score - beam_beta*seq_length

where acoustic_score is the score predicted by the acoustic encoder and lm_score is the one estimates by the LM.
Parameter beam_lpha specifies amount of importance to place on the N-gram language model, and beam_beta is a penalty term given to longer word sequences.

You may train the N-gram LM as the following:

.. code::

    python train_kenlm.py --nemo_model_file <path to the .nemo file of the model> \
                              --train_file <path to the training text or json manifest file \
                              --kenlm_bin_path <path to the bin folder of KenLM library> \
                              --kenlm_model_file <path to store the binary KenLM model> \
                              --ngram_length <order of N-gram model>


An script to evaluate an ASR model with N-gram models can be found at 'scripts/asr_language_modelling/ngram_lm/eval_beamsearch_ngram.py'.
You may evaluate an ASR model as the following:

.. code::

    python eval_beamsearch_ngram.py --nemo_model_file <path to the .nemo file of the model> \
                                         --input_manifest <path to the evaluation Json manifest file \
                                         --kenlm_model_file <path to the binary KenLM model> \
                                         --beam_width <list of the beam widths> \
                                         --beam_alpha <list of the beam alphas> \
                                         --beam_width <list of the beam betas> \
                                         --preds_output_folder <optional folder to store the predictions>

It would report the performances in terms of Word Error Rate (WER) and Character Error Rate (CER).
The script can perform grid search on the three hyperparameters of beam_width, beam_alpha, and beam_beta when a list of values are given.

.. _neural_rescoring:
Neural Rescoring
----------------

In this approach a neural network is used which can gives scores to a candidate. A candidate is the text transcript predicted by the decoder of the ASR model.
The top K candidates produced by the beam search decoding (beam width of K) are given to a neural language model to rank them.
Ranking can be done by a language model which gives a score to each candidate.
This score is usually combined with the scores from the beam search decoding to produce the final scores and rankings.
An example script to train such a language model with Transformer can be found at 'examples/nlp/language_modelling/transformer_lm.py'.

References
----------

.. bibliography:: asr_all.bib
    :style: plain
    :labelprefix: ASR-LM_MODELS
    :keyprefix: asr-lm-models-
