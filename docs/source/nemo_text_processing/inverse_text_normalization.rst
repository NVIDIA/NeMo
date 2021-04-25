Inverse Text Normalization
==========================

Inverse text normalization (ITN), also called denormalization, is a part of the Automatic Speech Recognition (ASR) post-processing 
pipeline. ITN is the task of converting the raw spoken output of the ASR model into its written form to improve text readability.

For example: 
`"in nineteen seventy"` -> `"in 1975"` 
and `"it costs one hundred and twenty three dollars"` -> `"it costs $123"`.

This tool is based on WFST-grammars. We also provide a deployment route to C++ using Sparrowhawk - an open-source version of Google 
Kestrel :cite:`textprocessing-itn-ebden2015kestrel`. See :doc:`ITN Deployment <../tools/inverse_text_normalization_deployment>` for 
details.

.. note::

    For more information, see the tutorial `NeMo/tutorials/text_processing/Inverse_Text_Normalization.ipynb <https://github.com/NVIDIA/NeMo/blob/main/tutorials/text_processing/Inverse_Text_Normalization.ipynb>`__ in `Google's Colab <https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/text_processing/Inverse_Text_Normalization.ipynb>`_.

Classes
-------

The base class for every grammar is :class:``GraphFst<nemo_text_processing.inverse_text_normalization.graph_utils.GraphFst>``. This 
tool is designed as a two-stage application: 

1. ``classification`` of the input into semiotic tokens
2. ``verbalization`` into written form

For every stage and every semiotic token class, there is a corresponding grammar, for example, :class:``taggers.CardinalFst<nemo_text_processing.inverse_text_normalization.taggers.cardinal.CardinalFst>``
and :class:``verbalizers.CardinalFst<nemo_text_processing.inverse_text_normalization.verbalizers.cardinal.CardinalFst>``.
Together, they compose the final grammars :class:``taggers.ClassifyFinalFst<nemo_text_processing.inverse_text_normalization.classify.tokenize_and_classify_final.ClassifyFinalFst>`` and 
:class:``verbalizers.VerbalizeFinalFst<nemo_text_processing.inverse_text_normalization.classify.verbalize_final.VerbalizeFinalFst>`` that are compiled into WFST and used for inference.

.. autoclass:: nemo_text_processing.inverse_text_normalization.taggers.tokenize_and_classify.ClassifyFst
    :show-inheritance:
    :members:

.. autoclass:: nemo_text_processing.inverse_text_normalization.VerbalizeFst
    :show-inheritance:
    :members:

Prediction
----------

For example, to start prediction, run:

.. code::

    python run_prediction.py  --input=<INPUT_TEXT_FILE> --output=<OUTPUT_PATH>  [--verbose]


Data Cleaning for Evaluation
----------------------------

For example, to start evaluation, run:

.. code::

    python clean_eval_data.py  --input=<INPUT_TEXT_FILE>

Evaluation
----------

For example, to start evaluation, run on (cleaned) `Google's text normalization dataset <https://www.kaggle.com/richardwilliamsproat/text-normalization-for-english-russian-and-polish>`__ :cite:`textprocessing-itn-sproat2016rnn`:

.. code::

    python run_evaluation.py  --input=./en_with_types/output-00001-of-00100 [--cat CLASS_CATEGORY] [--filter]

References
----------

.. bibliography:: textprocessing_all.bib
    :style: plain
    :labelprefix: TEXTPROCESSING-ITN
    :keyprefix: textprocessing-itn-