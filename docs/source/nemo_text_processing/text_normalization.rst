Text Normalization
==================

NeMo Text Normalization converts text from written form into its verbalized form. It is used as a preprocessing step before Text to Speech (TTS). It could also be used for preprocessing Automatic Speech Recognition (ASR) training transcripts.


For example, 
`"at 10:00"` -> `"at ten o'clock"` 
and `"it weighs 10kg"` -> `"it weights ten kilograms"`.

NeMo Text Normalization is based on WFST-grammars :cite `textprocessing-norm-zhang2021nemo`. We also provide a deployment route to C++ using Sparrowhawk -- an open-source version of Google Kestrel :cite:`textprocessing-norm-ebden2015kestrel`.
See :doc:`Text Procesing Deployment <../tools/text_processing_deployment>` for details.


.. note::

    For more details, see the tutorial `NeMo/tutorials/text_processing/Text_Normalization_Tutorial.ipynb <https://github.com/NVIDIA/NeMo/blob/main/tutorials/text_processing/Text_Normalization_Tutorial.ipynb>`__ in `Google's Colab <https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/text_processing/Text_Normalization_Tutorial.ipynb>`_.






Classes
----------------------------------


The base class for every grammar is :class:`GraphFst<nemo_text_processing.text_normalization.graph_utils.GraphFst>`.
This tool is designed as a two-stage application: 1. `classification` of the input into semiotic tokens and 2. `verbalization` into written form.
For every stage and every semiotic token class there is a corresponding grammar, e.g. :class:`taggers.CardinalFst<nemo_text_processing.text_normalization.taggers.cardinal.CardinalFst>`
and :class:`verbalizers.CardinalFst<nemo_text_processing.text_normalization.verbalizers.cardinal.CardinalFst>`.
Together, they compose the final grammars :class:`taggers.ClassifyFst<nemo_text_processing.text_normalization.classify.tokenize_and_classify.ClassifyFst>` and 
:class:`verbalizers.VerbalizeFinalFst<nemo_text_processing.text_normalization.classify.verbalize_final.VerbalizeFinalFst>` that are compiled into WFST and used for inference.






.. autoclass:: nemo_text_processing.text_normalization.taggers.tokenize_and_classify.ClassifyFst
    :show-inheritance:
    :members:

.. autoclass:: nemo_text_processing.text_normalization.VerbalizeFst
    :show-inheritance:
    :members:
 

Prediction
----------------------------------

Example prediction run:

.. code::

    python run_prediction.py  --input=<INPUT_TEXT_FILE> --output=<OUTPUT_PATH>


Evaluation
----------------------------------

Example evaluation run on `Google's text normalization dataset <https://www.kaggle.com/richardwilliamsproat/text-normalization-for-english-russian-and-polish>`__ :cite:`textprocessing-norm-sproat2016rnn`:

.. code::

    python run_evaluation.py  --input=./en_with_types/output-00001-of-00100 [--cat CLASS_CATEGORY]



References
----------

.. bibliography:: textprocessing_all.bib
    :style: plain
    :labelprefix: TEXTPROCESSING-NORM
    :keyprefix: textprocessing-norm-