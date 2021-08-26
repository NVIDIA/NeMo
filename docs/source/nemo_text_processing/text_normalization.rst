Text Normalization
==================

NeMo Text Normalization converts text from written form into its verbalized form. It is used as a preprocessing step before Text to Speech (TTS). It could also be used for preprocessing Automatic Speech Recognition (ASR) training transcripts.


For example, 
`"at 10:00"` -> `"at ten o'clock"` 
and `"it weighs 10kg."` -> `"it weights ten kilograms ."`.


NeMo Text Normalization :cite:`textprocessing-norm-zhang2021nemo` is based on WFST-grammars :cite:`textprocessing-norm-Mohri2009`. We also provide a deployment route to C++ using `Sparrowhawk <https://github.com/google/sparrowhawk>`_ :cite:`textprocessing-norm-sparrowhawk` -- an open-source version of Google Kestrel :cite:`textprocessing-norm-ebden2015kestrel`.
See :doc:`Text Procesing Deployment <../tools/text_processing_deployment>` for details.


.. note::

    For more details, see the tutorial `NeMo/tutorials/text_processing/Text_Normalization.ipynb <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/text_processing/Text_Normalization.ipynb>`__ in `Google's Colab <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/text_processing/Text_Normalization.ipynb>`_.






Classes
----------------------------------


The base class for every grammar is :class:`GraphFst<nemo_text_processing.text_normalization.en.GraphFst>`.
This tool is designed as a two-stage application: 1. `classification` of the input into semiotic tokens and 2. `verbalization` into written form.
For every stage and every semiotic token class there is a corresponding grammar, e.g. :class:`taggers.CardinalFst<nemo_text_processing.text_normalization.en.taggers.cardinal.CardinalFst>`
and :class:`verbalizers.CardinalFst<nemo_text_processing.text_normalization.en.verbalizers.cardinal.CardinalFst>`.
Together, they compose the final grammars :class:`ClassifyFst<nemo_text_processing.text_normalization.en.ClassifyFst>` and 
:class:`VerbalizeFinalFst<nemo_text_processing.text_normalization.en.VerbalizeFinalFst>` that are compiled into WFST and used for inference.




.. autoclass:: nemo_text_processing.text_normalization.en.ClassifyFst
    :show-inheritance:
    :members:

.. autoclass:: nemo_text_processing.text_normalization.en.VerbalizeFinalFst
    :show-inheritance:
    :members:
 

Prediction
----------------------------------

Example prediction run:

.. code::

    python run_prediction.py  <--input INPUT_TEXT_FILE> <--output OUTPUT_PATH> <--language LANGUAGE> [--input_case INPUT_CASE]

``INPUT_CASE`` specifies whether to treat the input as lower-cased or case sensitive. By default treat the input as cased since this is more informative, especially for abbreviations. Punctuation are outputted with separating spaces after semiotic tokens, e.g. `"I see, it is 10:00..."` -> `"I see, it is ten o'clock  .  .  ."`.
Inner-sentence white-space characters in the input are not maintained. 


Evaluation
----------------------------------

Example evaluation run on `Google's text normalization dataset <https://www.kaggle.com/richardwilliamsproat/text-normalization-for-english-russian-and-polish>`__ :cite:`textprocessing-norm-sproat2016rnn`:

.. code::

    python run_evaluation.py  --input=./en_with_types/output-00001-of-00100 --language=en [--cat CLASS_CATEGORY] [--input_case INPUT_CASE]
 


References
----------

.. bibliography:: textprocessing_all.bib
    :style: plain
    :labelprefix: TEXTPROCESSING-NORM
    :keyprefix: textprocessing-norm-