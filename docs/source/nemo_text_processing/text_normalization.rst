Text Normalization
==================

NeMo Text Normalization converts text from written form into its verbalized form. It is used as a preprocessing step before Text to 
Speech (TTS). It could also be used for preprocessing Automatic Speech Recognition (ASR) training transcripts.

For example: 
`"at 10:00"` -> `"at ten o'clock"` 
and `"it weighs 10kg"` -> `"it weights ten kilograms"`.

This tool is currently based on Python Regex.

.. note::

    For more details, see the tutorial `NeMo/tutorials/text_processing/Text_Normalization_Tutorial.ipynb <https://github.com/NVIDIA/NeMo/blob/main/tutorials/text_processing/Text_Normalization_Tutorial.ipynb>`__ in `Google's Colab <https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/text_processing/Text_Normalization_Tutorial.ipynb>`_.
    
Prediction
----------

For example, to start a prediction, run:

.. code::

    python run_prediction.py  --input=<INPUT_TEXT_FILE> --output=<OUTPUT_PATH>

Evaluation
----------

For example, to start evaluation on `Google's text normalization dataset <https://www.kaggle.com/richardwilliamsproat/text-normalization-for-english-russian-and-polish>`__ :cite:`textprocessing-norm-sproat2016rnn`, run:

.. code::

    python run_evaluation.py  --input=./en_with_types/output-00001-of-00100 [--cat CLASS_CATEGORY]

References
----------

.. bibliography:: textprocessing_all.bib
    :style: plain
    :labelprefix: TEXTPROCESSING-NORM
    :keyprefix: textprocessing-norm-