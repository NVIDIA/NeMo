Text Normalization
==================

This tool converts text from written form into its verbalized form, including numbers and dates, `10:00` -> `ten o'clock`, `10kg` -> `ten kilograms`.
Text normalization is used as a preprocessing step before Text to Speech (TTS). It could also be used for preprocessing Automatic Speech Recognition (ASR) training transcripts.

This tool is based on Python Regex and offers prediction on text files and evaluation on `Google text normalization dataset <https://www.kaggle.com/richardwilliamsproat/text-normalization-for-english-russian-and-polish>`__ :cite:`tools-norm-sproat2016rnn`.
It reaches 81% in sentence accuracy on `output-00001-of-00100` of `Google text normalization dataset <https://www.kaggle.com/richardwilliamsproat/text-normalization-for-english-russian-and-polish>`__, 97.4% in token accuracy.

.. note::

    We recommend you try the tutorial `NeMo/tutorials/tools/Text_Normalization_Tutorial.ipynb <https://github.com/NVIDIA/NeMo/blob/main/tutorials/tools/Text_Normalization_Tutorial.ipynb>`__.
    

Prediction
----------------------------------

Example prediction run:

.. code::

    python run_prediction.py  --input=<INPUT_TEXT_FILE> --output=<OUTPUT_PATH>


Evaluation
----------------------------------

Example evaluation run:

.. code::

    python run_evaluation.py  --input=./en_with_types/output-00001-of-00100 [--cat CLASS_CATEGORY]


References
----------

.. bibliography:: tools_all.bib
    :style: plain
    :labelprefix: TOOLS-NORM
    :keyprefix: tools-norm-


