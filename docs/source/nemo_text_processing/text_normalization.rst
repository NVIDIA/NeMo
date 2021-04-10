Text Normalization
==================

This tool converts text from written form into its verbalized form, including numbers and dates, `10:00` -> `ten o'clock`, `10kg` -> `ten kilograms`.
Text normalization is used as a preprocessing step before Text to Speech (TTS). It could also be used for preprocessing Automatic Speech Recognition (ASR) training transcripts.

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


