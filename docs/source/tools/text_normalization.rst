Text Normalization
==================

Text normalization converts text into its verbalized form. That is, tokens belonging to special semiotic classes to denote things like 
**numbers**, **times**, **dates**, **monetary amounts**, etc., that are often written in a way that differs from the way they are verbalized. 
For example, `10:00` -> `ten o'clock`, `10:00 a.m.` -> `ten a m`, `10kg` -> `ten kilograms`.
Text normalization is used as a preprocessing step before Text to Speech (TTS). It could also be used for preprocessing Automatic Speech Recognition (ASR) training transcripts.

This tool offers prediction on text files and evaluation on `Google text normalization dataset <https://www.kaggle.com/richardwilliamsproat/text-normalization-for-english-russian-and-polish>`__ :cite:`tools-norm-ebden2015kestrel`, :cite:`tools-norm-sproat2016rnn`, :cite:`tools-norm-taylor2009text`.
It reaches 81% in sentence accuracy on `output-00001-of-00100` of `Google text normalization dataset <https://www.kaggle.com/richardwilliamsproat/text-normalization-for-english-russian-and-polish>`__, 97.4% in token accuracy.

.. note::

    We recommend you try the tutorial `NeMo/tutorials/tools/Text_Normalization_Tutorial.ipynb <https://github.com/NVIDIA/NeMo/blob/main/tutorials/tools/Text_Normalization_Tutorial.ipynb>`__.
    

We use the same semiotic classes as in the Google Text normalization dataset: 
**PLAIN**, **PUNCT**, **DATE**, **CARDINAL**, **LETTERS**, **VERBATIM**, **MEASURE**, **DECIMAL**, **ORDINAL**, **DIGIT**, **MONEY**, **TELEPHONE**, **ELECTRONIC**, **FRACTION**, **TIME**, **ADDRESS**. 
We additionally added the class **WHITELIST** for all whitelisted tokens whose verbalizations are directly looked up from a user-defined list.

NeMo rule-based system is divided into a tagger and a verbalizer: 
the tagger is responsible for detecting and classifying semiotic classes in the underlying text, 
the verbalizer takes the output of the tagger and carries out the normalization. 
In the example `The alarm goes off at 10:30 a.m.`, the tagger for time detects `10:30 a.m.` as a valid time data with `hour=10`, `minutes=30`, `suffix=a.m.`, 
the verbalizer then turns this into ten thirty a m. 

The system is designed to be easily debuggable and extendable by more rules. 
We provide a set of rules that covers the majority of semiotic classes as found in the Google Text normalization dataset for the English language. As with every language there is a long tail of special cases.

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


