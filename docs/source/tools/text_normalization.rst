Text Normalization
==================

Text normalization system for english, e.g. `123 kg` -> `one hundred twenty three kilograms`

Offers prediction and evaluation on text normalization data, e.g. `Google text normalization dataset <https://www.kaggle.com/richardwilliamsproat/text-normalization-for-english-russian-and-polish>`__ :cite:`tools-norm-ebden2015kestrel`, :cite:`tools-norm-sproat2016rnn`, :cite:`tools-norm-taylor2009text`.

Reaches 81% in sentence accuracy on output-00001-of-00100 of `Google text normalization dataset <https://www.kaggle.com/richardwilliamsproat/text-normalization-for-english-russian-and-polish>`__, 97.4% in token accuracy.

More details could be found in `this tutorial <https://github.com/NVIDIA/NeMo/blob/main/tutorials/tools/Text_Normalization_Tutorial.ipynb>`__.

Example evaluation run:

.. code::

    python run_evaluation.py  --input=./en_with_types/output-00001-of-00100 --normalizer nemo


Example prediction run:

.. code::

    python run_prediction.py  --input=text_data.txt --output=. --normalizer nemo


References
----------

.. bibliography:: tools_all.bib
    :style: plain
    :labelprefix: TOOLS-NORM
    :keyprefix: tools-norm-


