.. _text_normalization:

Text Normalization Models
==========================
Text normalization is the task of converting a written text into its spoken form. For example,
``$123`` should be verbalized as ``one hundred twenty three dollars``, while ``123 King Ave``
should be verbalized as ``one twenty three King Avenue``. Text normalization is typically used as
a pre-processing step for a range of speech application such as text-to-speech synthesis (TTS).

Data format
------------------

The data needs to be stored in TAB separated files (.tsv) with three columns, the first of which
is the "semiotic class", the second is the input token and the third is the output. An example can
be the dataset used in the `Google Text Normalization Challenge <https://www.kaggle.com/google-nlu/text-normalization>`_.
