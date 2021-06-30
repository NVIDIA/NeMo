.. _text_normalization:

Text Normalization Models
==========================
Text normalization is the task of converting a written text into its spoken form. For example,
``$123`` should be verbalized as ``one hundred twenty three dollars``, while ``123 King Ave``
should be verbalized as ``one twenty three King Avenue``. At the same time, the inverse problem
is about converting a spoken sequence (e.g., an ASR output) into its written form.

NeMo Data Format
-----------
Both the DuplexTaggerModel model and the DuplexDecoderModel model use the same simple text format
as the dataset. The data needs to be stored in TAB separated files (``.tsv``) with three columns.
The first of which is the "semiotic class" (e.g.,  numbers, times, dates) , the second is the token
in written form, and the third is the spoken form. An example sentence in the dataset is shown below.
In the example, ``sil`` denotes that a token is a punctuation while ``self`` denotes that the spoken form is the
same as the written form. It is expected that a complete dataset contains three files: ``train.tsv``, ``dev.tsv``,
and ``test.tsv``.

.. code::

    PLAIN	The	<self>
    PLAIN	company 's	<self>
    PLAIN	revenues	<self>
    PLAIN	grew	<self>
    PLAIN	four	<self>
    PLAIN	fold	<self>
    PLAIN	between	<self>
    DATE	2005	two thousand five
    PLAIN	and	<self>
    DATE	2008	two thousand eight
    PUNCT	.	sil
    <eos>	<eos>


An example script for generating a dataset in this format from the `Google text normalization dataset <https://www.kaggle.com/google-nlu/text-normalization>`_
can be found at  `NeMo/examples/nlp/duplex_text_normalization/google_data_preprocessing.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/duplex_text_normalization/google_data_preprocessing.py>`__.
Note that the script also does some preprocessing on the spoken forms of the URLs. For example,
given the URL "Zimbio.com", the original expected spoken form in the Google dataset is
"z_letter i_letter m_letter b_letter i_letter o_letter dot c_letter o_letter m_letter".
However, our script will return a more concise output which is "zim bio dot com".
