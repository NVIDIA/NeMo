.. _text_normalization:

Text Normalization Models
==========================
Text normalization is the task of converting a written text into its spoken form. For example,
``$123`` should be verbalized as ``one hundred twenty three dollars``, while ``123 King Ave``
should be verbalized as ``one twenty three King Avenue``. At the same time, the inverse problem
is about converting a spoken sequence (e.g., an ASR output) into its written form.

NeMo has an implementation that allows you to build a neural-based system that is able to do
both text normalization (TN) and also inverse text normalization (ITN). At a high level, the
system consists of two individual components:

- `DuplexTaggerModel <https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/duplex_text_normalization/duplex_tagger.py/>`__ - a Transformer-based tagger for identifying "semiotic" spans in the input (e.g., spans that are about times, dates, or monetary amounts).
- `DuplexDecoderModel <https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/duplex_text_normalization/duplex_decoder.py/>`__ - a Transformer-based seq2seq model for decoding the semiotic spans into their appropriate forms (e.g., spoken forms for TN and written forms for ITN).

The typical workflow is to first train a DuplexTaggerModel and also a DuplexDecoderModel. An example training script
is provided: `duplex_text_normalization_train.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/duplex_text_normalization/duplex_text_normalization_train.py>`__.
After that, the two trained models can be used to initialize a `DuplexTextNormalizationModel <https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/duplex_text_normalization/duplex_tn.py/>`__ that can be used for end-to-end inference.
An example script for evaluation and inference is provided: `duplex_text_normalization_test.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/duplex_text_normalization/duplex_text_normalization_test.py>`__. The term
*duplex* refers to the fact that our system can be trained to do both TN and ITN. However, you can also specifically train the system for only one of the tasks.

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

More information about the Google text normalization dataset can be found in the paper `RNN Approaches to Text Normalization: A Challenge <https://arxiv.org/ftp/arxiv/papers/1611/1611.00068.pdf>`__ :cite:`nlp-textnorm-Sproat2016RNNAT`.


Model Training
--------------

An example training script is provided: `duplex_text_normalization_train.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/duplex_text_normalization/duplex_text_normalization_train.py>`__.
The config file used for the example is at `duplex_tn_config.yaml <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/duplex_text_normalization/conf/duplex_tn_config.yaml>`__.
You can change any of these parameters directly from the config file or update them with the command-line arguments.

The config file contains three main sections. The first section contains the configs for the tagger, the second section is about the decoder,
and the last section is about the dataset. Most arguments in the example config file are quite self-explanatory (e.g.,
*decoder_model.optim.lr* refers to the learning rate for training the decoder). We have set most of the hyper-parameters to
be the values that we found to be effective. Some arguments that you may want to modify are:

- *data.base_dir*: The path to the dataset directory. It is expected that the directory contains three files: train.tsv, dev.tsv, and test.tsv.

- *tagger_model.nemo_path*: This is the path where the final trained tagger model will be saved to.

- *decoder_model.nemo_path*: This is the path where the final trained decoder model will be saved to.

Example of a training command:

.. code::

    python examples/nlp/duplex_text_normalization/duplex_text_normalization_train.py \
        data.base_dir=<PATH_TO_DATASET_DIR> \
        mode={tn,itn,joint}

There are 3 different modes. "tn" mode is for training a system for TN only.
"itn" mode is for training a system for ITN. "joint" is for training a system
that can do both TN and ITN at the same time. Note that the above command will
first train a tagger and then train a decoder sequentially.

You can also train only a tagger (without training a decoder) by running the
following command:

.. code::

    python examples/nlp/duplex_text_normalization/duplex_text_normalization_train.py \
        data.base_dir=PATH_TO_DATASET_DIR \
        mode={tn,itn,joint} \
        decoder_model.do_training=false

Or you can also train only a decoder (without training a tagger):

.. code::

    python examples/nlp/duplex_text_normalization/duplex_text_normalization_train.py \
        data.base_dir=PATH_TO_DATASET_DIR \
        mode={tn,itn,joint} \
        tagger_model.do_training=false


Model Architecture
--------------

The tagger model first uses a Transformer encoder (e.g., DistilRoBERTa) to build a
contextualized representation for each input token. It then uses a classification head
to predict the tag for each token (e.g., if a token should stay the same, its tag should
be ``SAME``). The decoder model then takes the semiotic spans identified by the tagger and
transform them into the appropriate forms (e.g., spoken forms for TN and written forms for ITN).
The decoder model is essentially a Transformer-based encoder-decoder seq2seq model (e.g., the example
training script uses the T5-base model by default). Overall, our design is partly inspired by the
RNN-based sliding window model proposed in the paper
`Neural Models of Text Normalization for Speech Applications <https://research.fb.com/wp-content/uploads/2019/03/Neural-Models-of-Text-Normalization-for-Speech-Applications.pdf>`__ :cite:`nlp-textnorm-Zhang2019NeuralMO`.

We introduce a simple but effective technique to allow our model to be duplex. Depending on the
task the model is handling, we append the appropriate prefix to the input. For example, suppose
we want to transform the text ``I live in 123 King Ave`` to its spoken form (i.e., TN problem),
then we will simply append the prefix ``tn`` to it and so the final input to our models will actually
be ``tn I live in tn 123 King Ave``. Similarly, for the ITN problem, we just append the prefix ``itn``
to the input.

To improve the effectiveness and robustness of our models, we also apply some simple data
augmentation techniques during training.

Data Augmentation for Training DuplexTaggerModel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the Google English TN training data, about 93% of the tokens are not in any semiotic span. In other words, the ground-truth tags of most tokens are of trivial types (i.e., ``SAME`` and ``PUNCT``). To alleviate this class imbalance problem,
for each original instance with several semiotic spans, we create a new instance by simply concatenating all the semiotic spans together. For example, considering the following ITN instance:

Original instance: ``[The|SAME] [revenues|SAME] [grew|SAME] [a|SAME] [lot|SAME] [between|SAME] [two|B-TRANSFORM] [thousand|I-TRANSFORM] [two|I-TRANSFORM] [and|SAME] [two|B-TRANSFORM] [thousand|I-TRANSFORM] [five|I-TRANSFORM] [.|PUNCT]``

Augmented instance: ``[two|B-TRANSFORM] [thousand|I-TRANSFORM] [two|I-TRANSFORM] [two|B-TRANSFORM] [thousand|I-TRANSFORM] [five|I-TRANSFORM]``

The argument ``data.train_ds.tagger_data_augmentation`` in the config file controls whether this data augmentation will be enabled or not.


Data Augmentation for Training DuplexDecoderModel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Since the tagger may not be perfect, the inputs to the decoder may not all be semiotic spans. Therefore, to make the decoder become more robust against the tagger's potential errors,
we train the decoder with not only semiotic spans but also with some other more "noisy" spans. This way even if the tagger makes some errors, there will still be some chance that the
final output is still correct.

The argument ``data.train_ds.decoder_data_augmentation`` in the config file controls whether this data augmentation will be enabled or not.

References
----------

.. bibliography:: nlp_all.bib
    :style: plain
    :labelprefix: NLP-TEXTNORM
    :keyprefix: nlp-textnorm-
