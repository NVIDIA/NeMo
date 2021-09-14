.. _punctuation_and_capitalization:

Punctuation and Capitalization Model
====================================

Automatic Speech Recognition (ASR) systems typically generate text with no punctuation and capitalization of the words. 
There are two issues with non-punctuated ASR output:

- it could be difficult to read and understand
- models for some downstream tasks, such as named entity recognition, machine translation, or text-to-speech, are usually trained 
on punctuated datasets and using raw ASR output as the input to these models could deteriorate their performance

Quick Start Guide
-----------------

.. code-block:: python

    from nemo.collections.nlp.models import PunctuationCapitalizationModel

    # to get the list of pre-trained models
    PunctuationCapitalizationModel.list_available_models()

    # Download and load the pre-trained BERT-based model
    model = PunctuationCapitalizationModel.from_pretrained("punctuation_en_bert")

    # try the model on a few examples
    model.add_punctuation_capitalization(['how are you', 'great how about you'])

Model Description
-----------------

For each word in the input text, the Punctuation and Capitalization model:

- predicts a punctuation mark that should follow the word (if any). By default, the model supports commas, periods, and question marks.
- predicts if the word should be capitalized or not

In the Punctuation and Capitalization model, we are jointly training two token-level classifiers on top of a pre-trained 
language model, such as `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`__ :cite:`nlp-punct-devlin2018bert`.

.. note::

    We recommend you try this model in a Jupyter notebook (run on `Google's Colab <https://colab.research.google.com/notebooks/intro.ipynb>`_.): `NeMo/tutorials/nlp/Punctuation_and_Capitalization.ipynb <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/nlp/Punctuation_and_Capitalization.ipynb>`__.

    Connect to an instance with a GPU (**Runtime** -> **Change runtime type** -> select **GPU** for the hardware accelerator).

    An example script on how to train the model can be found at: `NeMo/examples/nlp/token_classification/punctuation_capitalization_train.py <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/punctuation_capitalization_train.py>`__.

    An example script on how to run evaluation and inference can be found at: `NeMo/examples/nlp/token_classification/punctuation_capitalization_evaluate.py <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/punctuation_capitalization_evaluate.py>`__.

    The default configuration file for the model can be found at: `NeMo/examples/nlp/token_classification/conf/punctuation_capitalization_config.yaml <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/conf/punctuation_capitalization_config.yaml>`__.

.. _raw_data_format_punct:

Raw Data Format
---------------

The Punctuation and Capitalization model can work with any text dataset, although it is recommended to balance the data, especially 
for the punctuation task. Before pre-processing the data to the format expected by the model, the data should be split into ``train.txt`` 
and ``dev.txt`` (and optionally ``test.txt``). Each line in the ``train.txt/dev.txt/test.txt`` should represent one or more full 
and/or truncated sentences.

Example of the ``train.txt``/``dev.txt`` file:

.. code::

    When is the next flight to New York?
    The next flight is ...
    ....


The ``source_data_dir`` structure should look similar to the following:

.. code::

   .
   |--sourced_data_dir
     |-- dev.txt
     |-- train.txt

NeMo Data Format
----------------

The Punctuation and Capitalization model expects the data in the following format:

The training and evaluation data is divided into 2 files: 
- ``text.txt``
- ``labels.txt``

Each line of the ``text.txt`` file contains text sequences, where words are separated with spaces.

[WORD] [SPACE] [WORD] [SPACE] [WORD], for example:

    ::

        when is the next flight to new york
        the next flight is ...
        ...

The ``labels.txt`` file contains corresponding labels for each word in ``text.txt``, the labels are separated with spaces. 
Each label in ``labels.txt`` file consists of 2 symbols:

- the first symbol of the label indicates what punctuation mark should follow the word (where ``O`` means no punctuation needed)
- the second symbol determines if a word needs to be capitalized or not (where ``U`` indicates that the word should be upper cased, 
and ``O`` - no capitalization needed)

By default, the following punctuation marks are considered: commas, periods, and question marks; the remaining punctuation marks were 
removed from the data. This can be changed by introducing new labels in the ``labels.txt`` files.

Each line of the ``labels.txt`` should follow the format: ``[LABEL] [SPACE] [LABEL] [SPACE] [LABEL]`` (for ``labels.txt``). For example, 
labels for the above ``text.txt`` file should be:

    ::

        OU OO OO OO OO OO OU ?U
        OU OO OO OO ...
        ...

The complete list of all possible labels used in this tutorial are: 

- ``OO``
- ``O``
- ``.O``
- ``?O``
- ``OU``
- <blank space> 
- ``U``
- ``.U``
- ``?U``

Converting Raw Data to NeMo Format
----------------------------------

To pre-process the raw text data, stored under :code:`sourced_data_dir` (see the :ref:`raw_data_format_punct`
section), run the following command:

.. code::

    python examples/nlp/token_classification/data/prepare_data_for_punctuation_capitalization.py \
           -s <PATH_TO_THE_SOURCE_FILE>
           -o <PATH_TO_THE_OUTPUT_DIRECTORY>


Required Argument for Dataset Conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :code:`-s` or :code:`--source_file`: path to the raw file
- :code:`-o` or :code:`--output_dir` - path to the directory to store the converted files

After the conversion, the :code:`output_dir` should contain :code:`labels_*.txt` and :code:`text_*.txt` files. The default names
for the training and evaluation in the :code:`conf/punctuation_capitalization_config.yaml` are the following:

.. code::

   .
   |--output_dir
     |-- labels_dev.txt
     |-- labels_train.txt
     |-- text_dev.txt
     |-- text_train.txt

Training Punctuation and Capitalization Model
---------------------------------------------

The language model is initialized with the pre-trained model from `HuggingFace Transformers <https://github.com/huggingface/transformers>`__, 
unless the user provides a pre-trained checkpoint for the language model. Example of model configuration file for training the model can be found at: `NeMo/examples/nlp/token_classification/conf/punctuation_capitalization_config.yaml <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/conf/punctuation_capitalization_config.yaml>`__.

The specification is roughly grouped into the following categories:

- Parameters that describe the training process: **trainer**
- Parameters that describe the datasets: **model.dataset**, **model.train_ds**, **model.validation_ds**
- Parameters that describe the model: **model**

More details about parameters in the config file can be found below and in the `model's config file <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/conf/punctuation_capitalization_config.yaml>`__:

+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **Parameter**                             | **Data Type**   |  **Description**                                                                                             |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **pretrained_model**                      | string          | Path to the pre-trained model ``.nemo`` file or pre-trained model name.                                      |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.dataset.data_dir**                | string          | Path to the data converted to the specified above format.                                                    |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.punct_head.punct_num_fc_layers**  | integer         | Number of fully connected layers.                                                                            |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.punct_head.fc_dropout**           | float           | Activation to use between fully connected layers.                                                            |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.punct_head.activation**           | string          | Dropout to apply to the input hidden states.                                                                 |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.punct_head.use_transrormer_init** | bool            | Whether to initialize the weights of the classifier head with the same approach used in Transformer.         |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.capit_head.punct_num_fc_layers**  | integer         | Number of fully connected layers.                                                                            |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.capit_head.fc_dropout**           | float           | Dropout to apply to the input hidden states.                                                                 |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.capit_head.activation**           | string          | Activation function to use between fully connected layers.                                                   |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.capit_head.use_transrormer_init** | bool            | Whether to initialize the weights of the classifier head with the same approach used in Transformer.         |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **training_ds.text_file**                 | string          | Name of the text training file located at ``data_dir``.                                                      |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **training_ds.labels_file**               | string          | Name of the labels training file located at ``data_dir``, such as ``labels_train.txt``.                      |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **training_ds.num_samples**               | integer         | Number of samples to use from the training dataset, ``-1`` - to use all.                                     |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **validation_ds.text_file**               | string          | Name of the text file for evaluation, located at ``data_dir``.                                               |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **validation_ds.labels_file**             | string          | Name of the labels dev file located at ``data_dir``, such as ``labels_dev.txt``.                             |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **validation_ds.num_samples**             | integer         | Number of samples to use from the dev set, ``-1`` - to use all.                                              |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+

For more information, refer to the :ref:`nlp_model` section.

To train the model from scratch, run:

.. code::

      python examples/nlp/token_classification/punctuation_and_capitalization_train.py \
             model.dataset.data_dir=<PATH/TO/DATA_DIR> \
             trainer.gpus=[0,1] \
             optim.name=adam \
             optim.lr=0.0001

The above command will start model training on GPUs 0 and 1 with Adam optimizer and learning rate of 0.0001; and the trained model is 
stored in the ``nemo_experiments/Punctuation_and_Capitalization`` folder.

To train from the pre-trained model, run:

.. code::

      python examples/nlp/token_classification/punctuation_and_capitalization_train.py \
             model.dataset.data_dir=<PATH/TO/DATA_DIR> \
             pretrained_model=<PATH/TO/SAVE/.nemo>


Required Arguments for Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`model.dataset.data_dir`: Path to the `data_dir` with the pre-processed data files.


.. note::

    All parameters defined in the configuration file can be changed with command arguments. For example, the sample config file 
    mentioned above has :code:`validation_ds.batch_size` set to ``64``. However, if you see that the GPU utilization can be
    optimized further by using a larger batch size, you may override to the desired value by adding the field :code:`validation_ds.batch_size=128`
    over the command-line. You can repeat this with any of the parameters defined in the sample configuration file.

Inference
---------

Inference is performed by a script `examples/nlp/token_classification/punctuate_capitalize_infer.py <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/punctuate_capitalize_infer.py>`_

.. code::

    python punctuate_capitalize_infer.py \
        --input_manifest <PATH_TO_INPUT_MANIFEST> \
        --output_manifest <PATH_TO_OUTPUT_MANIFEST> \
        --pretrained_name punctuation_en_bert \
        --max_seq_length 64 \
        --margin 16 \
        --step 8

:code:`<PATH_TO_INPUT_MANIFEST>` is a path to NeMo :ref:`ASR manifest <_LibriSpeech_dataset>` with text in which you need to
restore punctuation and capitalization. If manifest contains :code:`'pred_text'` key, then :code:`'pred_text'` elements
will be processed. Otherwise, punctuation and capitalization will be restored in :code:`'text'` elements.

:code:`<PATH_TO_OUTPUT_MANIFEST>` is a path to NeMo ASR manifest into which result will be saved. The text with restored
punctuation and capitalization is saved into :code:`'pred_text'` elements if :code:`'pred_text'` key is present in
input manifest. Otherwise result will be saved into :code:`'text'` elements.

Alternatively you can pass data for restoring punctuation and capitalization as plain text. See help for parameters :code:`--input_text`
and :code:`--output_text` of the script
`punctuate_capitalize_infer.py <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/punctuate_capitalize_infer.py>`_.

The script `punctuate_capitalize_infer.py <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/punctuate_capitalize_infer.py>`_
can restore punctuation and capitalization in a text of arbitrary length. Long sequences are split into segments
:code:`--max_seq_length - 2` tokens each. Each segment starts and ends with :code:`[CLS]` and :code:`[SEP]`
tokens correspondingly. Every segment is offset to the previous one by :code:`--step` tokens. For example, if
every character is a token, :code:`--max_seq_length=5`, :code:`--step=2`, then text :code:`"hello"` will be split into
segments :code:`[['[CLS]', 'h', 'e', 'l', '[SEP]'], ['[CLS]', 'l', 'l', 'o', '[SEP]']]`.

If segments overlap, then predicted probabilities for a token present in several segments are multiplied before
before selecting the best candidate.

Splitting leads to pour performance of a model near edges of segments. Use parameter :code:`--margin` to discard :code:`--margin`
probabilities predicted for :code:`--margin` tokens near segment edges. For example, if
every character is a token, :code:`--max_seq_length=5`, :code:`--step=1`, :code:`--margin=1`, then text :code:`"hello"` will be split into
segments :code:`[['[CLS]', 'h', 'e', 'l', '[SEP]'], ['[CLS]', 'e', 'l', 'l', '[SEP]'], ['[CLS]', 'l', 'l', 'o', '[SEP]']]`.
Before calculating final predictions, probabilities for tokens marked by asterisk are removed: :code:`[['[CLS]', 'h', 'e', 'l'*, '[SEP]'*], ['[CLS]'*, 'e'*, 'l', 'l'*, '[SEP]'*], ['[CLS]'*, 'l'*, 'l', 'o', '[SEP]']]`


Model Evaluation
----------------

An example script on how to evaluate the pre-trained model, can be found at `examples/nlp/token_classification/punctuation_capitalization_evaluate.py <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/punctuation_capitalization_evaluate.py>`_.

To start evaluation of the pre-trained model, run:

.. code::

    python punctuation_capitalization_evaluate.py \
           model.dataset.data_dir=<PATH/TO/DATA/DIR>  \
           pretrained_model=punctuation_en_bert \
           model.test_ds.text_file=<text_dev.txt> \
           model.test_ds.labels_file=<labels_dev.txt>


Required Arguments
^^^^^^^^^^^^^^^^^^

- :code:`pretrained_model`: pretrained Punctuation and Capitalization model from ``list_available_models()`` or path to a ``.nemo``
file. For example: ``punctuation_en_bert`` or ``your_model.nemo``.
- :code:`model.dataset.data_dir`: path to the directory that containes :code:`model.test_ds.text_file` and :code:`model.test_ds.labels_file`

During evaluation of the :code:`test_ds`, the script generates two classification reports: one for capitalization task and another
one for punctuation task. This classification reports include the following metrics:

- :code:`Precision`
- :code:`Recall`
- :code:`F1`

More details about these metrics can be found `here <https://en.wikipedia.org/wiki/Precision_and_recall>`__.

References
----------

.. bibliography:: nlp_all.bib
    :style: plain
    :labelprefix: NLP-PUNCT
    :keyprefix: nlp-punct-

