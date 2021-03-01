.. _token_classification:

Token Classification (Named Entity Recognition) Model
=====================================================

TokenClassification Model supports Named entity recognition (NER) and other token level classification tasks, \
as long as the data follows the format specified below. This model card will focus on the NER task.

Named entity recognition (NER), also referred to as entity chunking, identification or extraction, is the task of \
detecting and classifying key information (entities) in text. In other words, a NER model takes a piece of text as \
input and for each word in the text, the model identifies a category the word belongs to.
For example, in a sentence: `Mary lives in Santa Clara and works at NVIDIA`, the model should detect that `Mary` \
is a person, `Santa Clara` is a location and `NVIDIA` is a company.


.. note::

    We recommend you try this model in a Jupyter notebook \
    (can run on `Google's Colab <https://colab.research.google.com/notebooks/intro.ipynb>`_.): \
    `NeMo/tutorials/nlp/Token_Classification_Named_Entity_Recognition.ipynb <https://github.com/NVIDIA/NeMo/blob/main/tutorials/nlp/Token_Classification_Named_Entity_Recognition.ipynb>`__.

    Connect to an instance with a GPU (Runtime -> Change runtime type -> select "GPU" for hardware accelerator)

    An example script on how to train the model could be found here: `NeMo/examples/nlp/token_classification/token_classification_train.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/token_classification_train.py>`__.
    An example script on how to run evaluation and inference could be found here: `NeMo/examples/nlp/token_classification/token_classification_evaluate.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/token_classification_evaluate.py>`__.

    The default configuration file for the model could be found at: `NeMo/examples/nlp/token_classification/conf/token_classification_config.yaml <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/conf/token_classification_config.yaml>`__.




.. _dataset_token_classification:

Data Input for Token Classification model
-----------------------------------------

For pre-training or fine-tuning of the model, the data should be split into 2 files:

- text.txt and
- labels.txt.

Each line of the text.txt file contains text sequences, where words are separated with spaces, i.e.: [WORD] [SPACE] [WORD] [SPACE] [WORD].
The labels.txt file contains corresponding labels for each word in text.txt, the labels are separated with spaces, i.e.: [LABEL] [SPACE] [LABEL] [SPACE] [LABEL].
Example of a text.txt file:

    Jennifer is from New York City .
    She likes ...
    ...

Corresponding labels.txt file:

    B-PER O O B-LOC I-LOC I-LOC O
    O O ...
    ...

Dataset Conversion
------------------

To convert an IOB format (short for inside, outside, beginning) data to the format required for training use
`examples/nlp/token_classification/data/import_from_iob_format.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/data/import_from_iob_format.py)>`_.

.. code::

    # For conversion from IOB format, for example, for CoNLL-2003 dataset:
    python import_from_iob_format.py --data_file=<PATH/TO/THE/FILE/IN/IOB/FORMAT>

Convert Dataset Required Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`--data_file`: path to the file to convert from IOB to NeMo format

After running the above command, the data directory, where the :code:`--data_file` is stored, should contain :code:`text_*.txt` and :code:`labels_*.txt` files.
The default names for the training and evaluation in the :code:`conf/token_classification_config.yaml` are the following:

.. code::

   .
   |--data_dir
     |-- labels_dev.txt
     |-- labels_train.txt
     |-- text_dev.txt
     |-- text_train.txt


Note, the development set (or dev set) will be used to evaluate the performance of the model during model training. \
The hyper-parameters search and model selection should be based on the dev set, while the final evaluation of \
the selected model should be performed on the test set.

Training Token Classification Model
-----------------------------------

In the Token Classification Model, we are jointly training a classifier on top of a pre-trained \
language model, such as `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`__.

Unless the user provides a pre-trained checkpoint for the language model, the language model is initialized with the
pre-trained model from `HuggingFace Transformers <https://github.com/huggingface/transformers>`__.

Example of model configuration file for training the model could be found at: `NeMo/examples/nlp/token_classification/conf/token_classification_config.yaml <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/conf/token_classification_config.yaml>`__.

The specification can be roughly grouped into three categories:

* Parameters that describe the training process
* Parameters that describe the datasets, and
* Parameters that describe the model.

More details about parameters in the spec file could be found below:

+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **Parameter**                             | **Data Type**   |   **Default**                                                                    | **Description**                                                                                              |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.dataset.data_dir                    | string          | --                                                                               | Path to the data converted to the specified above format                                                     |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| trainer.max_epochs                        | integer         | 5                                                                                | Maximum number of epochs to train the model                                                                  |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.tokenizer.tokenizer_name            | string          | Will be filled automatically based on model.language_model.pretrained_model_name | Tokenizer name                                                                                               |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.tokenizer.vocab_file                | string          | null                                                                             | Path to tokenizer vocabulary                                                                                 |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.tokenizer.tokenizer_model           | string          | null                                                                             | Path to tokenizer model (only for sentencepiece tokenizer)                                                   |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.language_model.pretrained_model_name| string          | bert-base-uncased                                                                | Pre-trained language model name, for example: `bert-base-cased` or `bert-base-uncased`                       |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.language_model.lm_checkpoint        | string          | null                                                                             | Path to the pre-trained language model checkpoint                                                            |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.language_model.config_file          | string          | null                                                                             | Path to the pre-trained language model config file                                                           |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.language_model.config               | dictionary      | null                                                                             | Config of the pre-trained language model                                                                     |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.head.num_fc_layers                  | integer         | 2                                                                                | Number of fully connected layers                                                                             |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.head.fc_dropout                     | float           | 0.5                                                                              | Activation to use between fully connected layers                                                             |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.head.activation                     | string          | 'relu'                                                                           | Dropout to apply to the input hidden states                                                                  |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.punct_head.use_transrormer_init     | bool            | True                                                                             | Whether to initialize the weights of the classifier head with the same approach used in Transformer          |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| training_ds.text_file                     | string          | text_train.txt                                                                   | Name of the text training file located at `data_dir`                                                         |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| training_ds.labels_file                   | string          | labels_train.txt                                                                 | Name of the labels training file located at `data_dir`                                                       |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| training_ds.shuffle                       | bool            | True                                                                             | Whether to shuffle the training data                                                                         |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| training_ds.num_samples                   | integer         | -1                                                                               | Number of samples to use from the training dataset, -1 mean all                                              |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| training_ds.batch_size                    | integer         | 64                                                                               | Training data batch size                                                                                     |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| validation_ds.text_file                   | string          | text_dev.txt                                                                     | Name of the text file for evaluation, located at `data_dir`                                                  |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| validation_ds.labels_file                 | string          | labels_dev.txt                                                                   | Name of the labels dev file located at `data_dir`                                                            |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| validation_ds.shuffle                     | bool            | False                                                                            | Whether to shuffle the dev data                                                                              |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| validation_ds.num_samples                 | integer         | -1                                                                               | Number of samples to use from the dev set, -1 mean all                                                       |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| validation_ds.batch_size                  | integer         | 64                                                                               | Dev set batch size                                                                                           |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| optim.name                                | string          | adam                                                                             | Optimizer to use for training                                                                                |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| optim.lr                                  | float           | 5e-5                                                                             | Learning rate to use for training                                                                            |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| optim.weight_decay                        | float           | 0                                                                                | Weight decay to use for training                                                                             |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| optim.sched.name                          | string          | WarmupAnnealing                                                                  | Warm up schedule                                                                                             |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| optim.sched.warmup_ratio                  | float           | 0.1                                                                              | Warm up ratio                                                                                                |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+

Example of the command for training the model:

.. code::

    python token_classification_train.py \
           model.dataset.data_dir=<PATH_TO_DATA_DIR>  \
           trainer.max_epochs=<NUM_EPOCHS> \
           trainer.gpus=[<CHANGE_TO_GPU(s)_YOU_WANT_TO_USE>]


Required Arguments for Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`model.dataset.data_dir`: Path to the directory with pre-processed data.

Optional Arguments
^^^^^^^^^^^^^^^^^^

* Other arguments to override fields in the specification file.

.. note::

    While the arguments are defined in the spec file, if you wish to override these parameter definitions in the spec file \
    and experiment with them, you may do so over command line by simple defining the param. \
    For example, the sample spec file mentioned above has :code:`validation_ds.batch_size` set to 64. \
    However, if you see that the GPU utilization can be optimized further by using larger a batch size, \
    you may override to the desired value, by adding the field :code:`validation_ds.batch_size=128` over command line.
    You may repeat this with any of the parameters defined in the sample spec file.

Important parameters
^^^^^^^^^^^^^^^^^^^^

Below is the list of parameters could help improve the model:

- language model (`model.language_model.pretrained_model_name`)
    - pre-trained language model name, such as:
    - `megatron-bert-345m-uncased`, `megatron-bert-345m-cased`, `biomegatron-bert-345m-uncased`, `biomegatron-bert-345m-cased`, `bert-base-uncased`, `bert-large-uncased`, `bert-base-cased`, `bert-large-cased`
    - `distilbert-base-uncased`, `distilbert-base-cased`,
    - `roberta-base`, `roberta-large`, `distilroberta-base`
    - `albert-base-v1`, `albert-large-v1`, `albert-xlarge-v1`, `albert-xxlarge-v1`, `albert-base-v2`, `albert-large-v2`, `albert-xlarge-v2`, `albert-xxlarge-v2`

- classification head parameters:
    - the number of layers in the classification head (`model.head.num_fc_layers`)
    - dropout value between layers (`model.head.fc_dropout`)

- optimizer (`model.optim.name`, for example, `adam`)
- learning rate (`model.optim.lr`, for example, `5e-5`)


Inference
---------

An example script on how to run inference on a few examples, could be found
at `examples/nlp/token_classification/token_classification_evaluate.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/token_classification_evaluate.py>`_.

To run inference with the pre-trained model on a few examples, run:

.. code::

    python token_classification_evaluate.py \
           pretrained_model=<PRETRAINED_MODEL>

Required Arguments for inference:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`pretrained_model`: pretrained TokenClassification model from list_available_models() or path to a .nemo file, for example: ner_en_bert or your_model.nemo


Model Evaluation
----------------

An example script on how to evaluate the pre-trained model, could be found
at `examples/nlp/token_classification/token_classification_evaluate.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/token_classification_evaluate.py>`_.

To run evaluation of the pre-trained model, run:

.. code::

    python token_classification_evaluate.py \
           model.dataset.data_dir=<PATH/TO/DATA/DIR>  \
           pretrained_model=ner_en_bert \
           model.test_ds.text_file=<text_*.txt> \
           model.test_ds.labels_file=<labels_*.txt> \
           model.dataset.max_seq_length=512


Required Arguments:
^^^^^^^^^^^^^^^^^^^
* :code:`pretrained_model`: pretrained TokenClassification model from list_available_models() or path to a .nemo file, for example: ner_en_bert or your_model.nemo
* :code:`model.dataset.data_dir`: Path to the directory that containes :code:`model.test_ds.text_file` and :code:`model.test_ds.labels_file`.


Optional Arguments:
^^^^^^^^^^^^^^^^^^^
* :code:`model.test_ds.text_file` and :code:`model.test_ds.labels_file`: text_*.txt and labels_*.txt file names is the default text_dev.txt and labels_dev.txt from the config files should be overwritten.
* Other :code:`model.dataset` or :code:`model.test_ds` arguments to override fields in the config file of the pre-trained model.


During evaluation of the :code:`test_ds`, the script generates a classification reports that includes the following metrics:

* :code:`Precision`
* :code:`Recall`
* :code:`F1`

More details about these metrics could be found `here <https://en.wikipedia.org/wiki/Precision_and_recall>`__.

References
----------

.. bibliography:: nlp_references.bib
    :style: plain
    :labelprefix: nlp-
    :keyprefix: nlp-