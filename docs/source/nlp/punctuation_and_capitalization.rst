.. _punctuation_and_capitalization:

Punctuation and Capitalization Model Introduction
=================================================

Automatic Speech Recognition (ASR) systems typically generate text with no punctuation and capitalization of the words. \
Besides being hard to read, the ASR output could be an input to named entity recognition, \
machine translation or text-to-speech models. These model models could potentially benefit when the input text contains \
punctuation and the words are capitalized correctly.

For each word in the input text, the model:

1. predicts a punctuation mark that should follow the word (if any). The model supports commas, periods and question marks.
2. predicts if the word should be capitalized or not.

.. note::

    We recommend you try this model in a Jupyter notebook \
    (can run on `Google's Colab <https://colab.research.google.com/notebooks/intro.ipynb>`_.): \
    `NeMo/tutorials/nlp/Punctuation_and_Capitalization.ipynb <https://github.com/NVIDIA/NeMo/blob/main/tutorials/nlp/Punctuation_and_Capitalization.ipynb>`__.
    Connect to an instance with a GPU (Runtime -> Change runtime type -> select "GPU" for hardware accelerator)

    An example script on how to run training and inference with the model could be found here: `NeMo/examples/nlp/token_classification/punctuation_capitalization.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/punctuation_capitalization.py>`__.

    Default configuration file for the model could be found at: `NeMo/examples/nlp/token_classification/conf/punctuation_capitalization_config.yaml <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/conf/punctuation_capitalization_config.yaml>`__.



.. _dataset_punctuation_and_capitalization:

Data Input for Punctuation and Capitalization model
---------------------------------------------------

Raw Data Format
---------------

This model can work with any text dataset, although it is recommended to balance the data, especially for the punctuation task.

Before pre-processing the data to the required format, the data should be split into train.txt and dev.txt (and optionally test.txt).
The development set (or dev set) will be used to evaluate the performance of the model during model training. \
The hyper-parameters search and model selection should be based on the dev set, while the final evaluation of the selected model \
should be performed on the test set.

Each line in the train.txt/dev.txt/test.txt should represent one or more full and/or truncated sentences.

Example of the train.txt/dev.txt file:

.. code::

    When is the next flight to New York?
    The next flight is ...
    ....


The `source_data_dir` structure should look like this:

.. code::

   .
   |--sourced_data_dir
     |-- dev.txt
     |-- test.txt
     |-- train.txt



NeMo Data Format for training the model
---------------------------------------

Raw data files from the `source_data_dir` described above should be converted to the following format with `TODO add script to NeMo from TLT`:
The training and evaluation data is divided into 2 files: text.txt and labels.txt. \
Each line of the text.txt file contains text sequences, where words are separated with spaces, i.e. \
[WORD] [SPACE] [WORD] [SPACE] [WORD], for example:

    ::

        when is the next flight to new york
        the next flight is ...
        ...

The labels.txt file contains corresponding labels for each word in text.txt, the labels are separated with spaces. \
Each label in labels.txt file consists of 2 symbols:

* the first symbol of the label indicates what punctuation mark should follow the word (where O means no punctuation needed);
* the second symbol determines if a word needs to be capitalized or not (where U indicates that the word should be upper cased, and O - no capitalization needed.)

By default the following punctuation marks are considered: commas, periods, and question marks; the rest punctuation marks were removed from the data.
This can be changed by introducing new labels in the labels.txt files

Each line of the labels.txt should follow the format: [LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (for labels.txt). \
For example, labels for the above text.txt file should be:

    ::

        OU OO OO OO OO OO OU ?U
        OU OO OO OO ...
        ...

The complete list of all possible labels for this task used in this tutorial is: OO, ,O, .O, ?O, OU, ,U, .U, ?U.

Converting Raw data to NeMo format
----------------------------------

To pre-process the raw text data, stored under :code:`sourced_data_dir` (see the :ref:`Dataset<dataset_punctuation_and_capitalization>`
section), run the following command:

.. code::

    TODO


Convert Dataset Required Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`-e`: The experiment specification file.
* :code:`source_data_dir` - path to the raw data
* :code:`target_data_dir` - path to store the processed files
* :code:`-r`: Path to the directory to store the results.




+--------------------+----------------+---------------------------------+------------------------------------------------+-------------------------------+
| **Parameter**      | **Datatype**   | **Default**                     | **Description**                                | **Supported Values**          |
+====================+================+=================================+================================================+===============================+
| source_data_dir    | string         | -                               | Path to the dataset source data directory      | -                             |
+--------------------+----------------+---------------------------------+------------------------------------------------+-------------------------------+
| target_data_dir    | string         | -                               | Path to the dataset target data directory      | -                             |
+--------------------+----------------+---------------------------------+------------------------------------------------+-------------------------------+
| list_of_file_names | List of strings| ['train.txt','dev.txt']         | List of files for conversion                   | -                             |
+--------------------+----------------+---------------------------------+------------------------------------------------+-------------------------------+

After the conversion, the :code:`target_data_dir` should contain the following files:

.. code::

   .
   |--target_data_dir
     |-- labels_dev.txt
     |-- labels_test.txt
     |-- labels_train.txt
     |-- text_dev.txt
     |-- text_test.txt
     |-- text_train.txt

Training a Punctuation and Capitalization model
-----------------------------------------------

In the Punctuation and Capitalization Model, we are jointly training two token-level classifiers on top of a pre-trained \
language model, such as `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`__.

Unless the user provides a pre-trained checkpoint for the language model, the language model is initialized with the
pre-trained model from `HuggingFace Transformers <https://github.com/huggingface/transformers>`__.

Example of model configuration file for training the model could be found at: `NeMo/examples/nlp/token_classification/conf/punctuation_capitalization_config.yaml <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/conf/punctuation_capitalization_config.yaml>`__.

The specification can be roughly grouped into the following categories:

* Parameters that describe the training process: `trainer`
* Parameters that describe the datasets: `model.dataset`, `model.train_ds`, `model.validation_ds`
* Parameters that describe the model: `model`

More details about parameters in the spec file could be found below:

+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **Parameter**                             | **Data Type**   |   **Default**                                                                    | **Description**                                                                                              |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| pretrained_model                          | string          | --                                                                               | Path to the pre-trained model .nemo file or pre-trained model name                                           |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| data_dir                                  | string          | --                                                                               | Path to the data converted to the specified above format                                                     |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
|trainer.max_epochs                         | integer         | 5                                                                                | Maximum number of epochs to train the model                                                                  |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.punct_label_ids                     | dictionary      | O: 0, ',': 1, '.': 2, '?': 3                                                     | Labels string name to integer mapping for punctuation task, do NOT change                                    |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.capit_label_ids                     | dictionary      | O: 0, U: 1                                                                       | Labels string name to integer mapping for capitalization task, do NOT change                                 |
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
| model.punct_head.punct_num_fc_layers      | integer         | 1                                                                                | Number of fully connected layers                                                                             |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.punct_head.fc_dropout               | float           | 0.1                                                                              | Activation to use between fully connected layers                                                             |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.punct_head.activation               | string          | 'relu'                                                                           | Dropout to apply to the input hidden states                                                                  |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.punct_head.use_transrormer_init     | bool            | True                                                                             | Whether to initialize the weights of the classifier head with the same approach used in Transformer          |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.capit_head.punct_num_fc_layers      | integer         | 1                                                                                | Number of fully connected layers                                                                             |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.capit_head.fc_dropout               | float           | 0.1                                                                              | Activation to use between fully connected layers                                                             |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.capit_head.activation               | string          | 'relu'                                                                           | Dropout to apply to the input hidden states                                                                  |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.capit_head.use_transrormer_init     | bool            | True                                                                             | Whether to initialize the weights of the classifier head with the same approach used in Transformer          |
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
| optim.lr                                  | float           | 1e-5                                                                             | Learning rate to use for training                                                                            |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| optim.weight_decay                        | float           | 0                                                                                | Weight decay to use for training                                                                             |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| optim.sched.name                          | string          | WarmupAnnealing                                                                  | Warm up schedule                                                                                             |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| optim.sched.warmup_ratio                  | float           | 0.1                                                                              | Warm up ratio                                                                                                |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+

Example of the command for training the model:

.. code::

    tlt punctuation_and_capitalization train [-h] \
                                              -e /specs/nlp/punctuation_and_capitalization/train.yaml \
                                              -r /results/punctuation_and_capitalization/train/ \
                                              -g 4 \
                                              data_dir=/path/to/data_dir \
                                              trainer.max_epochs=2 \
                                              training_ds.num_samples=-1  \
                                              validation_ds.num_samples=-1 \
                                              -k $KEY

Required Arguments for Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`-e`: The experiment specification file to set up training.
* :code:`-r`: Path to the directory to store the results.
* :code:`-k`: Encryption key
* :code:`data_dir`: Path to the `data_dir` with the processed data files.

Optional Arguments
^^^^^^^^^^^^^^^^^^

* :code:`-h, --help`: Show this help message and exit
* :code:`-g`: The number of GPUs to be used in evaluation in a multi-gpu scenario (default: 1).
* Other arguments to override fields in the specification file.

.. note::

    While the arguments are defined in the spec file, if you wish to override these parameter definitions in the spec file \
    and experiment with them, you may do so over command line by simple defining the param. \
    For example, the sample spec file mentioned above has :code:`validation_ds.batch_size` set to 64. \
    However, if you see that the GPU utilization can be optimized further by using larger a batch size, \
    you may override to the desired value, by adding the field :code:`validation_ds.batch_size=128` over command line.
    You may repeat this with any of the parameters defined in the sample spec file.

Snippets of the output log from executing the :code:`punctuation_and_capitalization train` command:

.. code::

    # complete model's spec file will be shown
    [NeMo I] Spec file:
    restore_from: ???
    exp_manager:
      explicit_log_dir: null
      exp_dir: null
      name: trained-model
      version: null
      use_datetime_version: true
      resume_if_exists: true
      resume_past_end: false
      resume_ignore_no_checkpoint: true
      create_tensorboard_logger: false
      summary_writer_kwargs: null
      create_wandb_logger: false
      wandb_logger_kwargs: null
      create_checkpoint_callback: true
      checkpoint_callback_params:
        filepath: null
        monitor: val_loss
        verbose: true
        save_last: true
        save_top_k: 3
        save_weights_only: false
        mode: auto
        period: 1
        prefix: null
        postfix: .tlt
        save_best_model: false
      files_to_copy: null
    model:
      tokenizer: ...

    ...

    # The dataset will be processed and tokenized
    [NeMo I punctuation_capitalization_model:251] Setting model.dataset.data_dir to sample/.
    [NeMo I punctuation_capitalization_dataset:289] Processing text_train.txt
    [NeMo I punctuation_capitalization_dataset:333] Using the provided label_ids dictionary.
    [NeMo I punctuation_capitalization_dataset:408] Labels: {'O': 0, ',': 1, '.': 2, '?': 3}
    [NeMo I punctuation_capitalization_dataset:409] Labels mapping saved to : sample/punct_label_ids.csv
    [NeMo I punctuation_capitalization_dataset:408] Labels: {'O': 0, 'U': 1}
    [NeMo I punctuation_capitalization_dataset:409] Labels mapping saved to : sample/capit_label_ids.csv
    [NeMo I punctuation_capitalization_dataset:134] Max length: 35
    [NeMo I data_preprocessing:295] Some stats of the lengths of the sequences:

    # During training, you're going to see a progress bar for both training and evaluation of the model that is done during model training.

    # Once the training is complete, the results are going to be saved to the specified locations
    [NeMo I train:126] Experiment logs saved to 'nemo_experiments/trained-model'
    [NeMo I train:129] Trained model saved to 'nemo_experiments/trained-model/2021/checkpoints/trained-model.tlt'

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
    - the number of layers in the classification heads (`model.punct_head.punct_num_fc_layers` and `model.capit_head.capit_num_fc_layers`)
    - dropout value between layers (`model.punct_head.fc_dropout` and `model.capit_head.fc_dropout`)

- optimizer (`model.optim.name`, for example, `adam`)
- learning rate (`model.optim.lr`, for example, `5e-5`)


Fine-tuning a model on a different dataset
------------------------------------------

In the previous section <ref>:Training a punctuation and capitalization model, \
the Punctuation and Capitalization model was initialized with a pre-trained language model, \
but the classifiers were trained from scratch.
Now, that a user has trained the Punctuation and Capitalization model successfully (let's call it `trained-model.tlt`), \
there maybe scenarios where users are required to retrain this `trained-model.tlt` on a new smaller dataset. \
TLT conversational AI applications provide a separate tool called `fine-tune` to enable this.


Evaluating a trained model
--------------------------

Spec example to evaluate the pre-trained model:

.. code::

    # Name of the .tlt from which the model will be loaded.
    restore_from: trained-model.tlt

    # Test settings: dataset.
    data_dir: ???
    test_ds:
      text_file: text_dev.txt
      labels_file: labels_dev.txt
      batch_size: 64
      shuffle: false
      num_samples: -1 # number of samples to be considered, -1 means all the dataset

Use the following command to evaluate the model:

.. code::

    TODO

Required Arguments for Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`-e`: The experiment specification file to set up evaluation.
* :code:`-r`: Path to the directory to store the results.
* :code:`data_dir`: Path to data directory with the pre-processed data to use for evaluation
* :code:`-m`: Path to the pre-trained model checkpoint for evaluation. Should be a :code:`.tlt` file.
* :code:`-k`: Encryption key

Optional Arguments:
^^^^^^^^^^^^^^^^^^^
* :code:`-h, --help`: Show this help message and exit

:code:`punctuation_and_capitalization evaluate` generates two classification reports: one for capitalization task and \
another one for punctuaion task. This classification reports include the following metrics:
* :code:`Precision`
* :code:`Recall`
* :code:`F1`

More details about these metrics could be found `here <https://en.wikipedia.org/wiki/Precision_and_recall>`__.

Output log from executing the above command (note, the values below are for demonstration purposes only):

.. code::

    Punctuation report:

    label                                                precision    recall       f1        support
    O (label_id: 0)                                        100.00      97.00      98.48        100
    , (label_id: 1)                                        100.00     100.00     100.00          4
    . (label_id: 2)                                         76.92     100.00      86.96         10
    ? (label_id: 3)                                          0.00       0.00       0.00          0
    -------------------
    micro avg                                               97.37      97.37      97.37        114
    macro avg                                               92.31      99.00      95.14        114
    weighted avg                                            97.98      97.37      97.52        114



    Capitalization report:

    label                                                precision    recall       f1         support
    O (label_id: 0)                                         93.62      90.72      92.15         97
    U (label_id: 1)                                         55.00      64.71      59.46         17
    -------------------
    micro avg                                               86.84      86.84      86.84        114
    macro avg                                               74.31      77.71      75.80        114
    weighted avg                                            87.86      86.84      87.27        114


+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **Parameter**                             | **Data Type**   |   **Default**                                                                    | **Description**                                                                                              |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| data_dir                                  | string          | --                                                                               | Path to the data converted to the specified above format                                                     |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| test_ds.text_file                         | string          | text_dev.txt                                                                     | Name of the text file to run evaluation on located at `data_dir`                                             |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| test_ds.labels_file                       | string          | labels_dev.txt                                                                   | Name of the labels dev file located at `data_dir`                                                            |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| test_ds.shuffle                           | bool            | False                                                                            | Whether to shuffle the dev data                                                                              |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| test_ds.num_samples                       | integer         | -1                                                                               | Number of samples to use from the dev set, -1 mean all                                                       |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| test_ds.batch_size                        | integer         | 64                                                                               | Dev set batch size                                                                                           |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+


Running inference using a trained model
---------------------------------------

During inference, a batch of input sentences, listed in the spec files, are passed through the trained model \
to add punctuation and capitalize words.

Before doing inference on the model, specify the list of examples in the spec, for example:

.. code::

    input_batch:
      - 'what can i do for you today'
      - 'how are you'

To run inference:

.. code::

    TODO

Output log from executing the above command:

.. code::

    The prediction results of some sample queries with the trained model:
    Query : what can i do for you today
    Result: What can I do for you today?
    Query : how are you
    Result: How are you?



Required Arguments for Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



