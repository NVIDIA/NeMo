.. _token_classification:

Token Classification (Named Entity Recognition)
===============================================


Introduction
------------

TokenClassification Model supports Named entity recognition (NER) and other token level classification tasks, \
as long as the data follows the format specified below. This model card will focus on the NER task.

Named entity recognition (NER), also referred to as entity chunking, identification or extraction, is the task of \
detecting and classifying key information (entities) in text. In other words, a NER model takes a piece of text as \
input and for each word in the text, the model identifies a category the word belongs to.
For example, in a sentence: `Mary lives in Santa Clara and works at NVIDIA`, the model should detect that `Mary` \
is a person, `Santa Clara` is a location and `NVIDIA` is a company.

.. note::

    This documentation follows [TODO: add link to token-classification.ipynb]



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

To convert an IOB format (short for inside, outside, beginning) data to the format required for training.

.. code::

    # For conversion from IOB format, for example, for CoNLL-2003 dataset:
    TODO

The `source_data_dir` structure should look like this (test.txt is optional):

.. code::

   .
   |--sourced_data_dir
     |-- dev.txt
     |-- test.txt
     |-- train.txt

Note, the development set (or dev set) will be used to evaluate the performance of the model during model training. \
The hyper-parameters search and model selection should be based on the dev set, while the final evaluation of \
the selected model should be performed on the test set.

After the conversion, the `target_data_dir` should contain the following files:

.. code::

   .
   |--target_data_dir
     |-- label_ids.csv
     |-- labels_dev.txt
     |-- labels_test.txt
     |-- labels_train.txt
     |-- text_dev.txt
     |-- text_test.txt
     |-- text_train.txt


Training a Token Classification model
-------------------------------------

In the Token Classification Model, we are jointly training a classifier on top of a pre-trained \
language model, such as `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`__.

Unless the user provides a pre-trained checkpoint for the language model, the language model is initialized with the
pre-trained model from `HuggingFace Transformers <https://github.com/huggingface/transformers>`__.



The specification can be roughly grouped into three categories:

* Parameters that describe the training process
* Parameters that describe the datasets, and
* Parameters that describe the model.

More details about parameters in the spec file could be found below:

+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **Parameter**                             | **Data Type**   |   **Default**                                                                    | **Description**                                                                                              |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| data_dir                                  | string          | --                                                                               | Path to the data converted to the specified above format                                                     |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| trainer.max_epochs                        | integer         | 5                                                                                | Maximum number of epochs to train the model                                                                  |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.label_ids                           | string          | --                                                                               | Path to the string labels to integet mapping (is generated during the dataset conversion step)               |
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

      token_classification train [-h] \
                                    -e /specs/nlp/token_classification/train.yaml \
                                    -r /results/token_classification/train/ \
                                    -g 1 \
                                    -k $KEY
                                    data_dir=/path/to/data_dir \
                                    model.label_ids=/path/to/label_ids.csv \
                                    trainer.max_epochs=5 \
                                    training_ds.num_samples=-1 \
                                    validation_ds.num_samples=-1


Required Arguments for Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`-e`: The experiment specification file to set up training.
* :code:`-r`: Path to the directory to store the results.
* :code:`-k`: Encryption key
* :code:`data_dir`: Path to the `data_dir` with the processed data files.
* :code:`model.label_ids`: Path to the `label_ids.csv` file, usually stored at `data_dir`

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

Snippets of the output log from executing the :code:`token_classification train` command:

.. code::

    # complete model's spec file will be shown
    [NeMo I train:93] Spec file:
        restore_from: ???
        exp_manager:
          explicit_log_dir: /results/token_classification/train/
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
            postfix: . 
            save_best_model: false
          files_to_copy: null
        model:
          tokenizer:
            tokenizer_name: ...
        ...

    [NeMo I exp_manager:186] Experiments will be logged at /results/token_classification/train/

    # The dataset will be processed and tokenized
    [NeMo I token_classification_model:61] Reusing label_ids file found at data_dir/label_ids.csv.
    Using bos_token, but it is not set yet.
    Using eos_token, but it is not set yet.
    [NeMo I token_classification_model:105] Setting model.dataset.data_dir to data_dir.

    [NeMo I 2021-01-21 17:57:14 token_classification_utils:54] Processing data_dir/labels_train.txt
    [NeMo I 2021-01-21 17:57:14 token_classification_utils:75] Using provided labels mapping {'O': 0, 'B-GPE': 1, 'B-LOC': 2, 'B-MISC': 3, 'B-ORG': 4, 'B-PER': 5, 'B-TIME': 6, 'I-GPE': 7, 'I-LOC': 8, 'I-MISC': 9, 'I-ORG': 10, 'I-PER': 11, 'I-TIME': 12}
    [NeMo I 2021-01-21 17:57:15 token_classification_utils:101] Three most popular labels in data_dir/labels_train.txt:
    [NeMo I 2021-01-21 17:57:15 data_preprocessing:131] label: 0, 18417 out of 21717 (84.80%).
    [NeMo I 2021-01-21 17:57:15 data_preprocessing:131] label: 2, 829 out of 21717 (3.82%).
    [NeMo I 2021-01-21 17:57:15 data_preprocessing:131] label: 6, 433 out of 21717 (1.99%).
    [NeMo I 2021-01-21 17:57:15 token_classification_utils:103] Total labels: 21717. Label frequencies - {0: 18417, 2: 829, 6: 433, 4: 357, 11: 352, 5: 349, 1: 338, 10: 281, 8: 181, 12: 142, 3: 21, 9: 12, 7: 5}
    [NeMo I 2021-01-21 17:57:15 token_classification_utils:112] Class Weights: {0: 0.09070632901875775, 2: 2.015124802820822, 6: 3.858056493160419, 4: 4.679379444085327, 11: 4.7458479020979025, 5: 4.786643156270664, 1: 4.942421483841602, 10: 5.9449767314535995, 8: 9.229494262643433, 12: 11.764355362946912, 3: 79.54945054945055, 9: 139.21153846153845, 7: 334.10769230769233}
    [NeMo I 2021-01-21 17:57:15 token_classification_utils:116] Class weights saved to data_dir/labels_train_weights.p
    [NeMo I 2021-01-21 17:57:19 token_classification_dataset:116] Setting Max Seq length to: 64
    [NeMo I 2021-01-21 17:57:19 data_preprocessing:295] Some stats of the lengths of the sequences:
    [NeMo I 2021-01-21 17:57:19 data_preprocessing:301] Min: 6 |                  Max: 64 |                  Mean: 26.357 |                  Median: 26.0
    [NeMo I 2021-01-21 17:57:19 data_preprocessing:303] 75 percentile: 32.00
    [NeMo I 2021-01-21 17:57:19 data_preprocessing:304] 99 percentile: 51.00
    [NeMo W 2021-01-21 17:57:19 token_classification_dataset:145] 0 are longer than 64
    [NeMo I 2021-01-21 17:57:19 token_classification_dataset:148] *** Example ***
    [NeMo I 2021-01-21 17:57:19 token_classification_dataset:149] i: 0
    [NeMo I 2021-01-21 17:57:19 token_classification_dataset:150] subtokens: [CLS] new zealand ' s cricket team has scored a morale - boost ##ing win over bangladesh in the first of three one - day internationals in new zealand . [SEP]
    [NeMo I 2021-01-21 17:57:19 token_classification_dataset:151] loss_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    [NeMo I 2021-01-21 17:57:19 token_classification_dataset:152] input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    [NeMo I 2021-01-21 17:57:19 token_classification_dataset:153] subtokens_mask: 0 1 1 1 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    [NeMo I 2021-01-21 17:57:19 token_classification_dataset:155] labels: 0 2 8 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 6 12 12 12 12 12 0 0 2 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    [NeMo I 2021-01-21 17:57:19 token_classification_dataset:264] features saved to data_dir/cached_text_train.txt_BertTokenizer_128_30522_-1
    [NeMo I 2021-01-21 17:57:19 token_classification_utils:54] Processing data_dir/labels_dev.txt
    [NeMo I 2021-01-21 17:57:19 token_classification_utils:75] Using provided labels mapping {'O': 0, 'B-GPE': 1, 'B-LOC': 2, 'B-MISC': 3, 'B-ORG': 4, 'B-PER': 5, 'B-TIME': 6, 'I-GPE': 7, 'I-LOC': 8, 'I-MISC': 9, 'I-ORG': 10, 'I-PER': 11, 'I-TIME': 12}
    [NeMo I 2021-01-21 17:57:20 token_classification_utils:101] Three most popular labels in data_dir/labels_dev.txt:
    [NeMo I 2021-01-21 17:57:20 data_preprocessing:131] label: 0, 18266 out of 21775 (83.89%).
    [NeMo I 2021-01-21 17:57:20 data_preprocessing:131] label: 2, 809 out of 21775 (3.72%).
    [NeMo I 2021-01-21 17:57:20 data_preprocessing:131] label: 6, 435 out of 21775 (2.00%).
    [NeMo I 2021-01-21 17:57:20 token_classification_utils:103] Total labels: 21775. Label frequencies - {0: 18266, 2: 809, 6: 435, 4: 418, 11: 414, 5: 392, 1: 351, 10: 351, 8: 174, 12: 146, 7: 8, 3: 8, 9: 3}
    [NeMo I 2021-01-21 17:57:24 token_classification_dataset:116] Setting Max Seq length to: 70
    [NeMo I 2021-01-21 17:57:24 data_preprocessing:295] Some stats of the lengths of the sequences:
    [NeMo I 2021-01-21 17:57:24 data_preprocessing:301] Min: 7 |                  Max: 70 |                  Mean: 26.437 |                  Median: 26.0
    [NeMo I 2021-01-21 17:57:24 data_preprocessing:303] 75 percentile: 33.00
    [NeMo I 2021-01-21 17:57:24 data_preprocessing:304] 99 percentile: 50.00
    [NeMo W 2021-01-21 17:57:24 token_classification_dataset:145] 0 are longer than 70
    [NeMo I 2021-01-21 17:57:24 token_classification_dataset:148] *** Example ***
    [NeMo I 2021-01-21 17:57:24 token_classification_dataset:149] i: 0
    [NeMo I 2021-01-21 17:57:24 token_classification_dataset:150] subtokens: [CLS] hamas refuses to recognize israel , and has vowed to undermine palestinian leader mahmoud abbas ' s efforts to make peace with the jewish state . [SEP]
    [NeMo I 2021-01-21 17:57:24 token_classification_dataset:151] loss_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    [NeMo I 2021-01-21 17:57:24 token_classification_dataset:152] input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    [NeMo I 2021-01-21 17:57:24 token_classification_dataset:153] subtokens_mask: 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    [NeMo I 2021-01-21 17:57:24 token_classification_dataset:155] labels: 0 4 0 0 0 2 0 0 0 0 0 0 1 0 5 11 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    [NeMo I 2021-01-21 17:57:24 token_classification_dataset:264] features saved to data_dir/cached_text_dev.txt_BertTokenizer_128_30522_-1

    [NeMo I 2021-01-21 17:00:09 modelPT:830] Optimizer config = Adam (
        Parameter Group 0
            amsgrad: False
            betas: (0.9, 0.999)
            eps: 1e-08
            lr: 5e-05
            weight_decay: 0.0
        )
    [NeMo I 2021-01-21 17:00:09 lr_scheduler:621] Scheduler "<nemo.core.optim.lr_scheduler.WarmupAnnealing object at 0x7f3b6d05f400>"
        will be used during training (effective maximum steps = 16) -
        Parameters :
        (warmup_steps: null
        warmup_ratio: 0.1
        last_epoch: -1
        max_steps: 16
        )
    initializing ddp: GLOBAL_RANK: 0, MEMBER: 1/1
    [NeMo I 2021-01-21 17:00:11 modelPT:704] No optimizer config provided, therefore no optimizer was created

    110 M     Trainable params
    0         Non-trainable params
    110 M     Total params
    Validation sanity check:  50%|████████████████████████████▌                            | 1/2 [00:00<00:00,  1.47it/s][NeMo I 2021-01-21 17:00:13 token_classification_model:178]
        label                                                precision    recall       f1           support
        O (label_id: 0)                                         82.08     100.00      90.16       2300
        B-GPE (label_id: 1)                                      0.00       0.00       0.00         41
        B-LOC (label_id: 2)                                      0.00       0.00       0.00        119
        B-MISC (label_id: 3)                                     0.00       0.00       0.00          2
        B-ORG (label_id: 4)                                      0.00       0.00       0.00         71
        B-PER (label_id: 5)                                      0.00       0.00       0.00         62
        B-TIME (label_id: 6)                                     0.00       0.00       0.00         56
        I-GPE (label_id: 7)                                      0.00       0.00       0.00          4
        I-LOC (label_id: 8)                                      0.00       0.00       0.00         18
        I-MISC (label_id: 9)                                     0.00       0.00       0.00          0
        I-ORG (label_id: 10)                                     0.00       0.00       0.00         52
        I-PER (label_id: 11)                                     0.00       0.00       0.00         61
        I-TIME (label_id: 12)                                    0.00       0.00       0.00         16
        -------------------
        micro avg                                               82.08      82.08      82.08       2802
        macro avg                                                6.84       8.33       7.51       2802
        weighted avg                                            67.38      82.08      74.01       2802

    Training: 0it [00:00, ?it/s]
    [NeMo I 2021-01-21 17:00:38 train:124] Experiment logs saved to 'output'
    [NeMo I 2021-01-21 17:00:38 train:127] Trained model saved to 'output/checkpoints/trained-model. '
    INFO: Internal process exited


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



Fine-tuning a model on a different dataset
------------------------------------------

In the previous section <ref>:Training a token classification model, \
the Token Classification (NER) model was initialized with a pre-trained language model, \
but the classifiers were trained from scratch.
Now, that a user has trained the Token Classification model successfully (let's call it `trained-model. `), \
there maybe scenarios where users are required to retrain this `trained-model. ` on a new smaller dataset. \
  conversational AI applications provide a separate tool called `fine-tune` to enable this.

Note, all labels from the dataset that is used for fine-tuning, should be present in the dataset the model was originally trained.
If it is not the case, use the :code:`  token_classification train` with your data.

Evaluating a trained model
--------------------------

Spec example to evaluate the pre-trained model:

.. code::

    restore_from: trained-model. 
    data_dir: ???

    # Test settings: dataset.
    test_ds:
      text_file: text_dev.txt
      labels_file: labels_dev.txt
      batch_size: 1
      shuffle: false
      num_samples: -1 # number of samples to be considered, -1 means the whole the dataset

Use the following command to evaluate the model:

.. code::

    TBD


Required Arguments for Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`-e`: The experiment specification file to set up evaluation.
* :code:`-r`: Path to the directory to store the results.
* :code:`data_dir`: Path to data directory with the pre-processed data to use for evaluation
* :code:`-m`: Path to the pre-trained model checkpoint for evaluation. Should be a :code:`. ` file.
* :code:`-k`: Encryption key

:code:`token_classification evaluate` generates a classification report that includes the following metrics:

* :code:`Precision`
* :code:`Recall`
* :code:`F1`

More details about these metrics could be found `here <https://en.wikipedia.org/wiki/Precision_and_recall>`__.

Output log for :code:`token_classification evaluate` (note, the values below are for demonstration purposes only):

.. code::

    label                                                precision    recall       f1           support
    O (label_id: 0)                                         83.89     100.00      91.24      18266
    B-GPE (label_id: 1)                                      0.00       0.00       0.00        351
    B-LOC (label_id: 2)                                      0.00       0.00       0.00        809
    B-MISC (label_id: 3)                                     0.00       0.00       0.00          8
    B-ORG (label_id: 4)                                      0.00       0.00       0.00        418
    B-PER (label_id: 5)                                      0.00       0.00       0.00        392
    B-TIME (label_id: 6)                                     0.00       0.00       0.00        435
    I-GPE (label_id: 7)                                      0.00       0.00       0.00          8
    I-LOC (label_id: 8)                                      0.00       0.00       0.00        174
    I-MISC (label_id: 9)                                     0.00       0.00       0.00          3
    I-ORG (label_id: 10)                                     0.00       0.00       0.00        351
    I-PER (label_id: 11)                                     0.00       0.00       0.00        414
    I-TIME (label_id: 12)                                    0.00       0.00       0.00        146
    -------------------
    micro avg                                               83.89      83.89      83.89      21775
    macro avg                                                6.45       7.69       7.02      21775
    weighted avg                                            70.37      83.89      76.53      21775

    Testing: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:39<00:00, 25.59it/s]
    --------------------------------------------------------------------------------
    DATALOADER:0 TEST RESULTS
        {'f1': tensor(7.0182, device='cuda:0'),
         'precision': tensor(6.4527, device='cuda:0'),
         'recall': tensor(7.6923, device='cuda:0'),
         'test_loss': tensor(1.0170, device='cuda:0')}

Running inference using a trained model
---------------------------------------

During inference, a batch of input sentences, listed in the spec files, are passed through the trained model \
to add token classification label.

To run inference on the model, specify the list of examples in the spec, for example:

.. code::

    input_batch:
      - 'We bought four shirts from the Nvidia gear store in Santa Clara.'
      - 'Nvidia is a company.'

To run inference:

.. code::

    TBD

Required Arguments for Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`-e`: The experiment specification file to set up inference.
  This requires the :code:`input_batch` with the list of examples to run inference on.
* :code:`-r`: Path to the directory to store the results.
* :code:`-m`: Path to the pre-trained model checkpoint from which to infer. Should be a :code:`. ` file.
* :code:`-k`: Encryption key


Optional Arguments
^^^^^^^^^^^^^^^^^^

* :code:`-h, --help`: Show this help message and exit
* :code:`-g`: The number of GPUs to be used for fine-tuning in a multi-gpu scenario (default: 1).
* Other arguments to override fields in the specification file.

Output log sample:

.. code::

    Query : we bought four shirts from the nvidia gear store in santa clara.
    Result: we bought four shirts from the nvidia[B-LOC] gear store in santa[B-LOC] clara[I-LOC].
    Nvidia is a company.
    Result: Nvidia[B-ORG] is a company.

