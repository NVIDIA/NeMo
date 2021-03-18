.. _text_classification:

Text Classification Model
=========================

Text Classification Model is a sequence classification model based on BERT-based encoders. It can be used for a
variety of tasks like text classification, sentiment analysis, domain/intent detection for dialogue systems, etc.
The model takes a text input and predicts a label/class for the whole sequence. Megatron-LM and most of the BERT-based encoders
supported by HuggingFace including BERT, RoBERTa and DistilBERT can be used as the encoder of this model.

An example script on how to train the model can be found here: `NeMo/examples/nlp/text_classification/text_classification_with_bert.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/text_classification/text_classification_with_bert.py>`__.
The default configuration file for the model can be found at: `NeMo/examples/nlp/text_classification/conf/text_classification_config.yaml <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/text_classification/conf/text_classification_config.yaml>`__.

There is also a Jupyter notebook which has shown how to work with this model. We recommend you try this model in the Jupyter notebook (can run on `Google's Colab <https://colab.research.google.com/notebooks/intro.ipynb>`_.):
`NeMo/tutorials/nlp/Text_Classification_Sentiment_Analysis.ipynb <https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0rc1/tutorials/nlp/Text_Classification_Sentiment_Analysis.ipynb>`__.
This tutorial shows an example of how to the text classification model on a sentiment analysis task. You may connect to an instance with a GPU (Runtime -> Change runtime type -> select "GPU" for hardware accelerator) to run the notebook.

Data Format
-----------

The text classification model uses a simple text format as dataset. It requires the data to be stored in TAB separated files (.tsv) with two columns of sentence and label.
Each line of the data file contains text sequences, where words are separated with spaces and the label is separated with [TAB], i.e.:

.. code::

    [WORD][SPACE][WORD][SPACE][WORD][TAB][LABEL]

Labels need to be integers starting from 0. Some examples taken from SST2 dataset, which is a two-class dataset for sentiment analysis:

.. code::

    saw how bad this movie was  0
    lend some dignity to a dumb story   0
    the greatest musicians  1

You may need separate files for train, validation and test with this format.

Dataset Conversion
------------------

If your dataset is stored in another format, you need to convert it to NeMo's format to use this model.
There are some conversion scripts available for datasets: SST2 :cite:`nlp-textclassify-socher2013`, IMDB :cite:`nlp-textclassify-maas2011`, ChemProt :cite:`nlp-textclassify-lim2018chemical`, and THUCnews :cite:`nlp-textclassify-li2007scalable`. They can to convert them from their original format to NeMo's format.
To convert the original datasets to NeMo's format, you can use 'examples/text_classification/data/import_datasets.py' script as the following example:

.. code::
    python import_datasets.py \
        --dataset_name DATASET_NAME \
        --target_data_dir TARGET_PATH \
        --source_data_dir SOURCE_PATH

It reads the dataset specified by DATASET_NAME from SOURCE_PATH and converts it to NeMo's format. Then saves the new dataset at TARGET_PATH.

Arguments:

- dataset_name: name of the dataset to convert ("sst-2", "chemprot", "imdb", and "thucnews" are currently supported)
- source_data_dir: directory of your dataset
- target_data_dir: directory to save the converted dataset

After the conversion, the TARGET_PATH should contain the following files:

.. code::

   .
   |--TARGET_PATH
     |-- train.tsv
     |-- dev.tsv
     |-- test.tsv

Some datasets do not have the test set or their test set does not have any labels, therefore the corresponding file may be missing.

Model Training
--------------
You may find an example of a config file to be used for training of the text classification model at `NeMo/examples/nlp/text_classification/conf/text_classification_config.yaml <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/text_classification/conf/text_classification_config.yaml>`__.
You can change any of these parameters directly from the config file or update them with the command line arguments.

The config file of the Text Classification Model contains three main sections of trainer, exp_manager, and model.
You may find more detail about the trainer and exp_manager at :doc:`../nlp_model.html`. Some sub-sections of the model section including tokenizer, language_model, optim are shared among most of the NLP models. The details of these sections can be found at :doc:`../nlp_model.html`.

Example of a command for training a text classification model on two GPUs for 50 epochs:

.. code::

    python examples/nlp/text_classification/text_classification_with_bert.py \
        model.training_ds.file_path=<TRAIN_FILE_PATH> \
        model.validation_ds.file_path=<VALIDATION_FILE_PATH> \
        trainer.max_epochs=50 \
        trainer.gpus=[0,1] \
        optim.name=adam \
        optim.lr=0.0001 \
        model.nemo_path=<NEMO_FILE_PATH>


By default, the final model after training is saved in the path specified by 'NEMO_FILE_PATH'.

Model Arguments
^^^^^^^^^^^^^^^
The following table lists some of the model's parameters you may use in the config files or set them from command line when training a model:

+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **Parameter**                             | **Data Type**   |   **Default**                                  | **Description**                                                                                              |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.class_labels.class_labels_file      | string          | null                                           | Path to an optional file containing the labels; each line is the string label corresponding to a label       |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.dataset.num_classes                 | int             | ?                                              | Number of the categories or classes, 0 < Label <num_classes                                                  |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.dataset.do_lower_case               | boolean         | true for uncased models, false for cased       | Specifies if inputs should be made lower case, would be set automatically if pre-trained model is used       |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.dataset.max_seq_length              | int             | 256                                            | Maximum length of the input sequences.                                                                       |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.dataset.class_balancing             | string          | null                                           | null or 'weighted_loss'. 'weighted_loss' enables the weighted class balancing to handle unbalanced classes   |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.dataset.use_cache                   | boolean         | false                                          | uses a cache to store the processed dataset, you may use it for large datasets for speed up                  |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.classifier_head.num_output_layers   | integer         | 2                                              | Number of fully connected layers of the Classifier on top of Bert model                                      |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.classifier_head.fc_dropout          | float           | 0.1                                            | Dropout ratio of the fully connected layers                                                                  |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| {training,validation,test}_ds.file_path   | string          | ??                                             | Path of the training '.tsv file                                                                              |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------------+
| {training,validation,test}_ds.batch_size  | integer         | 32                                             | Data loader's batch size                                                                                     |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------------+
| {training,validation,test}_ds.num_workers | integer         | 2                                              | Number of worker threads for data loader                                                                     |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| {training,validation,test}_ds.shuffle     | boolean         | true (training), false (test and validation)   | Shuffles data for each epoch                                                                                 |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| {training,validation,test}_ds.drop_last   | boolean         | false                                          | Specifies if last batch of data needs to get dropped if it is smaller than batch size                        |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| {training,validation,test}_ds.pin_memory  | boolean         | false                                          | Enables pin_memory of PyTorch's data loader to enhance speed                                                 |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| {training,validation,test}_ds.num_samples | integer         | -1                                             | Number of samples to be used from the dataset; -1 means all samples                                          |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+


Training Procedure
^^^^^^^^^^^^^^^^^^

At the start of each training experiment, there will a printed log of the experiment specification including any parameters added or overridden via the command line.
It will also show additional information, such as which GPUs are available and where logs will be saved.
Then it shows some samples from the datasets with their corresponding inputs to the model. It also provides some stats on the lengths of sequences in the dataset.

After each epoch, you should see a summary table of metrics on the validation set which include the following metrics:

* :code:`Precision`
* :code:`Recall`
* :code:`F1`

At the end of training, NeMo will save the last checkpoint at the path specified in '.nemo' format.

Model Evaluation and Inference
------------------------------

After saving the model in '.nemo' format, you may load the model and perform evaluation or inference on the model.
You may find some example in the example script: `NeMo/examples/nlp/text_classification/text_classification_with_bert.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/text_classification/text_classification_with_bert.py>`__

References
----------

.. bibliography:: nlp_all.bib
    :style: plain
    :labelprefix: NLP-TEXTCLASSIFY
    :keyprefix: nlp-textclassify-
