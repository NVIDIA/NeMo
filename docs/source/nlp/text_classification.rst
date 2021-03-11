.. _text_classification:

Text Classification Model
=========================

Text Classification Model is a sequence classification model based on BERT-based encoders. It can be used for a
variety of tasks like text classification, sentiment analysis, domain/intent detection for dialogue systems, etc.
The model takes a text input and classifies it into predefined categories. Most of the BERT-based encoders
supported by HuggingFace including BERT, Megatron-LM, RoBERTa, DistilBERT, XLNet, etc can be used with this model.

...TODO: add a link to list of supported models.

An example script on how to train the model can be found here: `NeMo/examples/nlp/text_classification/text_classification_with_bert.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/text_classification/text_classification_with_bert.py>`__.
The default configuration file for the model can be found at: `NeMo/examples/nlp/text_classification/conf/text_classification_config.yaml <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/text_classification/conf/text_classification_config.yaml>`__.

There is also a Jupyter notebook which has shown how to work with this model. We recommend you try this model in the Jupyter notebook (can run on `Google's Colab <https://colab.research.google.com/notebooks/intro.ipynb>`_.): \
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
There are some conversion scripts available for datasets: SST2, IMDB, ChemProt, and THUCnews. They can to convert them from their original format to NeMo's format.
To convert the original datasets to NeMo's format, you can use 'examples/text_classification/data/import_datasets.py' script as the following example:

.. code::
    python import_datasets.py \
        --dataset_name DATASET_NAME \
        --target_data_dir TARGET_PATH \
        --source_data_dir SOURCE_PATH

Arguments:

- dataset_name: name of the dataset to convert ("sst-2", "chemprot", "imdb", and "thucnews" are currently supported)
- source_data_dir: directory of your dataset
- target_data_dir: directory to save the converted dataset

It converts the SST2 dataset stored as SOURCE_PATH to NeMo's format and saves the new dataset at TARGET_PATH.

You may download the SST2 dataset from 'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip' and
extract it into the folder specified by SOURCE_PATH. After the conversion, the TARGET_PATH should contain the following files:

.. code::

   .
   |--TARGET_PATH
     |-- train.tsv
     |-- dev.tsv


Model Training
--------------
Example of config spec for training *train.yaml* file. You can change any of these parameters and pass them to the training command.

.. code::

    trainer:
      max_epochs: 100

    model:
      # Labels that will be used to "decode" predictions.
      class_labels:
        class_labels_file : null # optional to specify a file containing the list of the labels

      tokenizer:
          tokenizer_name: ${model.language_model.pretrained_model_name} # or sentencepiece
          vocab_file: null # path to vocab file
          tokenizer_model: null # only used if tokenizer is sentencepiece
          special_tokens: null

      language_model:
        pretrained_model_name: bert-base-uncased
        lm_checkpoint: null
        config_file: null # json file, precedence over config
        config: null

      classifier_head:
        # This comes directly from number of labels/target classes.
        num_output_layers: 2
        fc_dropout: 0.1


    training_ds:
      file_path: ???
      batch_size: 64
      shuffle: true
      num_samples: -1 # number of samples to be considered, -1 means all the dataset
      num_workers: 3
      drop_last: false
      pin_memory: false

    validation_ds:
      file_path: ???
      batch_size: 64
      shuffle: false
      num_samples: -1 # number of samples to be considered, -1 means all the dataset
      num_workers: 3
      drop_last: false
      pin_memory: false

    optim:
      name: adam
      lr: 2e-5
      # optimizer arguments
      betas: [0.9, 0.999]
      weight_decay: 0.001

      # scheduler setup
      sched:
        name: WarmupAnnealing
        # Scheduler params
        warmup_steps: null
        warmup_ratio: 0.1
        last_epoch: -1
        # pytorch lightning args
        monitor: val_loss
        reduce_on_plateau: false

Example of the command for training the model on four GPUs for 50 epochs:

.. code::

    tlt text_classification train -e /specs/nlp/text_classification/train.yaml \
    training_ds.file_path=PATH_TO_TRAIN_FILE \
    trainer.max_epochs=50 \
    -g 4  \
    -k $KEY

By default, the final model after training is done is saved in 'trained-model.tlt'.

Required Arguments for Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`-e`: The experiment specification file to set up training.
* :code:`training_ds.file_path`: Path to the training '.tsv' file
* :code:`-k`: Encryption key

Optional Arguments
^^^^^^^^^^^^^^^^^^

* :code:`trainer.max_epochs`: Training epochs number.
* :code:`-g`: Number of GPUs to use for training
* Other arguments to override fields in the specification file.

The following table lists some of the parameters you may use in the config files and set them from command line when training a model:

+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **Parameter**                             | **Data Type**   |   **Default**                                                                    | **Description**                                                                                              |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.class_labels.class_labels_file      | string          | null                                                                             | Path to an optional file containing the labels; each line is the string label corresponding to a label       |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.intent_loss_weight                  | float           | 0.6                                                                              | Relation of intent to slot loss in total loss                                                                |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.tokenizer.tokenizer_name            | string          | Will be filled automatically based on model.language_model.pretrained_model_name | Tokenizer name                                                                                               |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.tokenizer.vocab_file                | string          | null                                                                             | Path to tokenizer vocabulary                                                                                 |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.tokenizer.tokenizer_model           | string          | null                                                                             | Path to tokenizer model (only for sentencepiece tokenizer)                                                   |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.tokenizer.special_tokens            | string          | null                                                                             | Special tokens of the tokenizer if it exists                                                                 |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.language_model.max_seq_length       | integer         | 50                                                                               | Maximal length of the input queries (in tokens)                                                              |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.language_model.pretrained_model_name| string          | bert-base-uncased                                                                | Pre-trained language model name, for example: `bert-base-cased` or `bert-base-uncased`                       |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.language_model.lm_checkpoint        | string          | null                                                                             | Path to the pre-trained language model checkpoint                                                            |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.language_model.config_file          | string          | null                                                                             | Path to the pre-trained language model config file                                                           |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.language_model.config               | dictionary      | null                                                                             | Config of the pre-trained language model                                                                     |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.head.num_output_layers              | integer         | 2                                                                                | Number of fully connected layers of the Classifier on top of Bert model                                      |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.head.fc_dropout                     | float           | 0.1                                                                              | Dropout ratio of the fully connected layers                                                                  |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| {training,validation,test}_ds.file_path   | string          | ??                                                                               | Path of the training '.tsv file                                                                              |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| {training,validation,test}_ds.batch_size  | integer         | 32                                                                               | Data loader's batch size                                                                                     |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| {training,validation,test}_ds.num_workers | integer         | 2                                                                                | Number of worker threads for data loader                                                                     |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| {training,validation,test}_ds.shuffle     | boolean         | true (training), false (test and validation)                                     | Shuffles data for each epoch                                                                                 |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| {training,validation,test}_ds.drop_last   | boolean         | false                                                                            | Specifies if last batch of data needs to get dropped if it is smaller than batch size                        |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| {training,validation,test}_ds.pin_memory  | boolean         | false                                                                            | Enables pin_memory of PyTorch's data loader to enhance speed                                                 |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| {training,validation,test}_ds.num_samples | integer         | -1                                                                               | Number of samples to be used from the dataset; -1 means all samples                                          |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| optim.name                                | string          | adam                                                                             | Optimizer to use for training                                                                                |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| optim.lr                                  | float           | 2e-5                                                                             | Learning rate to use for training                                                                            |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| optim.weight_decay                        | float           | 0.01                                                                             | Weight decay to use for training                                                                             |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| optim.sched.name                          | string          | WarmupAnnealing                                                                  | Warmup schedule                                                                                              |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| optim.sched.warmup_ratio                  | float           | 0.1                                                                              | Warmup ratio                                                                                                 |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+


Training Procedure
^^^^^^^^^^^^^^^^^^

At the start of each training experiment, TLT will print out a log of the experiment specification,
including any parameters added or overridden via the command line.
It will also show additional information, such as which GPUs are available and where logs will be
saved. Then it shows some samples from the datasets with their corresponding inputs to the model.

.. code::

    GPU available: True, used: True
    TPU available: None, using: 0 TPU cores
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
    [NeMo W 2021-01-20 19:49:30 exp_manager:304] There was no checkpoint folder at checkpoint_dir :/home/user/tlt-pytorch/nlp/text_classification/entrypoint/nemo_experiments/trained-model/2021-01-20_19-49-30/checkpoints. Training from scratch.
    [NeMo I 2021-01-20 19:49:30 exp_manager:194] Experiments will be logged at /home/user/tlt-pytorch/nlp/text_classification/entrypoint/nemo_experiments/trained-model/2021-01-20_19-49-30

Then for each dataset provided, it shows some samples from the dataset with their corresponding inputs to the model.
It also provides some stats on the lengths of sequences in the dataset.

.. code::

    [NeMo I 2021-01-20 19:49:36 text_classification_dataset:120] Read 67350 examples from ../data/SST-2/train.tsv.
    [NeMo I 2021-01-20 19:49:37 text_classification_dataset:233] *** Example ***
    [NeMo I 2021-01-20 19:49:37 text_classification_dataset:234] example 0: ['girl-meets-girl', 'romantic', 'comedy']
    [NeMo I 2021-01-20 19:49:37 text_classification_dataset:235] subtokens: [CLS] girl - meets - girl romantic comedy [SEP]
    [NeMo I 2021-01-20 19:49:37 text_classification_dataset:236] input_ids: 101 2611 1011 6010 1011 2611 6298 4038 102
    [NeMo I 2021-01-20 19:49:37 text_classification_dataset:237] segment_ids: 0 0 0 0 0 0 0 0 0
    [NeMo I 2021-01-20 19:49:37 text_classification_dataset:238] input_mask: 1 1 1 1 1 1 1 1 1
    [NeMo I 2021-01-20 19:49:37 text_classification_dataset:239] label: 1

Before training starts, information on the optimizer and scheduler will be shown in the logs:

.. code::

    [NeMo I 2021-01-20 19:50:19 modelPT:830] Optimizer config = Adam (
        Parameter Group 0
            amsgrad: False
            betas: [0.9, 0.999]
            eps: 1e-08
            lr: 2e-05
            weight_decay: 0.01
        )
    [NeMo I 2021-01-20 19:50:19 lr_scheduler:621] Scheduler "<nemo.core.optim.lr_scheduler.WarmupAnnealing object at 0x7fcd2232b160>"
        will be used during training (effective maximum steps = 1053) -
        Parameters :
        (warmup_steps: null
        warmup_ratio: 0.1
        last_epoch: -1
        max_steps: 1053
        )


You should next see a full printout of the number of parameters in each module and submodule,
as well as the total number of trainable and non-trainable parameters in the model.
For example, this model has 100M parameters in total:

.. code::

        | Name                                                   | Type                 | Params
    --------------------------------------------------------------------------------------------------
    0   | bert_model                                             | BertEncoder          | 109 M
    1   | bert_model.embeddings                                  | BertEmbeddings       | 23.8 M
    2   | bert_model.embeddings.word_embeddings                  | Embedding            | 23.4 M
    3   | bert_model.embeddings.position_embeddings              | Embedding            | 393 K
    4   | bert_model.embeddings.token_type_embeddings            | Embedding            | 1.5 K
    5   | bert_model.embeddings.LayerNorm                        | LayerNorm            | 1.5 K
    6   | bert_model.embeddings.dropout                          | Dropout              | 0
    7   | bert_model.encoder                                     | BertEncoder          | 85.1 M
    8   | bert_model.encoder.layer                               | ModuleList           | 85.1 M
    9   | bert_model.encoder.layer.0                             | BertLayer            | 7.1 M
    10  | bert_model.encoder.layer.0.attention                   | BertAttention        | 2.4 M
    11  | bert_model.encoder.layer.0.attention.self              | BertSelfAttention    | 1.8 M
    12  | bert_model.encoder.layer.0.attention.self.query        | Linear               | 590 K
    ...
    212 | bert_model.encoder.layer.11.output.dropout             | Dropout              | 0
    213 | bert_model.pooler                                      | BertPooler           | 590 K
    214 | bert_model.pooler.dense                                | Linear               | 590 K
    215 | bert_model.pooler.activation                           | Tanh                 | 0
    216 | classifier                                             | SequenceClassifier   | 592 K
    217 | classifier.dropout                                     | Dropout              | 0
    218 | classifier.mlp                                         | MultiLayerPerceptron | 592 K
    219 | classifier.mlp.layer0                                  | Linear               | 590 K
    220 | classifier.mlp.layer2                                  | Linear               | 1.5 K
    221 | loss                                                   | CrossEntropyLoss     | 0
    222 | classification_report                                  | ClassificationReport | 0
    --------------------------------------------------------------------------------------------------
    110 M     Trainable params
    0         Non-trainable params
    110 M     Total params

As the model starts training, you should see a progress bar per epoch.

.. code::

    Epoch 0: 100%|████████████████████████████| 1067/1067 [03:10<00:00,  5.60it/s, loss=0.252, val_loss=0.258, Epoch 0, global step 1052: val_loss reached 0.25792 (best 0.25792), saving model to "/home/user/tlt-pytorch/nlp/text_classification/entrypoint/nemo_experiments/trained-model/2021-01-20_20-19-44/checkpoints/trained-model---val_loss=0.26-epoch=0.ckpt" as top 3
    Epoch 1: 100%|████████████████████████████| 1067/1067 [03:10<00:00,  5.60it/s, loss=0.187, val_loss=0.245, Epoch 1, global step 2105: val_loss reached 0.24499 (best 0.24499), saving model to "/home/user/tlt-pytorch/nlp/text_classification/entrypoint/nemo_experiments/trained-model/2021-01-20_20-19-44/checkpoints/trained-model---val_loss=0.24-epoch=1.ckpt" as top 3
    Epoch 2: 100%|████████████████████████████| 1067/1067 [03:09<00:00,  5.62it/s, loss=0.158, val_loss=0.235, Epoch 2, global step 3158: val_loss reached 0.23505 (best 0.23505), saving model to "/home/user/tlt-pytorch/nlp/text_classification/entrypoint/nemo_experiments/trained-model/2021-01-20_20-19-44/checkpoints/trained-model---val_loss=0.24-epoch=2.ckpt" as top 3
    ...

After each epoch, you should see a summary table of metrics on the validation set.

.. code::

    Validating:  100%|████████████████████████████| 14/14 [00:00<00:00, 13.94it/s]
    [NeMo I 2021-01-20 19:53:32 text_classification_model:173] val_report:
        label                                                precision    recall       f1           support
        label_id: 0                                             91.97      88.32      90.11        428
        label_id: 1                                             89.15      92.57      90.83        444
        -------------------
        micro avg                                               90.48      90.48      90.48        872
        macro avg                                               90.56      90.44      90.47        872
        weighted avg                                            90.54      90.48      90.47        872

At the end of training, TLT will save the last checkpoint at the path specified by the experiment
spec file before finishing.

.. code::

    Saving latest checkpoint...
    [NeMo I 2021-01-20 21:09:39 train:124] Experiment logs saved to '/home/user/tlt-pytorch/nlp/text_classification/entrypoint/nemo_experiments/trained-model/2021-01-20_21-06-17'
    [NeMo I 2021-01-20 21:09:39 train:127] Trained model saved to '/home/user/tlt-pytorch/nlp/text_classification/entrypoint/nemo_experiments/trained-model/2021-01-20_21-06-17/checkpoints/trained-model.tlt'

The output logs for the evaluation and fine-tuning look similar.

Training Suggestions
--------------------
When you want to train this model on other data or with different batch sizes, you may need to tune at least the configs of your optimizer and
scheduler like the learning rate and weight decay. Higher effective batch sizes need larger learning rate.
Effective batch size is the total number of your samples per each update step.
For example, when your batch size per GPU is set to 64, and you use four GPUs with accumulate_grad_batches of two, then your effective batch size would be 512=64*4*2.
You may use other BERT-like models or models with different sizes based on your performance requirements.

Model Fine-tuning
-----------------

There are scenarios where users are required to re-train or fine-tune a pretrained TLT model like `trained-model.tlt` on a new dataset. \
TLT toolkit provides a separate tool called `fine-tune` to enable this.

Example of spec file to be used for fine-tuning of a model:

.. code::

    trainer:
      max_epochs: 100
    data_dir: ???

    # Fine-tuning settings: training dataset.
    finetuning_ds:
      file_path: ???
      batch_size: 64
      shuffle: false
      num_samples: -1 # number of samples to be considered, -1 means all the dataset
      num_workers: 3
      drop_last: false
      pin_memory: false

    # Fine-tuning settings: validation dataset.
    validation_ds:
      file_path: ???
      batch_size: 64
      shuffle: false
      num_samples: -1 # number of samples to be considered, -1 means all the dataset
      num_workers: 3
      drop_last: false
      pin_memory: false

    # Fine-tuning settings: different optimizer.
    optim:
      name: adam
      lr: 2e-5
      betas: [0.9, 0.9998]
      weight_decay: 0.001

Use the following command to fine-tune a pre-trained model on a training file specified by 'finetuning_ds.file_path':

.. code::

    tlt text_classification finetune [-h]  -e /specs/nlp/text_classification/finetune.yaml \
                                                      -r PATH_TO_RESULT_FOLDER \
                                                      -m PATH_OF_PRETRAINED_TLT_MODEL \
                                                      -g 1 \
                                                      finetuning_ds.file_path=PATH_TO_TRAIN_FILE \
                                                      trainer.max_epochs=3 \
                                                      -k $KEY

Required Arguments for Fine-tuning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`-e`: The experiment specification file to set up fine-tuning
* :code:`-r`: Path to the directory to store the results of the fine-tuning.
* :code:`-m`: Path to the pre-trained model to use for fine-tuning.
* :code:`training_ds.file_path`: Path to the training '.tsv' file
* :code:`-k`: Encryption key

Optional Arguments
^^^^^^^^^^^^^^^^^^

* :code:`-h, --help`: Show this help message and exit
* :code:`-g`: The number of GPUs to be used in evaluation in a multi-GPU scenario (default: 1).
* Other arguments to override fields in the specification file.


Model Evaluation
----------------

The evaluation tool enables the user to evaluate a saved model in TLT format on a dataset.

Spec example to evaluate the pre-trained model on test data:

.. code::

    restore_from: trained-model.tlt

    test_ds:
      file_path: PATH_TO_TEST_FILE
      num_workers: 2
      batch_size: 32
      shuffle: false
      num_samples: -1

Use the following command to evaluate the model:

.. code::

    tlt text_classification evaluate \
    -e /specs/nlp/text_classification/evaluate.yaml \
    test_ds.file_path=PATH_TO_TEST_FILE \

Required Arguments for Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`-e`: The experiment specification file to set up evaluation.
* :code:`test_ds.file_path`: Path to the test '.tsv' file

The output should be similar to the training process and the metrics table is shown at the end:

.. code::

    Validating:  100%|████████████████████████████| 14/14 [00:00<00:00, 13.94it/s]
    [NeMo I 2021-01-20 19:53:32 text_classification_model:173] val_report:
        label                                                precision    recall       f1           support
        label_id: 0                                             91.97      88.32      90.11        428
        label_id: 1                                             89.15      92.57      90.83        444
        -------------------
        micro avg                                               90.48      90.48      90.48        872
        macro avg                                               90.56      90.44      90.47        872
        weighted avg                                            90.54      90.48      90.47        872

This table contains the metrics for each class separately, like precision, recall, F1, and support.
It also shows Micro Average, Macro Average, and Weighted Average, which may show the overall performance of the model on all classes.


Model Inference
----------------

Inference tool would take some inputs in text format and produces the predictions of a saved model for them.
To run inference on the model, specify the list of examples in the spec file "infer.yaml", for example:

.. code::

    input_batch:
  - "by the end of no such thing the audience , like beatrice , has a watchful affection for the monster ."
  - "director rob marshall went out gunning to make a great one ."
  - "uneasy mishmash of styles and genres ."
  - "I love exotic science fiction / fantasy movies but this one was very unpleasant to watch . Suggestions and images of child abuse , mutilated bodies (live or dead) , other gruesome scenes , plot holes , boring acting made this a regretable experience , The basic idea of entering another person's mind is not even new to the movies or TV (An Outer Limits episode was better at exploring this idea) . i gave it 4 / 10 since some special effects were nice ."

The list of inputs specified by 'input_batch' would be passed through the model to get the label predictions.

To run the inference on a trained model 'trained-model.tlt':

.. code::

    tlt text_classification infer \
    -e /specs/nlp/text_classification/infer.yaml \
    -m trained-model.tlt \

Required Arguments for Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`-e`: The experiment specification file to set up inference.
  This requires the :code:`input_batch` with the list of examples to run inference on.
* :code:`-m`: Path to the pre-trained model checkpoint from which to infer. Should be a :code:`.tlt` file.

The output should be like this:

.. code::

    [NeMo I 2021-01-27 16:58:32 infer:68] Spec file:
        restore_from: trained-model.tlt
        exp_manager:
          task_name: infer
          explicit_log_dir: ./
        input_batch:
        - by the end of no such thing the audience , like beatrice , has a watchful affection
          for the monster .
        - director rob marshall went out gunning to make a great one .
        - uneasy mishmash of styles and genres .
        - I love exotic science fiction / fantasy movies but this one was very unpleasant
          to watch . Suggestions and images of child abuse , mutilated bodies (live or dead)
          , other gruesome scenes , plot holes , boring acting made this a regretable experience
          , The basic idea of entering another person's mind is not even new to the movies
          or TV (An Outer Limits episode was better at exploring this idea) . i gave it 4
          / 10 since some special effects were nice .
        encryption_key: null
    ...
    [NeMo I 2021-01-27 16:58:50 infer:101] Query: by the end of no such thing the audience , like beatrice , has a watchful affection for the monster .
    [NeMo I 2021-01-27 16:58:50 infer:102] Predicted label: positive
    [NeMo I 2021-01-27 16:58:50 infer:101] Query: director rob marshall went out gunning to make a great one .
    [NeMo I 2021-01-27 16:58:50 infer:102] Predicted label: positive
    [NeMo I 2021-01-27 16:58:50 infer:101] Query: uneasy mishmash of styles and genres .
    [NeMo I 2021-01-27 16:58:50 infer:102] Predicted label: negative
    [NeMo I 2021-01-27 16:58:50 infer:101] Query: I love exotic science fiction / fantasy movies but this one was very unpleasant to watch . Suggestions and images of child abuse , mutilated bodies (live or dead) , other gruesome scenes , plot holes , boring acting made this a regretable experience , The basic idea of entering another person's mind is not even new to the movies or TV (An Outer Limits episode was better at exploring this idea) . i gave it 4 / 10 since some special effects were nice .
    [NeMo I 2021-01-27 16:58:50 infer:102] Predicted label: negative

Each query would be printed out along with its predicted label.

Model Export
------------

You may use the export toolkit to convert a pre-trained saved TLT model into Jarvis format. This format would enable faster inference.
An example of the spec file for model export:

.. code::

    # Name of the .tlt EFF archive to be loaded/model to be exported.
    restore_from: trained-model.tlt

    # Set export format to JARVIS
    export_format: JARVIS

    # Output EFF archive containing Jarvis file.
    export_to: exported-model.ejrvs

+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **Parameter**                             | **Data Type**   |   **Default**                                                                    | **Description**                                                                                              |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| restore_from                              | string          | trained-model.tlt                                                                | Path to the pre-trained model                                                                                |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| export_format                             | string          | ONNX                                                                             | Export format, choose from: ONNX  or JARVIS                                                                  |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| export_to                                 | string          | exported-model.eonnx                                                             | Path to the exported model                                                                                   |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+

To export a pre-trained model to JARVIS's format, run:

.. code::

    tlt text_classification export \
        -e /specs/nlp/text_classification/export.yaml \
        -m finetuned-model.tlt \
        -k $KEY \
        export_format=JARVIS \
        export_to=exported-model.ejrvs

Required Arguments for Export
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`-e`: The experiment specification file to set up inference.
  This requires the :code:`input_batch` with the list of examples to run inference on.
* :code:`-m`: Path to the pre-trained model checkpoint from which to infer. Should be a :code:`.tlt` file.
* :code:`-k`: Encryption key

The output should be something like this:

.. code::

    [NeMo I 2021-01-27 17:04:05 export:37] Spec file:
        restore_from: ./trained-model.tlt
        export_to: exported-model.ejrvs
        export_format: JARVIS
        exp_manager:
          task_name: export
          explicit_log_dir: ./
        encryption_key: null
    ...
    [NeMo W 2021-02-01 16:00:22 exp_manager:27] Exp_manager is logging to `./``, but it already exists.
    [NeMo W 2021-02-01 16:00:28 modelPT:193] Using /tmp/tmpmke24h_1/tokenizer.vocab_file instead of tokenizer.vocab_file.
    [NeMo W 2021-02-01 16:00:35 modelPT:193] Using /tmp/tmpmke24h_1/label_ids.csv instead of ../data/SST-2/label_ids.csv.
    [NeMo I 2021-02-01 16:00:37 export:52] Model restored from '/home/user/tlt-pytorch/nlp/text_classification/entrypoint/nemo_experiments/trained-model/2021-01-27_16-53-38/checkpoints/trained-model.tlt'
    [NeMo I 2021-02-01 16:01:08 export:66] Experiment logs saved to '.'
    [NeMo I 2021-02-01 16:01:08 export:67] Exported model to './exported-model.ejrvs'

Automatic Speech Recognition (ASR) systems typically generate text with no punctuation and capitalization of the words. \
There are two issues with non-punctuated ASR output:

- it could be difficult to read and understand;
- models for some downstream tasks such as named entity recognition, machine translation or text-to-speech are usually trained on punctuated datasets and using raw ASR output as the input to these models could deteriorate their performance.

Model Description
-----------------

For each word in the input text, the Punctuation and Capitalization model:

1. predicts a punctuation mark that should follow the word (if any). By default, the model supports commas, periods and question marks.
2. predicts if the word should be capitalized or not.

.. note::

    We recommend you try this model in a Jupyter notebook \
    (can run on `Google's Colab <https://colab.research.google.com/notebooks/intro.ipynb>`_.): \
    `NeMo/tutorials/nlp/Punctuation_and_Capitalization.ipynb <https://github.com/NVIDIA/NeMo/blob/main/tutorials/nlp/Punctuation_and_Capitalization.ipynb>`__.

    Connect to an instance with a GPU (Runtime -> Change runtime type -> select "GPU" for hardware accelerator)

    An example script on how to train the model could be found here: `NeMo/examples/nlp/token_classification/punctuation_capitalization_train.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/punctuation_capitalization_train.py>`__.

    An example script on how to run evaluation and inference could be found here: `NeMo/examples/nlp/token_classification/punctuation_capitalization_evaluate.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/punctuation_capitalization_evaluate.py>`__.

    The default configuration file for the model could be found at: `NeMo/examples/nlp/token_classification/conf/punctuation_capitalization_config.yaml <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/conf/punctuation_capitalization_config.yaml>`__.



.. _raw_data_format_punct:

Raw Data Format
---------------

The Punctuation and Capitalization model can work with any text dataset, although it is recommended to balance the data, especially for the punctuation task.
Before pre-processing the data to the format expected by the model, the data should be split into train.txt and dev.txt (and optionally test.txt).
Each line in the **train.txt/dev.txt/test.txt** should represent one or more full and/or truncated sentences.

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
     |-- train.txt



NeMo Data Format for training the model
---------------------------------------

The punctuation and capitalization model expects the data in the following format:

The training and evaluation data is divided into 2 files: text.txt and labels.txt. \
Each line of the **text.txt** file contains text sequences, where words are separated with spaces, i.e.

[WORD] [SPACE] [WORD] [SPACE] [WORD], for example:

    ::

        when is the next flight to new york
        the next flight is ...
        ...

The **labels.txt** file contains corresponding labels for each word in text.txt, the labels are separated with spaces. \
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

To pre-process the raw text data, stored under :code:`sourced_data_dir` (see the :ref:`raw_data_format_punct`
section), run the following command:

.. code::

    python examples/nlp/token_classification/data/prepare_data_for_punctuation_capitalization.py \
           -s <PATH_TO_THE_SOURCE_FILE>
           -o <PATH_TO_THE_OUTPUT_DIRECTORY>


Convert Dataset Required Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`-s` or :code:`--source_file`: path to the raw file
* :code:`-o` or :code:`--output_dir` - path to the directory to store the converted files

After the conversion, the :code:`output_dir` should contain :code:`labels_*.txt` and :code:`text_*.txt` files.
The default names for the training and evaluation in the :code:`conf/punctuation_capitalization_config.yaml` are the following:

.. code::

   .
   |--output_dir
     |-- labels_dev.txt
     |-- labels_train.txt
     |-- text_dev.txt
     |-- text_train.txt

Training Punctuation and Capitalization Model
---------------------------------------------

In the Punctuation and Capitalization Model, we are jointly training two token-level classifiers on top of a pre-trained \
language model, such as `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`__ :cite:`nlp-punct-devlin2018bert`.
Unless the user provides a pre-trained checkpoint for the language model, the language model is initialized with the
pre-trained model from `HuggingFace Transformers <https://github.com/huggingface/transformers>`__.
Example of model configuration file for training the model could be found at: `NeMo/examples/nlp/token_classification/conf/punctuation_capitalization_config.yaml <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/conf/punctuation_capitalization_config.yaml>`__.

The specification can be roughly grouped into the following categories:

* Parameters that describe the training process: **trainer**
* Parameters that describe the datasets: **model.dataset**, **model.train_ds**, **model.validation_ds**
* Parameters that describe the model: **model**

More details about parameters in the config file could be found below and in the `model's config file <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/conf/punctuation_capitalization_config.yaml>`__:


+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **Parameter**                             | **Data Type**   |  **Description**                                                                                             |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| pretrained_model                          | string          | Path to the pre-trained model .nemo file or pre-trained model name                                           |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| model.dataset.data_dir                    | string          | Path to the data converted to the specified above format                                                     |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| model.tokenizer.tokenizer_name            | string          | Tokenizer name, will be filled automatically based on model.language_model.pretrained_model_name             |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| model.tokenizer.vocab_file                | string          | Path to tokenizer vocabulary                                                                                 |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| model.tokenizer.tokenizer_model           | string          | Path to tokenizer model (only for sentencepiece tokenizer)                                                   |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| model.language_model.pretrained_model_name| string          | Pre-trained language model name, for example: `bert-base-cased` or `bert-base-uncased`                       |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| model.language_model.lm_checkpoint        | string          | Path to the pre-trained language model checkpoint                                                            |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| model.language_model.config_file          | string          | Path to the pre-trained language model config file                                                           |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| model.language_model.config               | dictionary      | Config of the pre-trained language model                                                                     |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| model.punct_head.punct_num_fc_layers      | integer         | Number of fully connected layers                                                                             |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| model.punct_head.fc_dropout               | float           | Activation to use between fully connected layers                                                             |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| model.punct_head.activation               | string          | Dropout to apply to the input hidden states                                                                  |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| model.punct_head.use_transrormer_init     | bool            | Whether to initialize the weights of the classifier head with the same approach used in Transformer          |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| model.capit_head.punct_num_fc_layers      | integer         | Number of fully connected layers                                                                             |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| model.capit_head.fc_dropout               | float           | Dropout to apply to the input hidden states                                                                  |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| model.capit_head.activation               | string          | Activation function to use between fully connected layers                                                    |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| model.capit_head.use_transrormer_init     | bool            | Whether to initialize the weights of the classifier head with the same approach used in Transformer          |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| training_ds.text_file                     | string          | Name of the text training file located at `data_dir`                                                         |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| training_ds.labels_file                   | string          | Name of the labels training file located at `data_dir`, such as `labels_train.txt`                           |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| training_ds.num_samples                   | integer         | Number of samples to use from the training dataset, -1 - to use all                                          |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| validation_ds.text_file                   | string          | Name of the text file for evaluation, located at `data_dir`                                                  |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| validation_ds.labels_file                 | string          | Name of the labels dev file located at `data_dir`, such as `labels_dev.txt`                                  |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| validation_ds.num_samples                 | integer         | Number of samples to use from the dev set, -1 mean all                                                       |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+

To train the model from scratch, run:

.. code::

      python examples/nlp/token_classification/punctuation_and_capitalization_train.py \
             model.dataset.data_dir=<PATH/TO/DATA_DIR> \
             trainer.gpus=[0,1] \
             optim.name=adam \
             optim.lr=0.0001 \
             model.nemo_path=<PATH/TO/SAVE/.nemo>

To train from the pre-trained model, use:

.. code::

      python examples/nlp/token_classification/punctuation_and_capitalization_train.py \
             model.dataset.data_dir=<PATH/TO/DATA_DIR> \
             pretrained_model=<PATH/TO/SAVE/.nemo>


Required Arguments for Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`model.dataset.data_dir`: Path to the `data_dir` with the pre-processed data files.

Optional Arguments
^^^^^^^^^^^^^^^^^^
* :code:`pretrained_model`: pretrained PunctuationCapitalization model from list_available_models() or path to a .nemo file, for example: punctuation_en_bert or your_model.nemo
* :code:`--config-name`: Path to the config file to use. The default config file for the model is `/examples/nlp/token_classification/conf/punctuation_capitalization_config.yaml`. You may update the config file from the file directly. The other option is to set another config file via command line arguments by :code:`--config-name=<CONFIG/FILE/PATH>`. For more details about the config files and different ways of model restoration, see tutorials/00_NeMo_Primer.ipynb
* Other arguments to override fields in the specification file, please see the note below.

.. note::

    All parameters defined in the configuration file could be changed with command arguments. \
    For example, the sample config file mentioned above has :code:`validation_ds.batch_size` set to 64. \
    However, if you see that the GPU utilization can be optimized further by using a larger batch size, \
    you may override to the desired value, by adding the field :code:`validation_ds.batch_size=128` over the command line.
    You may repeat this with any of the parameters defined in the sample configuration file.



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


Inference
---------

An example script on how to run inference on a few examples, could be found
at `examples/nlp/token_classification/punctuation_capitalization_evaluate.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/punctuation_capitalization_evaluate.py>`_.

To run inference with the pre-trained model on a few examples, run:

.. code::

    python punctuation_capitalization_evaluate.py \
           pretrained_model=<PRETRAINED_MODEL>

Required Arguments for inference:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`pretrained_model`: pretrained PunctuationCapitalization model from list_available_models() or path to a .nemo file, for example: punctuation_en_bert or your_model.nemo


Model Evaluation
----------------

An example script on how to evaluate the pre-trained model, could be found
at `examples/nlp/token_classification/punctuation_capitalization_evaluate.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/punctuation_capitalization_evaluate.py>`_.

To run evaluation of the pre-trained model, run:

.. code::

    python punctuation_capitalization_evaluate.py \
           model.dataset.data_dir=<PATH/TO/DATA/DIR>  \
           pretrained_model=punctuation_en_bert \
           model.test_ds.text_file=<text_dev.txt> \
           model.test_ds.labels_file=<labels_dev.txt>


Required Arguments:
^^^^^^^^^^^^^^^^^^^
* :code:`pretrained_model`: pretrained PunctuationCapitalization model from list_available_models() or path to a .nemo file, for example: punctuation_en_bert or your_model.nemo
* :code:`model.dataset.data_dir`: Path to the directory that containes :code:`model.test_ds.text_file` and :code:`model.test_ds.labels_file`.


Optional Arguments:
^^^^^^^^^^^^^^^^^^^
* :code:`model.test_ds.text_file` and :code:`model.test_ds.labels_file`: text_*.txt and labels_*.txt file names is the default text_dev.txt and labels_dev.txt from the config files should be overwritten.
* Other :code:`model.dataset` or :code:`model.test_ds` arguments to override fields in the config file of the pre-trained model.


During evaluation of the :code:`test_ds`, the script generates two classification reports: one for capitalization task and \
another one for punctuation task. This classification reports include the following metrics:

* :code:`Precision`
* :code:`Recall`
* :code:`F1`

More details about these metrics could be found `here <https://en.wikipedia.org/wiki/Precision_and_recall>`__.

References
----------

.. bibliography:: nlp_all.bib
    :style: plain
    :labelprefix: NLP-PUNCT
    :keyprefix: nlp-punct-
