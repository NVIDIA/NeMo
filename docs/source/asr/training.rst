Training & finetuning models
============================

Many training and finetuning scripts are provided in the NeMo `examples <https://github.com/NVIDIA/NeMo/tree/main/examples>`_.

On this page we will explain how those scripts work, and how you can modify them to suit your intended task.

We recommend first reading the :doc:`NeMo basics <../starthere/basics>` section which explains how a typical NeMo ASR training script works.

Changing the ASR Model
----------------------

To adjust the architecture of the ASR model you use for training/fine-tuning, you will need to:

* change the default config file (by pointing the ``--config_path`` and ``--config_name`` to the `config file <https://github.com/NVIDIA/NeMo/tree/main/examples/asr/conf>`_ of the architecture you are interested in), and

* you may need to adjust the ASR model class.

* If finetuning, make sure the ``init_from_nemo_model`` parameter points to the model you want to start with.

.. figure:: ../../../../new_docs_diagrams/change_asr_model.png
   :alt: how to change asr model 

   How to change the ASR model used for training/fine-tuning.


Changing the ASR Model's output tokens
--------------------------------------

You may want to change the output tokens used by the ASR model, for example if you want to train the model in a different language, or simply want to use a different set of tokens.

To change the tokens, you will need to adjust the config of the model, and potentially the ASR model class. Your changes will depend on the on whether you want to use character-based tokens or BPE tokens. The specifics of the required changes are shown in the diagram below.

.. figure:: ../../../../new_docs_diagrams/change_tokens.png
   :alt: how to change output tokens 

   How to change the output tokens used by the ASR model. 

If you need to create a new BPE tokenizer, use `this script <https://github.com/NVIDIA/NeMo/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py>`_.

Adjusting hyperparameters
-------------------------

You may want to adjust other "hyperparameters" such as the maximum number of steps made during training, or learning rate. In most cases, you can simply adjust the parameter of interest, either by changing the value in the config file that you use, or (recommended) by overriding the parameter when you call the training/fine-tuning script.

Unless you are confident with your changes, we recommend first trying the default optimization-related hyperparameters provided in the `config files <https://github.com/NVIDIA/NeMo/tree/main/examples/asr/conf>`_ of the ASR models, as they have been tuned to work well for the given model.

Change the training data
------------------------

You will need to modify the config so that it points to the training data that you want to use (as well as some validation data, and (optionally) test data).

NeMo expects training data to be inside a JSON :ref:`"manifest file" <section-with-manifest-format-explanation>`, in which each line is formatted like below:

.. code-block:: json

  {"audio_filepath": "/path/to/audio.wav", "text": "the transcription of the utterance", "duration": 23.147}

You should make sure that your training and validation data is present in this format, then point the config to the manifest files, for example by overriding like below:

.. code-block:: bash

   python NeMo/examples/asr/asr_transducer/speech_to_text_rnnt.py \
      model.train_ds.manifest_filepath="/path/to/train_manifest.json" \
      model.validation_ds.manifest_filepath="/path/to/train_manifest.json" \
      ...


If running training/fine-tuning on a large cluster, you can optimize your training run by using a :ref:`"tarred dataset" <section-with-tarred-dataset-explanation>`.

Further resources
-----------------

Learn more about :doc:`NeMo ASR Configuration files <./configs>`.
