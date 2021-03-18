Checkpoints
===========

There are two main ways to load pretrained checkpoints in NeMo as introduced in `loading ASR checkpoints <../results.html#checkpoints>`__.
In speaker diarization, the diarizer would take care of model loading, you can load checkpoints by passing it thourgh config file. For example:

Loading Local Checkpoints
---------------------------

Load VAD model

.. code-block:: bash

  pretrained_vad_model='/path/to/vad_marblenet.nemo' # local .nemo or pretrained vad model name
  ...
  # pass with hydra config
  config.diarizer.vad.model_path=pretrained_vad_model


Load Speaker Embedding model

.. code-block:: bash

  pretrained_speaker_model='/path/to/speakerdiarization_speakernet.nemo' # local .nemo or pretrained speakernet model name
  ...
  # pass with hydra config
  config.diarizer.speaker_embeddings.model_path=pretrained_speaker_model



NeMo will automatically save checkpoints of a model you are training in a `.nemo` format.
You can also manually save your models at any point using :code:`model.save_to(<checkpoint_path>.nemo)`.


NGC Pretrained Checkpoints
----------------------------

The ASR collection has checkpoints of several models trained on various datasets for a variety of tasks.
These checkpoints are obtainable via NGC `NeMo Automatic Speech Recognition collection <https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels>`_.
The model cards on NGC contain more information about each of the checkpoints available.

In general, you can load models with model name in the following format, 

.. code-block:: python

  pretrained_vad_model='vad_telephony_marblenet' 
  pretrained_speaker_model='speakerdiarization_speakernet' 
  ...
  config.diarizer.vad.model_path=retrained_vad_model \
  config.diarizer.speaker_embeddings.model_path=pretrained_speaker_model

where the model name is the value under "Model Name" entry in the tables below.

Models for Speaker Diarization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :file: data/diarization_results.csv
   :align: left
   :widths: 30, 30, 40
   :header-rows: 1
