NeMo ASR Configuration Files
============================

This page covers NeMo configuration file setup that is specific to models in the ASR collection.
For general information about how to set up and run experiments that is common to all NeMo models (e.g.
experiment manager and PyTorch Lightning trainer parameters), see the :doc:`../core` page.

The model section of NeMo ASR configuration files will generally require information about the dataset(s) being
used, the preprocessor for audio files, parameters for any augmentation being performed, as well as the
model architecture specification.
The sections on this page cover each of these in more detail.

Example configuration files for all of the NeMo ASR scripts can be found at
`example <https://github.com/NVIDIA/NeMo/tree/r1.0.0rc1/examples/asr/conf>`_.

Dataset Configuration
---------------------

Training, validation, and test parameters are specified using the ``train_ds``, ``validation_ds``, and
``test_ds`` sections of your configuration file, respectively.
Depending on the task, you may have arguments specifying the sample rate of your audio files, the vocabulary
of your dataset (for character prediction), whether or not to shuffle the dataset, and so on.
You may also decide to leave fields such as the ``manifest_filepath`` blank, to be specified via the command line
at runtime.

Any initialization parameters that are accepted for the Dataset class used in your experiment
in the config file.
See the :ref:`api:Datasets` section of the API for a list of Datasets and their respective parameters.

An example ASR train and validation configuration could look like:

.. code-block:: yaml

  # Specified at the beginning of the config file
  labels: &labels [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
           "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]

  model:
    train_ds:
      manifest_filepath: ???
      sample_rate: 16000
      labels: *labels   # Uses the labels above
      batch_size: 32
      trim_silence: True
      max_duration: 16.7
      shuffle: True
      is_tarred: False  # If set to true, uses the tarred version of the Dataset
      tarred_audio_filepaths: null      # Not used if is_tarred is false
      tarred_shard_strategy: "scatter"  # Not used if is_tarred is false

    validation_ds:
      manifest_filepath: ???
      sample_rate: 16000
      labels: *labels   # Uses the labels above
      batch_size: 32
      shuffle: False    # No need to shuffle the validation data


Preprocessor Configuration
--------------------------


Augmentation Configurations
---------------------------


Model Architecture Configurations
---------------------------------

Jasper and QuartzNet
~~~~~~~~~~~~~~~~~~~~
