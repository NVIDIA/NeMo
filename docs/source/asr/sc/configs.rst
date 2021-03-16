NeMo Speech Classification Configuration Files
============================

This page covers NeMo configuration file setup that is specific to models in the Speech Classification collection.
For general information about how to set up and run experiments that is common to all NeMo models (e.g.
experiment manager and PyTorch Lightning trainer parameters), see the :doc:`../../introduction/core` page.

The model section of NeMo Speech Classification configuration files will generally require information about the dataset(s) being
used, the preprocessor for audio files, parameters for any augmentation being performed, as well as the
model architecture specification.
The sections on this page cover each of these in more detail.

Example configuration files for all of the NeMo ASR scripts can be found in the
`config directory of the examples <https://github.com/NVIDIA/NeMo/tree/r1.0.0rc1/examples/asr/conf>`_.


Dataset Configuration
---------------------

Training, validation, and test parameters are specified using the ``train_ds``, ``validation_ds``, and
``test_ds`` sections of your configuration file, respectively.
Depending on the task, you may have arguments specifying the sample rate of your audio files, the vocabulary
of your dataset (for character prediction), whether or not to shuffle the dataset, and so on.
You may also decide to leave fields such as the ``manifest_filepath`` blank, to be specified via the command line
at runtime.

Any initialization parameters that are accepted for the Dataset class used in your experiment
can be set in the config file.
See the `Datasets <../api.html#Datasets>`__ section of the API for a list of Datasets and their respective parameters.

An example Speech Classiciation train and validation configuration could look like:

.. code-block:: yaml

  model:
    sample_rate: 16000
    repeat: 2
    dropout: 0.0
    kernel_size_factor: 1.0
    labels: ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin',
    'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up',
    'wow', 'yes', 'zero']

    train_ds:
      manifest_filepath: ???
      sample_rate: ${model.sample_rate}
      labels: ${model.labels} # Uses the labels above
      batch_size: 128
      shuffle: True
      is_tarred: False  # If set to true, uses the tarred version of the Dataset
      tarred_audio_filepaths: null      # Not used if is_tarred is false
      tarred_shard_strategy: "scatter"  # Not used if is_tarred is false

    validation_ds:
      manifest_filepath: ???
      sample_rate: ${model.sample_rate}
      labels: ${model.labels} # Uses the labels above
      batch_size: 128
      shuffle: False
      val_loss_idx: # No need to shuffle the validation data


Preprocessor Configuration
--------------------------

If you are loading audio files for your experiment, you will likely want to use a preprocessor to convert from the
raw audio signal to features (e.g. mel-spectrogram or MFCC).
The ``preprocessor`` section of the config specifies the audio preprocessor to be used via the ``_target_`` field,
as well as any initialization parameters for that preprocessor.

An example of specifying a preprocessor is as follows:

.. code-block:: yaml

  model:
    ...
    preprocessor:
      _target_: nemo.collections.asr.modules.AudioToMFCCPreprocessor
      window_size: 0.025
      window_stride: 0.01
      window: "hann"
      n_mels: &n_mels 64
      n_mfcc: *n_mels
      n_fft: 512

See the `Audio Preprocessors <../api.html#Audio Preprocessors>`__ API page for the preprocessor options, expected arguments, and defaults.


Augmentation Configurations
---------------------------

There are a few on-the-fly spectrogram augmentation options for NeMo ASR, which can be specified by the
configuration file using a ``spec_augment`` section.

For example, there are options for `Cutout <https://arxiv.org/abs/1708.04552>`_ and
`SpecAugment <https://arxiv.org/abs/1904.08779>`_ available via the ``SpectrogramAugmentation`` module.

The following example sets up both Cutout (via the ``rect_*`` parameters) and SpecAugment (via the ``freq_*``
and ``time_*`` parameters).

.. code-block:: yaml

  model:
    ...
    spec_augment:
      _target_: nemo.collections.asr.modules.SpectrogramAugmentation
      # Cutout parameters
      rect_masks: 5   # Number of rectangles to cut from any given spectrogram
      rect_freq: 15   # Max cut of size 50 along the frequency dimension
      rect_time: 25  # Max cut of size 120 along the time dimension
      # SpecAugment parameters
      freq_masks: 2   # Cut two frequency bands
      freq_width: 15  # ... of width 15 at maximum
      time_masks: 2    # Cut out 10 time bands
      time_width: 25  # ... of width 25 at maximum

You can use any combination of Cutout, frequency/time SpecAugment, or none of them.

See the `Audio Augmentors <../api.html#Audio Augmentors>`__ API section for more details.


Model Architecture Configurations
---------------------------------

Each configuration file should describe the model architecture being used for the experiment.
Models in the NeMo ASR collection need a ``encoder`` section and a ``decoder`` section, with the ``_target_`` field
specifying the module to use for each.

The following sections go into more detail about the specific configurations of each model architecture.

The `MatchboxNet <./models.html#MatchboxNet (Commands)>`__ and `MarbleNet <./models.html#MarbleNet (VAD)>`__ models are very similar, and they are based on `QuartzNet <../models.html#QuartzNet)>`__  and as such the components in their
configs are very similar as well.

Decoder Configurations
------------------------

After features have been computed from ConvASREncoder, we pass the features to decoder to compute embeddings and then to compute log_probs 
for training models.

.. code-block:: yaml

  model:
    ...
    decoder:
      _target_: nemo.collections.asr.modules.ConvASRDecoderClassification
      feat_in: *enc_final_filters
      return_logits: true
      pooling_type: 'avg'