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

If you are loading audio files for your experiment, you will likely want to use a preprocessor to convert from the
raw audio signal to features (e.g. mel-spectrogram or MFCC).
The ``preprocessor`` section of the config specifies the audio preprocessor to be used via the ``_target_`` field,
as well as any initialization parameters for that preprocessor.

An example of specifying a preprocessor is as follows:

.. code-block:: yaml

  model:
    ...
    preprocessor:
      # _target_ is the audio preprocessor module you want to use
      _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
      normalize: "per_feature"
      window_size: 0.02
      ...
      # Other parameters for the preprocessor

See the :ref:`api:Audio Preprocessors` API page for the preprocessor options, expected arguments, and defaults.


Augmentation Configurations
---------------------------

There are a few on-the-fly spectrogram augmentation options for NeMo ASR, which can be specified by the
configuration file using a ``spec_augment`` section.

The following example is the configuration for `SpecCutout <https://arxiv.org/abs/1708.04552>`_, using the
``SpectrogramAugmentation`` module.

.. code-block:: yaml

  model:
    ...
    spec_augment:
      # _target_ is the augmentor module you want to use
      _target_: nemo.collections.asr.modules.SpectrogramAugmentation
      rect_masks: 5   # Number of rectangles to cut from any given spectrogram
      rect_freq: 50   # Max cut of size 50 along the frequency dimension
      rect_time: 120  # Max cut of size 120 along the time dimension

You can also use this module for `SpecAugment <https://arxiv.org/abs/1904.08779>`_ if you would like to cut
out bands of audio features rather than rectangles.

See the :ref:`api:Audio Augmentors` section for more details.


Model Architecture Configurations
---------------------------------

Each configuration file should describe the model architecture being used for the experiment.
Models in the NeMo ASR collection need a ``encoder`` section and a ``decoder`` section, with the ``_target_`` field
specifying the module to use for each.

The following sections go into more detail about the specific configurations of each model architecture.

For more information about each model, see the :doc:`Models <./models>` page.

Jasper and QuartzNet
~~~~~~~~~~~~~~~~~~~~

The Jasper and QuartzNet models are very similar, and as such the components in their configs look similar as well.
