NeMo ASR Configuration Files
============================

This page covers NeMo configuration file setup that is specific to models in the ASR collection.
For general information about how to set up and run experiments that is common to all NeMo models (e.g.
experiment manager and PyTorch Lightning trainer parameters), see the :doc:`../core` page.

The model section of NeMo ASR configuration files will generally require information about the dataset(s) being
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
See the `Datasets <./api.html#Datasets>`__ section of the API for a list of Datasets and their respective parameters.

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

See the `Audio Preprocessors <./api.html#Audio Preprocessors>`__ API page for the preprocessor options, expected arguments, and defaults.


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
      rect_freq: 50   # Max cut of size 50 along the frequency dimension
      rect_time: 120  # Max cut of size 120 along the time dimension
      # SpecAugment parameters
      freq_masks: 2   # Cut two frequency bands
      freq_width: 15  # ... of width 15 at maximum
      time_masks: 5    # Cut out 10 time bands
      time_width: 25  # ... of width 25 at maximum

You can use any combination of Cutout, frequency/time SpecAugment, or none of them.

See the `Audio Augmentors <./api.html#Audio Augmentors>`__ API section for more details.


Model Architecture Configurations
---------------------------------

Each configuration file should describe the model architecture being used for the experiment.
Models in the NeMo ASR collection need a ``encoder`` section and a ``decoder`` section, with the ``_target_`` field
specifying the module to use for each.

The following sections go into more detail about the specific configurations of each model architecture.

For more information about the ASR models, see the :doc:`Models <./models>` page.

Jasper and QuartzNet
~~~~~~~~~~~~~~~~~~~~

The `Jasper <./models.html#Jasper>`__ and `QuartzNet <./models.html#QuartzNet>`__ models are very similar, and as such the components in their
configs are very similar as well.

Both architectures use the ``ConvASREncoder`` for the ``encoder``, with parameters detailed in the table below.
The encoder parameters include details about the Jasper/QuartzNet [BxR] encoder architecture, including how many
blocks to use (B), how many times to repeat each sub-block (R), and the convolution parameters for each block.

The number of blocks B is determined by the number of list elements under ``jasper`` minus the one prologue and
two epilogue blocks, and the number of sub-blocks R is determined by setting the ``repeat`` parameter.

To use QuartzNet (which uses more compact time-channel separable convolutions) instead of Jasper,
add :code:`separable: true` to all but the last block in the architecture.
(You should not change the parameter name ``jasper``.)

+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+---------------------------------+
| **Parameter**           | **Datatype**     | **Description**                                                                                               | **Supported Values**            |
+=========================+==================+===============================================================================================================+=================================+
| :code:`feat_in`         | int              | The number of input features. Should be equal to :code:`features` in the preprocessor parameters.             |                                 |
+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+---------------------------------+
| :code:`activation`      | string           | What activation function to use in the encoder.                                                               | :code:`hardtanh`, :code:`relu`, |
|                         |                  |                                                                                                               | :code:`selu`, :code:`swish`     |
+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+---------------------------------+
| :code:`conv_mask`       | bool             | Whether to used masked convolutions in the encoder. Defaults to true.                                         |                                 |
+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+---------------------------------+
| :code:`jasper`          |                  | | A list of blocks that specifies your encoder architecture. Each entry in this list represents one block in  |                                 |
|                         |                  | | the architecture and contains the parameters for that block, including convolution parameters, dropout, and |                                 |
|                         |                  | | the number of times the block is repeated. See the `Jasper <https://arxiv.org/pdf/1904.03288.pdf>`_ and     |                                 |
|                         |                  | | `QuartzNet <https://arxiv.org/pdf/1910.10261.pdf>`_ papers for details about specific model configurations. |                                 |
+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+---------------------------------+

A QuartzNet 15x5 (fifteen blocks, each sub-block repeated five times) encoder configuration may look like
the example below.

.. code-block:: yaml

  # Specified at the beginning of the file for convenience
  n_mels: &n_mels 64    # Used for both the preprocessor and encoder as number of input features
  repeat: &repeat 5     # R=5
  dropout: &dropout 0.0
  separable: &separable true  # Set to true for QN. Set to false for Jasper.

  model:
    ...
    encoder:
      _target_: nemo.collections.asr.modules.ConvASREncoder
      feat_in: *n_mels  # Should match "features" in the preprocessor.
      activation: relu
      conv_mask: true

      jasper:   # This field name should be "jasper" for both types of models.

      # Prologue block
      - dilation: [1]
        dropout: *dropout
        filters: 256
        kernel: [33]
        repeat: 1   # Prologue block is not repeated.
        residual: false
        separable: *separable
        stride: [2]

      # Block 1
      - dilation: [1]
        dropout: *dropout
        filters: 256
        kernel: [33]
        repeat: *repeat
        residual: true
        separable: *separable
        stride: [1]

      ... # Entries for blocks 2~14

      # Block 15
      - dilation: [1]
        dropout: *dropout
        filters: 512
        kernel: [75]
        repeat: *repeat
        residual: true
        separable: *separable
        stride: [1]

      # Two epilogue blocks
      - dilation: [2]
        dropout: *dropout
        filters: 512
        kernel: [87]
        repeat: 1   # Epilogue blocks are not repeated
        residual: false
        separable: *separable
        stride: [1]

      - dilation: [1]
        dropout: *dropout
        filters: &enc_filters 1024
        kernel: [1]
        repeat: 1   # Epilogue blocks are not repeated
        residual: false
        stride: [1]

Both Jasper and QuartzNet use the ``ConvASRDecoder`` as the decoder.
The decoder parameters are detailed in the following table.

+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+---------------------------------+
| **Parameter**           | **Datatype**     | **Description**                                                                                               | **Supported Values**            |
+=========================+==================+===============================================================================================================+=================================+
| :code:`feat_in`         | int              | The number of input features to the decoder. Should be equal to the number of filters in the last block of    |                                 |
|                         |                  | the encoder.                                                                                                  |                                 |
+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+---------------------------------+
| :code:`vocabulary`      | list             | A list of the valid output characters for your model. For example, for an English dataset, this could be a    |                                 |
|                         |                  | list of all lowercase letters, space, and apostrophe.                                                         |                                 |
+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+---------------------------------+
| :code:`num_classes`     | int              | Number of output classes, i.e. the length of :code:`vocabulary`.                                              |                                 |
+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+---------------------------------+

For example, a decoder config corresponding to the encoder above would look like this:

.. code-block:: yaml

  model:
    ...
    decoder:
      _target_: nemo.collections.asr.modules.ConvASRDecoder
      feat_in: *enc_filters
      vocabulary: *labels
      num_classes: 28   # Length of the vocabulary list
