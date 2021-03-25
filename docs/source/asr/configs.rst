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
      num_workers: 8
      pin_memory: true

    validation_ds:
      manifest_filepath: ???
      sample_rate: 16000
      labels: *labels   # Uses the labels above
      batch_size: 32
      shuffle: False    # No need to shuffle the validation data
      num_workers: 8
      pin_memory: true


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

With NeMo ASR one can also add augmentation pipelines that can be used to simulate various kinds of noise
added to audio in the channel. Augmentors in a pipeline are applied on the audio data read in the data layer. Online
augmentors can be specified in the config file using an ``augmentor`` section in ``train_ds``. The following example
adds an augmentation pipeline that first adds white noise to an audio sample with a probability of 0.5 and at a level
randomly picked between -50 dB and -10 dB and then pass the resultant samples through a room impulse response randomly
picked from the manifest file provided for ``impulse`` augmentation in the config file.

.. code-block:: yaml

  model:
    ...
    train_ds:
    ...
        augmentor:
            white_noise:
                prob: 0.5
                min_level: -50
                max_level: -10
            impulse:
                prob: 0.3
                manifest_path: /path/to/impulse_manifest.json

See the `Audio Augmentors <./api.html#Audio Augmentors>`__ API section for more details.

Tokenizer Configurations
------------------------

Some models utilize sub-word encoding via an external tokenizer instead of explicitly defining their vocabulary.

For such models, a ``tokenizer`` section is added  to the model config. ASR Models currently support two types of
custom tokenizers - Google Sentencepiece tokenizers (tokenizer type of ``bpe`` in the config) or HuggingFace WordPiece tokenizers (tokenizer type of ``wpe`` in the config).

In order to build custom tokenizers, please refer to the ``ASR_with_Subword_Tokenization`` notebook available in the
ASR tutorials directory.

The following example sets up a ``SentencePiece Tokenizer`` at a path specified by the user:

.. code-block:: yaml

  model:
    ...
    tokenizer:
      dir: "<path to the directory that contains the custom tokenizer files>"
      type: "bpe"  # can be "bpe" or "wpe"

For models which utilize sub-word tokenization, we share the decoder module (``ConvASRDecoder``) with character tokenization models. All parameters are shared, but for models which utilize sub-word encoding, there are minor differences when setting up the config. For such models, the tokenizer is utilized to fill in the missing information when the model is constructed automatically.

For example, a decoder config corresponding to a sub-word tokenization model would look like this:

.. code-block:: yaml

  model:
    ...
    decoder:
      _target_: nemo.collections.asr.modules.ConvASRDecoder
      feat_in: *enc_final
      num_classes: -1  # filled with vocabulary size from tokenizer at runtime
      vocabulary: []  # filled with vocabulary from tokenizer at runtime


Model Architecture Configurations
---------------------------------

Each configuration file should describe the model architecture being used for the experiment.
Models in the NeMo ASR collection need a ``encoder`` section and a ``decoder`` section, with the ``_target_`` field
specifying the module to use for each.

Here is the list of the parameters in the model section which are shared among most of the ASR models:

+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+---------------------------------+
| **Parameter**           | **Datatype**     | **Description**                                                                                               | **Supported Values**            |
+=========================+==================+===============================================================================================================+=================================+
| :code:`log_prediction`  | bool             | Whether a random sample should be printed in the output at each step, along with its predicted transcript     |                                 |
+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+---------------------------------+
| :code:`ctc_reduction`   | string           | Specifies the reduction type of CTC loss. Defaults to 'mean_batch' which would take average over the batch    | :code:`none`, :code:`mean_batch`|
|                         |                  | after taking the average over the length of each sample.                                                      | :code:`mean`, :code:`sum`       |
+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+---------------------------------+

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

Citrinet
~~~~~~~~

The `Citrinet <./models.html#Citrinet>`__ and `QuartzNet <./models.html#QuartzNet>`__ models are very similar, and as such the components in their configs are very similar as well. Citrinet utilizes Squeeze and Excitation, as well as sub-word tokenization, in contrast to QuartzNet. Depending on the dataset, we utilize different tokenizers. For Librispeech, we utilize the HuggingFace WordPiece tokenizer, and for all other datasets we utilize the Google Sentencepiece tokenizer - usually the ``unigram`` tokenizer type.

Both architectures use the ``ConvASREncoder`` for the ``encoder``, with parameters detailed above.
The encoder parameters include details about the Citrinet-C encoder architecture, including how many
filters are used per channel (C). The Citrinet-C configuration is a shortform notation for Citrinet-21x5xC, such that B = 21 and R = 5 are the default and should generally not be changed.

To use Citrinet instead of QuartzNet, please refer to the ``citrinet_512.yaml`` configuration found inside the ``examples/asr/conf/citrinet`` directory. Citrinet is primarily comprised of the same :class:`~nemo.collections.asr.parts.jasper.JasperBlock` as ``Jasper`` or ``QuartzNet`.

While the configs for Citrinet and QuartzNet are similar, we note the additional flags used for Citrinet below. Please refer to the ``JasperBlock`` documentation for the meaning of these arguments.

+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+---------------------------------+
| **Parameter**           | **Datatype**     | **Description**                                                                                               | **Supported Values**            |
+=========================+==================+===============================================================================================================+=================================+
| :code:`se`              | bool             | Whether to apply squeeze-and-excitation mechanism or not.                                                     | :code:`true` or :code:`false`   |
+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+---------------------------------+
| :code:`se_context_size` | int              | SE context size. -1 means global context.                                                                     | :code:`-1` or :code:`+ve int`   |
+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+---------------------------------+
| :code:`stride_last`     | bool             | Stride on the final repeated block or all repeated blocks.                                                    | :code:`true` or :code:`false`   |
+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+---------------------------------+
| :code:`residual_mode`   | str              | | Type of residual branch to construct.                                                                       | :code:`"add"` or                |
|                         |                  | | Can be pointwise residual addition or pointwise strided residual attention                                  | :code:`"stride_add"`            |
+-------------------------+------------------+---------------------------------------------------------------------------------------------------------------+---------------------------------+

A Citrinet-512 config might look like below.

.. code-block:: yaml

  model:
    ...
    # Specify some defaults across the entire model
    model_defaults:
      repeat: 5
      dropout: 0.1
      separable: true
      se: true
      se_context_size: -1
    ...
    encoder:
      _target_: nemo.collections.asr.modules.ConvASREncoder
      feat_in: *n_mels  # Should match "features" in the preprocessor.
      activation: relu
      conv_mask: true

      jasper:   # This field name should be "jasper" for the JasperBlock (which constructs Citrinet).

      # Prologue block
      - filters: 512
        repeat: 1
        kernel: [5]
        stride: [1]
        dilation: [1]
        dropout: 0.0
        residual: false
        separable: ${model.model_defaults.separable}
        se: ${model.model_defaults.se}
        se_context_size: ${model.model_defaults.se_context_size}

      # Block 1
      - filters: 512
        repeat: ${model.model_defaults.repeat}
        kernel: [11]
        stride: [2]
        dilation: [1]
        dropout: ${model.model_defaults.dropout}
        residual: true
        separable: ${model.model_defaults.separable}
        se: ${model.model_defaults.se}
        se_context_size: ${model.model_defaults.se_context_size}
        stride_last: true
        residual_mode: "stride_add"

      ... # Entries for blocks 2~21

      # Block 22
      - filters: 512
        repeat: ${model.model_defaults.repeat}
        kernel: [39]
        stride: [1]
        dilation: [1]
        dropout: ${model.model_defaults.dropout}
        residual: true
        separable: ${model.model_defaults.separable}
        se: ${model.model_defaults.se}
        se_context_size: ${model.model_defaults.se_context_size}

      # Epilogue block

      - filters: &enc_final 640
        repeat: 1
        kernel: [41]
        stride: [1]
        dilation: [1]
        dropout: 0.0
        residual: false
        separable: ${model.model_defaults.separable}
        se: ${model.model_defaults.se}
        se_context_size: ${model.model_defaults.se_context_size}

As discussed above, Citrinet uses the ``ConvASRDecoder`` as the decoder layer similar to QuartzNet. Only the configuration must be changed slightly as Citrinet is utilizes sub-word tokenization.


Conformer-CTC
~~~~~~~~~~~~~

You may find the config files for Conformer-CTC model with character-based encoding and sub-word encoding at ``<NeMo_git_root>/examples/asr/conf/conformer/conformer_ctc_char.yaml`` and ``<NeMo_git_root>/examples/asr/conf/conformer/conformer_ctc_bpe.yaml`` respectively.
Some components of the configs of `Conformer-CTC <./models.html#Conformer-CTC>`__ including
datasets (train_ds, validation_ds, and test_ds), opimizer (optim), augmentation (spec_augment), decoder, trainer, and exp_manager are
similar to other ASR models like `QuartzNet <./models.html#QuartzNet>`__. There should be a tokenizer section which you may specify the tokenizer if you want to use sub-word encoding instead of character-based encoding.

The encoder section includes the details about the Conformer-CTC encoder architecture.
You may find more info on this section in the config files and also here :doc:`nemo.collections.asr.modules.ConformerEncoder<./api.html#nemo.collections.asr.modules.ConformerEncoder>`.
