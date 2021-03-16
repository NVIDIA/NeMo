NeMo ASR Configuration Files
============================

This page covers NeMo configuration file setup that is specific to speaker recognition models.
For general information about how to set up and run experiments that is common to all NeMo models (e.g.
experiment manager and PyTorch Lightning trainer parameters), see the :doc:`../../introduction/core` page.

The model section of NeMo ASR configuration files will generally require information about the dataset(s) being
used, the preprocessor for audio files, parameters for any augmentation being performed, as well as the
model architecture specification.
The sections on this page cover each of these in more detail.

Example configuration files for all of the Speaker related scripts can be found in the
`config directory of the examples <https://github.com/NVIDIA/NeMo/tree/r1.0.0rc1/examples/speaker_recognition/conf>`_.


Dataset Configuration
---------------------

Training, validation, and test parameters are specified using the ``train_ds``, ``validation_ds``, and
``test_ds`` sections of your configuration file, respectively.
Depending on the task, you may have arguments specifying the sample rate of your audio files, max time length to consider for each audio file , whether or not to shuffle the dataset, and so on.
You may also decide to leave fields such as the ``manifest_filepath`` blank, to be specified via the command line
at runtime.

Any initialization parameters that are accepted for the Dataset class used in your experiment
can be set in the config file.

An example SpeakerNet train and validation configuration could look like:

.. code-block:: yaml

  model:
    train_ds:
      manifest_filepath: ???
      sample_rate: 16000
      labels: None   # finds labels based on manifest file
      batch_size: 32
      trim_silence: False
      time_length: 8
      shuffle: True
      is_tarred: False  # If set to true, uses the tarred version of the Dataset
      tarred_audio_filepaths: null      # Not used if is_tarred is false
      tarred_shard_strategy: "scatter"  # Not used if is_tarred is false

    validation_ds:
      manifest_filepath: ???
      sample_rate: 16000
      labels: None   # Keep None, to match with labels extracted during training
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

See the `Audio Preprocessors <../api.html#Audio Preprocessors>`__ API page for the preprocessor options, expected arguments, and defaults.


Augmentation Configurations
---------------------------

For SpeakerNet training we use on-the fly augmentations with MUSAN and RIR impulses using ``noise`` augmentor section

The following example sets up musan augmentation with audio files taken from manifest path and 
minimum and maximum SNR specified with min_snr and max_snr respectively. This section can be added to 
train_ds part in model

.. code-block:: yaml

  model:
    ...
    train_ds:
      ...
      augmentor:
        noise:
          manifest_path: /path/to/musan/manifest_file
          prob: 0.2  # probability to augment the incoming batch audio with augmentor data
          min_snr_db: 5 
          max_snr_db: 15        


See the `nemo.collections.asr.parts.perturb.AudioAugmentor`  API section for more details.


Model Architecture Configurations
---------------------------------

Each configuration file should describe the model architecture being used for the experiment.
Models in the NeMo ASR collection need a ``encoder`` section and a ``decoder`` section, with the ``_target_`` field
specifying the module to use for each.

The following sections go into more detail about the specific configurations of each model architecture.

For more information about the SpeakerNet Encoder models, see the :doc:`Models <./models>` page and at `Jasper and QuartzNet <../configs.html#jasper-and-quartznet>`__

Decoder Configurations
------------------------

After features have been computed from speakernet encoder, we pass the features to decoder to compute embeddings and then to compute log_probs 
for training models.

.. code-block:: yaml

  model:
    ...
    decoder:
      _target_: nemo.collections.asr.modules.SpeakerDecoder
      feat_in: *enc_final
      num_classes: 7205  # Total number of classes in training manifest file
      pool_mode: xvector # xvector for variance and mean bases statistics pooling 
      emb_sizes: 256 # number of inermediate emb layers. can be comma separated for additional layers like 512,512
      angular: true # if true then loss will be changed to angular softmax loss and consider scale and margin from loss section else train with cross-entrophy loss
    
    loss:
      scale: 30
      margin 0.2