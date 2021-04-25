NeMo ASR Configuration Files
============================

The model section of NeMo ASR configuration files requires information about the datasets being used, the preprocessor for audio files, 
parameters for any augmentation being performed, as well as the model architecture specification. The sections on this page cover each 
of these in more detail.

Example configuration files for all of the Speaker related scripts can be found in the config directory of the examples 
``{NEMO_ROOT/examples/speaker_recognition/conf}``.

For general information about how to set up and run experiments that is common to all NeMo models (for example, Experiment Manager and 
PyTorch Lightning Trainer parameters), refer to the :doc:`../../starthere/core` section.

Dataset Configuration
---------------------

Training, validation, and test parameters are specified using the ``train_ds``, ``validation_ds``, and ``test_ds`` sections of your 
configuration file, respectively. Depending on the task, you may have arguments specifying the sample rate of your audio files, max 
time length to consider for each audio file, whether or not to shuffle the dataset, and so on. You may also decide to leave fields such 
as the ``manifest_filepath`` blank, to be specified via the command-line at runtime.

Any initialization parameters that are accepted for the dataset class used in your experiment can be set in the config file.

An example SpeakerNet train and validation configuration should look similar to the following:

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

    validation_ds:
      manifest_filepath: ???
      sample_rate: 16000
      labels: None   # Keep None, to match with labels extracted during training
      batch_size: 32
      shuffle: False    # No need to shuffle the validation data
      
If you would like to use a tarred dataset, refer to the `Datasets Configuration <../configs.html#dataset-configuration>`__ section.

Preprocessor Configuration
--------------------------

Preprocessor helps to compute MFCC or mel spectrogram features that are given as inputs to model. For details on how to write this 
section, refer to `Preprocessor Configuration <../configs.html#preprocessor-configuration>`__.

Augmentation Configurations
---------------------------

For SpeakerNet training, we use augmentations with MUSAN and RIR impulses using ``noise`` augmentor section. The following example sets 
up musan augmentation with audio files taken from the manifest path and the minimum and maximum SNR specified with ``min_snr`` and 
``max_snr`` respectively. This section can be added to ``train_ds`` part as shown below.

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

Refer to the :class:``nemo.collections.asr.parts.perturb.AudioAugmentor`` API section for more details.

Model Architecture Configurations
---------------------------------

Each configuration file should describe the model architecture being used for the experiment. Models in the NeMo ASR collection need 
a ``encoder`` section and a ``decoder`` section, with the ``_target_`` field specifying the module to use for each.

The following sections go into more detail about the specific configurations of each model architecture.

For more information about the SpeakerNet models, refer to the :doc:`Models <./models>` and `Jasper and QuartzNet <../configs.html#jasper-and-quartznet>`__ sections.

Decoder Configurations
----------------------

After the features have been computed from the SpeakerNet encoder, we pass these features to the decoder to compute the embeddings 
and then to compute the log probabilities for training models.

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