NeMo ASR collection API
=======================


Model Classes
-------------

.. autoclass:: nemo.collections.asr.models.EncDecCTCModel
    :show-inheritance:
    :members: transcribe, change_vocabulary, setup_training_data, setup_optimization, setup_validation_data, setup_test_data, register_artifact


.. autoclass:: nemo.collections.asr.models.EncDecCTCModelBPE
    :show-inheritance:
    :members: transcribe, change_vocabulary, setup_training_data, setup_optimization, setup_validation_data, setup_test_data, register_artifact


.. autoclass:: nemo.collections.asr.models.EncDecRNNTModel
    :show-inheritance:
    :members: transcribe, change_vocabulary, setup_training_data, setup_optimization, setup_validation_data, setup_test_data, register_artifact


.. autoclass:: nemo.collections.asr.models.EncDecRNNTBPEModel
    :show-inheritance:
    :members: transcribe, change_vocabulary, setup_training_data, setup_optimization, setup_validation_data, setup_test_data, register_artifact


.. autoclass:: nemo.collections.asr.models.EncDecClassificationModel
    :show-inheritance:
    :members: setup_training_data, setup_optimization, setup_validation_data, setup_test_data, register_artifact


.. autoclass:: nemo.collections.asr.models.EncDecSpeakerLabelModel
    :show-inheritance:
    :members: setup_training_data, setup_optimization, setup_validation_data, setup_test_data, register_artifact


Modules
-------

.. autoclass:: nemo.collections.asr.modules.ConvASREncoder
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.modules.ConvASRDecoder
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.modules.ConvASRDecoderClassification
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.modules.SpeakerDecoder
    :show-inheritance:
    :members:


Mixins
------

.. autoclass:: nemo.collections.asr.parts.mixins.ASRBPEMixin
    :show-inheritance:
    :members:

Datasets
--------

Character Encoding Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nemo.collections.asr.data.audio_to_text.AudioToCharDataset
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.data.audio_to_text.AudioToCharWithDursDataset
    :show-inheritance:
    :members:

Subword Encoding Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nemo.collections.asr.data.audio_to_text.AudioToBPEDataset
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset
    :show-inheritance:
    :members:

Audio Preprocessors
-------------------

.. autoclass:: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.modules.AudioToMFCCPreprocessor
    :show-inheritance:
    :members:

Audio Augmentors
----------------

.. autoclass:: nemo.collections.asr.modules.SpectrogramAugmentation
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.modules.CropOrPadSpectrogramAugmentation
    :show-inheritance:
    :members:
