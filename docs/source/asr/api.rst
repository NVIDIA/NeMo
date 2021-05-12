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

.. autoclass:: nemo.collections.asr.modules.ConformerEncoder
    :show-inheritance:
    :members:


Parts
-----

.. autoclass:: nemo.collections.asr.parts.jasper.JasperBlock
    :show-inheritance:
    :members:


Mixins
------

.. autoclass:: nemo.collections.asr.parts.mixins.ASRBPEMixin
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.mixins.ASRModuleMixin
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

.. autoclass:: nemo.collections.asr.data.audio_to_text.AudioToCharWithDursF0Dataset
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

.. autoclass:: nemo.collections.asr.parts.perturb.SpeedPerturbation
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.perturb.TimeStretchPerturbation
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.perturb.GainPerturbation
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.perturb.ImpulsePerturbation
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.perturb.ShiftPerturbation
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.perturb.NoisePerturbation
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.perturb.WhiteNoisePerturbation
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.perturb.RirAndNoisePerturbation
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.perturb.TranscodePerturbation
    :show-inheritance:
    :members:

Miscellaneous Classes
---------------------

RNNT Decoding
~~~~~~~~~~~~~

.. autoclass:: nemo.collections.asr.metrics.rnnt_wer.RNNTDecoding
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.metrics.rnnt_wer_bpe.RNNTBPEDecoding
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.rnnt_greedy_decoding.GreedyRNNTInfer
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.rnnt_greedy_decoding.GreedyBatchedRNNTInfer
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.parts.rnnt_beam_decoding.BeamRNNTInfer
    :show-inheritance:
    :members:

Hypotheses
~~~~~~~~~~

.. autoclass:: nemo.collections.asr.parts.rnnt_utils.Hypothesis
    :show-inheritance:
    :no-members:

.. autoclass:: nemo.collections.asr.parts.rnnt_utils.NBestHypotheses
    :show-inheritance:
    :no-members:
