NeMo TTS Collection API
=======================

TTS Base Classes
----------------

The classes below are the base of the TTS pipeline.
To read more about them, see the `Base Classes <./intro.html#Base Classes>`__ section of the intro page.

.. autoclass:: nemo.collections.tts.models.base.SpectrogramGenerator
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.tts.models.base.Vocoder
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.tts.models.base.TextToWaveform
    :show-inheritance:
    :members:

TTS Datasets
------------

.. autoclass:: nemo.collections.tts.data.datalayers.AudioDataset
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.tts.data.datalayers.MelAudioDataset
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.tts.data.datalayers.SplicedAudioDataset
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.tts.data.datalayers.NoisySpecsDataset
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.asr.data.audio_to_text.FastPitchDataset
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.tts.data.datalayers.FastSpeech2Dataset
    :show-inheritance:
    :members:
