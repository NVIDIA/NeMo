Base Classes
============

The NeMo TTS has two base classes corresponding to the two stage pipeline:

  - :class:`SpectrogramGenerator<nemo.collections.tts.models.base.SpectrogramGenerator>`
  - :class:`Vocoder<nemo.collections.tts.models.base.Vocoder>`

The :class:`SpectrogramGenerator<nemo.collections.tts.models.base.SpectrogramGenerator>` class has two important
functions: :py:meth:`parse<nemo.collections.tts.models.base.SpectrogramGenerator.parse>` which
accepts raw python strings and returns a torch.tensor that respresents tokenized text ready to pass to
:py:meth:`generate_spectrogram<nemo.collections.tts.models.base.SpectrogramGenerator.generate_spectrogram>` which
accepts a batch of tokenized text and returns a torch.tensor that represents a batch of spectrograms

The :class:`Vocoder<nemo.collections.tts.models.base.Vocoder>` class has one important functions
:py:meth:`convert_spectrogram_to_audio<nemo.collections.tts.models.base.Vocoder.convert_spectrogram_to_audio>` which
accepts a batch of spectrograms and returns a torch.tensor that represents a batch of raw audio.

.. autoclass:: nemo.collections.tts.models.base.SpectrogramGenerator
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.tts.models.base.Vocoder
    :show-inheritance:
    :members:
