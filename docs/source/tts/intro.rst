Speech Synthesis (TTS)
======================
Speech Synthesis or Text-to-Speech (TTS) involves turning text into human speech. The NeMo TTS collection currently
supports a two stage pipeline. First, a model is used to generate a mel spectrogram from text. Second, a model is used
to generate audio from a mel spectrogram.

Quick Start:

.. code-block:: python

    import soundfile as sf
    from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder

    # Download and load the pretrained tacotron2 model
    spec_generator = SpectrogramGenerator.from_pretrained("Tacotron2-22050Hz")
    # Download and load the pretrained waveglow model
    vocoder = Vocoder.from_pretrained("WaveGlow-22050Hz")

    # All spectrogram generators start by parsing raw strings to a tokenized version of the string
    parsed = spec_gen.parse("You can type your sentence here to get nemo to produce speech.")
    # They then take the tokenized string and produce a spectrogram
    spectrogram = spec_gen.generate_spectrogram(tokens=parsed)
    # Finally, a vocoder converts the spectrogram to audio
    audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

    # Save the audio to disk in a file called speech.wav
    sf.write("speech.wav", audio.to('cpu').numpy(), 22050)

.. note::

   For an interactive version of the quick start above, refer to the TTS inference notebook that can be found on the
   github readme.

Available Models
################

NeMo supports a variety of models that can be used for TTS.

.. list-table:: *TTS Models*
   :widths: 5 5 10 25
   :header-rows: 1

   * - Model
     - Base Class
     - Pretrained Checkpoint
     - Description
   * - Tacotron2
     - :class:`SpectrogramGenerator<nemo.collections.tts.models.base.SpectrogramGenerator>`
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:tts_en_tacotron2
     - LSTM encoder decoder based model that generates spectrograms
   * - GlowTTS
     - :class:`SpectrogramGenerator<nemo.collections.tts.models.base.SpectrogramGenerator>`
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:tts_en_glowtts
     - Glow-based spectrogram generator
   * - WaveGlow
     - :class:`Vocoder<nemo.collections.tts.models.base.Vocoder>`
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:tts_waveglow_88m
     - Glow-based vocoder
   * - SqueezeWave
     - :class:`Vocoder<nemo.collections.tts.models.base.Vocoder>`
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:tts_squeezewave
     - Glow-based vocoder based on WaveGlow but with less parameters
   * - UniGlow
     - :class:`Vocoder<nemo.collections.tts.models.base.Vocoder>`
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:tts_uniglow
     - Glow-based vocoder based on WaveGlow but shares 1 set of parameters across all flow steps
   * - MelGAN
     - :class:`Vocoder<nemo.collections.tts.models.base.Vocoder>`
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:tts_melgan
     - GAN-based vocoder
   * - HiFiGAN
     - :class:`Vocoder<nemo.collections.tts.models.base.Vocoder>`
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:tts_hifigan
     - GAN-based vocoder

Base Classes
############

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

Training
########
Training of TTS models can be done using the scripts inside the NeMo examples/tts folders. The majority of the TTS
YAML configurations should work out of the box with the LJSpeech dataset. If you want to train on other data, it is
recommended that you walk through the Tacotron 2 Training notebook. Please pay special attention to the sample rate and
FFT parameters for new data.
