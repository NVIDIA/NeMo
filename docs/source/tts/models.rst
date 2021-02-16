Models
======

Currently, NeMo's TTS collection supports the following models:

Tacotron 2
----------

Tacotron 2 :cite:`tts-models-shen2018natural` is an autoregressive rnn-based Encoder-Attention-Decoder model that
generates mel spectrograms.

Tacotron 2 can be instantiated using the :class:`Tacotron2Model<nemo.collections.tts.models.Tacotron2Model>` class.

Glow-TTS
--------

Glow-TTS :cite:`tts-models-kim2020glow` is a Flow-based generative model that generates mel spectrograms.

Glow-TTS can be instantiated using the :class:`GlowTTSModel<nemo.collections.tts.models.GlowTTSModel>` class.

.. _WaveGlow_Model:

WaveGlow
--------

WaveGlow :cite:`tts-models-prenger2018waveglow` is a Flow-based generative model that generates audio from mel spectrograms.
Comprised of several flow steps, WaveGlow learns an invertible mapping from a simple latent space to audio waveforms.

    .. image:: waveglow.png
        :align: center
        :alt: waveglow model

WaveGlow can be instantiated using the :class:`WaveGlowModel<nemo.collections.tts.models.WaveGlowModel>` class.


SqueezeWave
-----------

SqueezeWave :cite:`tts-models-zhai2020squeezewave` is a version of WaveGlow :cite:`tts-models-prenger2018waveglow` that simplifies the architecture of the WaveNet (WN) module by introducing depthwise separable convolutions and removing dual channels.
SqueezeWave also uses larger group sizes, which reduces computation along the temporal axis and allows for less upsampling layers for mel spectrogram.

    .. image:: squeezewave_wn.png
        :align: center
        :alt: squeezewave vs waveglow wavenet modules

SqueezeWave can be instantiated using the :class:`SqueezeWaveModel<nemo.collections.tts.models.SqueezeWaveModel>` class.

Full-band MelGAN
----------------

The :class:`MelGanModel<nemo.collections.tts.models.MelGanModel>` class implements Full-band MelGAN as described in
:cite:`yang2020multiband`. MelGAN is a GAN-based vocoder that converts mel-spectrograms to audio.

HifiGAN
-------

The :class:`HifiGanModel<nemo.collections.tts.models.HifiGanModel>` class implements HifiGAN as described in
:cite:`kong2020hifi`. HifiGAN is a GAN-based vocoder that converts mel-spectrograms to audio. 

References
----------

.. bibliography:: tts_all.bib
    :style: plain
    :labelprefix: TTS-MODELS
    :keyprefix: tts-models-
