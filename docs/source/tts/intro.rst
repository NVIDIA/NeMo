Speech Synthesis (TTS)
======================
Speech Synthesis or Text-to-Speech (TTS) involves turning text into human speech. The NeMo TTS collection currently
supports a two stage pipeline. First, a model is used to generate a mel spectrogram from text. Second, a model is used
to generate audio from a mel spectrogram. If you want to generate speech with NeMo's TTS models, please follow the link
on NeMo's README to the TTS notebook.

.. toctree::
   :maxdepth: 8

   base
   models
   model_API