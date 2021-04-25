Speech Data Explorer
====================

The Speech Data Explorer is a `Dash <https://plotly.com/dash/>`__-based tool for interactive exploration of ASR/TTS datasets that can 
be found in `NeMo/tools/speech_data_explorer <https://github.com/NVIDIA/NeMo/tree/main/tools/speech_data_explorer>`__.

Its main features include:

- dataset's statistics (alphabet, vocabulary, duration-based histograms)
- navigation across dataset (sorting, filtering)
- inspection of individual utterances (waveform, spectrogram, audio player)
- errors' analysis (Word Error Rate, Character Error Rate, Word Match Rate, Mean Word Accuracy, diff)

Ensure that requirements are installed, then run:

.. code::

    python data_explorer.py path_to_manifest.json

The JSON manifest file should contain the following fields:

- ``audio_filepath`` (path to audio file)
- ``duration`` (duration of the audio file in seconds)
- ``text`` (reference transcript)

Errors' analysis requires ``pred_text`` (ASR transcript) for all utterances.

Any additional field is parsed and displayed in the ``Samples`` tab.

