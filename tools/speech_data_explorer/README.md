Speech Data Explorer
--------------------

[Dash](https://plotly.com/dash/)-based tool for interactive exploration of ASR/TTS datasets. 

Features:
- dataset's statistics (alphabet, vocabulary, duration-based histograms)
- navigation across dataset (sorting, filtering)
- inspection of individual utterances (waveform, spectrogram, audio player)

Please make sure that requirements are installed. Then run:
```
python data_explorer.py path_to_manifest.json
```

![Speech Data Explorer](screenshot.png)
