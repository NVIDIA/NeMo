Speech Data Explorer
--------------------

[Dash](https://plotly.com/dash/)-based tool for interactive exploration of ASR/TTS datasets.

Features:
- dataset's statistics (alphabet, vocabulary, duration-based histograms)
- navigation across dataset (sorting, filtering)
- inspection of individual utterances (waveform, spectrogram, audio player)
- errors' analysis (Word Error Rate, Character Error Rate, Word Match Rate, Mean Word Accuracy, diff)
- visual comparation of two models (on same dataset) (beta)

Please make sure that requirements are installed. Then run:
```
python data_explorer.py path_to_manifest.json

or to try new features (comparation tool):

python data_explorer.py path_to_manifest.json -c1 path_to_manifest_transcribed_1.json -c2 path_to_manifest_transcribed_2.json

```

Errors' analysis requires "pred_text" (ASR transcript) for all utterances.

Any additional field will be parsed and displayed in 'Samples' tab.

JSON manifest file should contain the following fields:
- "audio_filepath" (path to audio file)
- "duration" (duration of the audio file in seconds)
- "text" (reference transcript)

JSON manifests for comparation tool should have the same origin. 
examle: Lets assume, that we have manifest of dataset called data_1. To use comparation tool you need to do next steps:
1) transcribe data_1.json by a first model and rename resulting .json file to <Model_1_name>.json
2) transcribe data_1.json by a second model and rename resulting .json file to <Model_2_name>.json
Then run with -c1 and -c2 options and go to "Comparation tool" tab


There you will see an interactive graph, and choose which data will display on each axis. I'd reccomend accuracy_model_1 from accuracy_model_2 as it's easy to understand


![Speech Data Explorer](screenshot.png)

To try Comparation tool - go to corresponding tab
![image](https://user-images.githubusercontent.com/37293288/183735563-ba6c1819-a320-46bc-8eaa-14ed77e93787.png)

