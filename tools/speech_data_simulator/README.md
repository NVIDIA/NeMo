**Speech Data Simulator**
===============

Outline
------------

The speech data simulator generates synthetic multispeaker audio sessions for training or evaluating models for multispeaker ASR or speaker diarization. This tool aims to address the lack of labelled multispeaker training data and to help models deal with overlapping speech.

The simulator loads audio files from different speakers from the LibriSpeech dataset as well as forced alignments for each sentence and concatenates the audio files together to build a synthetic multispeaker audio session. The simulator uses the word alignments to segment the audio from each speaker to produce utterances of the desired length. The simulator also incorporates synthetic room impulse response generation in order to simulate multi-microphone multispeaker sessions.

Features
------------

The simulator is reconfigurable and has several options including:

* Amount of overlapping speech
* Percentage of silence
* Sentence length distribution
* Number of speakers
* Session length
* Speaker dominance
* Turn taking
* Background noise

The simulator can be used in two modes: near field (no Room Impulse Response) as well as far field (including synthetic RIR). When using synthetic RIR generation, multiple microphones can be placed in the simulated room environment for multichannel simulations.

The simulator also has a speaker enforcement mode which ensures that the correct number of speakers appear in each session (which is not guaranteed since speaker turns are stochastic). In speaker enforcement mode, the length of the session or speaker probabilites may be adjusted to ensure all speakers are present.

Required Datasets
------------

* LibriSpeech (or another single-speaker dataset)
* LibriSpeech word alignments from [here](https://github.com/CorentinJ/librispeech-alignments) (or alignments corresponding to another single-speaker dataset)

Optional Datasets
------------

* Room Impulse Response and Noise Database from [here](https://www.openslr.org/resources/28/rirs_noises.zip) (or another background noise dataset)

Installation (after installing NeMo)
------------

```bash
pip install cmake
pip install https://github.com/DavidDiazGuerra/gpuRIR/zipball/master
```

Parameters
------------

* Data simulator parameters are contained in conf/data_simulator.yaml
* Additional RIR generation parameters are contained in conf/data_simulator_rir.yaml

Running the data simulator
------------

1. Download the LibriSpeech dataset

```bash
export PYTHONPATH=<NeMo base path>:$PYTHONPATH \
&& python scripts/dataset_processing/get_librispeech_data.py \
  --data_root <path to download LibriSpeech dataset to> \
  --data_sets ALL
```

2. Download LibriSpeech alignments from [here](https://drive.google.com/file/d/1WYfgr31T-PPwMcxuAq09XZfHQO5Mw8fE/view?usp=sharing) (the base directory is the LibriSpeech-Alignments directory)

3. Create the manifest file with alignments

```bash
export PYTHONPATH=<NeMo base path>:$PYTHONPATH \
&& python <NeMo base path>/scripts/speaker_tasks/create_librispeech_alignment_manifest.py \
  --input_manifest_filepath <Path to train_clean_100.json manifest file> \
  --base_alignment_path <Path to LibriSpeech_Alignments directory> \
  --dataset train-clean-100 \
  --output_path train-clean-100-align.json
```

4. (Optional) Create the background noise manifest file

```bash
export PYTHONPATH=<NeMo base path>:$PYTHONPATH \
&& python <NeMo base path>/scripts/speaker_tasks/pathfiles_to_diarize_manifest.py \
--paths2audio_files <Path to noise list file> \
--manifest_filepath bg_noise.json
```

5. Create audio sessions (near field)

```bash
export PYTHONPATH=<NeMo base path>:$PYTHONPATH \
&& python multispeaker_simulator.py --config-path='conf' --config-name='data_simulator.yaml' \
  data_simulator.random_seed=42 \
  data_simulator.manifest_path=./train-clean-100-align.json \
  data_simulator.outputs.output_dir=./test \
  data_simulator.background_noise.background_manifest=./bg_noise.json
```

6. Create multi-microphone audio sessions (with synthetic RIR generation)

```bash
export PYTHONPATH=<NeMo base path>:$PYTHONPATH \
&& python multispeaker_simulator.py --config-path='conf' --config-name='data_simulator_rir.yaml' \
  data_simulator.random_seed=42 \
  data_simulator.manifest_path=./train-clean-100-align.json \
  data_simulator.outputs.output_dir=./test_rir \
  data_simulator.background_noise.background_manifest=./bg_noise.json
```

