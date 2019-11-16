Tutorial
========

Make sure you have installed ``nemo``, ``nemo_asr``, and ``nemo_tts``
collection. See :ref:`installation` section.

.. note::
    You only need `nemo`, `nemo_asr`, and `nemo_tts` collection for this
    tutorial.

Introduction
-------------
Speech synthesis, also called Text to Speech (TTS), generally creates human
audible speech from text. TTS using neural networks typically consists of two
neural networks. The first neural network converts text to an intermediate
audio representation, usually derived from a spectrogram. The second neural
network, called the neural vocoder, converts the audio representation to audio
files, for us in the form of .wav files. While there has been recent research
shown to be able to combine these two models into one, we focus on the two
model approach.

NeMo supports the following models:

1. Tacotron 2: a model that converts text to mel spectrograms (citation needed)
2. Waveglow: a model that converts mel spectrograms to audio (citation needed)

Get data
--------
Both Tacotron 2 and Waveglow are trained using the LJSpeech (citation) dataset.

<I should write a script to grab the data>.

Training
---------
<Should talk about how to train Tacotron 2>


Mixed Precision training
-------------------------
Enabling or disabling mixed precision training can be changed through a command
line argument --amp_opt_level. Recommended and default values for Tacotron 2
and Waveglow are O1. It can be:

- O0: float32 training
- O1: mixed precision training
- O2: mixed precision training
- O3: float16 training


Multi-GPU training
-------------------
`python -m torch.distributed.launch --nproc_per_node=<num_gpus> <nemo_git_repo_root>/examples/tts/tacotron2.py ...`
<I need to describe this section in detail>


Inference
---------
Use the tts_infer script.
<I need to describe this section in detail>
