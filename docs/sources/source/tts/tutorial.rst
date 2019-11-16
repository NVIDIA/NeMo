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
<I should write a script for this>.

Training
---------
Training tacotron 2 is pretty easy. Try it here!


Mixed Precision training
-------------------------
For mixed precision training, I need to change the code.


Multi-GPU training
-------------------
`python -m torch.distributed.launch --nproc_per_node=<num_gpus> <nemo_git_repo_root>/examples/tts/tacotron2.py ...`


Inference
---------
Use the tts_infer script.