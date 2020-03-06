.. _fastspeech:

Fast Speech
===========

Model
-----
This model is based on the
`Fast Speech model <https://www.microsoft.com/en-us/research/blog/fastspeech-new-text-to-speech-model-improves-on-speed-accuracy-and-controllability>`_
(see also `paper <https://arxiv.org/abs/1905.09263>`_).

Fast Speech operates within two distinct stages: durations extraction and actual training.

Durations Extraction
++++++++++++++++++++

First, for each input dataset char, you should obtain duration integer, which corresponds to the number of steps this
char lasts in audio sample. For that, NeMo uses alignment map from Tacotron 2 inference teacher forced with ground
truth mel spectrogram for shape matching. For each timestep, we add +1 duration to char with strongest alignment signal
in alignment map.

To do that, run fastspeech_alignments.py from NeMo/examples/tts with following arguments (provide path to durations
dir to save):

.. code-block:: bash

    python fastspeech_durations.py --spec_model=tacotron2 --spec_model_config=configs/tacotron2.yaml --spec_model_load_dir=<directory_with_tacotron2_checkopints> --eval_dataset=<data_root>/ljspeech_train.json --durations_dir=<data_root>/durs

Fast Speech Training
++++++++++++++++++++

Second stage is actual model training. NeMo wrap all fast speech mel generation and durations calculation logic in one
neural model with corresponded name. FastSpeechLoss, then uses its output to calculate two term loss value.

To begin training with librispeech data durations obtained from previous step, run this:

.. code-block:: bash

    python fastspeech.py --model_config=configs/fastspeech.yaml --train_dataset=<data_root>/ljspeech_train.json --durations_dir=<data_root>/durs
