Tutorial
========

Make sure you have installed ``nemo``, ``nemo_asr``, and ``nemo_tts``
collection. See the :ref:`installation` section.

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

NeMo supports the following models in full:

1. `Tacotron 2 <https://arxiv.org/abs/1712.05884>`_: a model that converts
   text to mel spectrograms
2. `Waveglow <https://arxiv.org/abs/1811.00002>`_: a model that converts mel
   spectrograms to audio

NeMo additionally supports the following models as experimental:

3. `FastSpeech <https://arxiv.org/abs/1905.09263>`_: a model that converts
   text to mel spectrograms
4. `Talknet <https://arxiv.org/abs/2005.05514>`_: a model that converts
   text to mel spectrograms

To train your own models, you can go through the following sections. If you
want to run inference with our pre-trained models, skip to the
:ref:`Inference <TTS Inference>` section.

Get data
--------
Both Tacotron 2 and Waveglow are trained using the
`LJSpeech <https://keithito.com/LJ-Speech-Dataset/>`__ dataset.
You can use a helper script to get and process the dataset for use with NeMo.
The script is located at NeMo/scripts and can be run like so:

.. code-block:: bash

    python scripts/get_ljspeech_data.py --data_root=<where_you_want_to_save_data>

For more details on the LJSpeech dataset, see :ref:`our docs here <ljspeech>`.

For Mandarin, we use `Chinese Standard Mandarin Speech Copus <https://www.data-baker.com/open_source.html>`__.
You can use a helper script to get and process the dataset for use with NeMo. The dataset download link in the
script is provided by Databaker (Beijing) Technology Co.,Ltd.
The script is located at NeMo/scripts and can be run like so:

.. code-block:: bash

    python scripts/get_databaker_data.py --data_root=<where_you_want_to_save_data>

For more details on the Chinese Standard Mandarin Speech Corpus, see :ref:`our docs here <csmsc>`.

Training
---------
NeMo supports training both Tacotron 2 and Waveglow. For the purposes of this
tutorial, we will be focusing on training Tacotron 2 as that determines the
majority of the characteristics of the trained audio such as the gender, and
prosody. Furthermore, in our experiments, Waveglow has been shown to work as
an universal vocoder. Our pretrained Waveglow, though trained on read female
English speech, can be used as vocoder for male voices as well as other languages
such as Mandarin.

Training Tacotron 2 can be done by running the `tacotron2.py` file inside
NeMo/examples/tts. Assuming you are inside the NeMo/examples/tts directory,
you can run the following to start training:

.. code-block:: bash

    python tacotron2.py --train_dataset=<data_root>/ljspeech_train.json --eval_datasets <data_root>/ljspeech_eval.json --model_config=configs/tacotron2.yaml --max_steps=30000

Training Tacotron 2 on Mandarin also can be done by running the `tacotron2.py` file.
You can run the following to start training:

.. code-block:: bash

    python tacotron2.py --train_dataset=<data_root>/databaker_csmsc_train.json --eval_datasets <data_root>/databaker_csmsc_eval.json --model_config=configs/tacotron2_mandarin.yaml --max_steps=30000

.. tip::
    Tacotron 2 normally takes around 20,000 steps for attention to be learned.
    Once attention is learned, this is when you can use the model to generate
    audible speech.

Mixed Precision training
-------------------------
Enabling or disabling mixed precision training can be changed through a command
line argument ``--amp_opt_level``. Recommended and default values for Tacotron 2 are O0,
whereas values for Waveglow are O1. Options for amp_opt_level are:

- O0: float32 training
- O1: mixed precision training
- O2: mixed precision training
- O3: float16 training

.. note::
    Because mixed precision requires Tensor Cores it only works on NVIDIA
    Volta and Turing based GPUs

Multi-GPU training
-------------------
Running on multiple GPUs can be enabled but calling running the
torch.distributed.launch module and specifying the num_gpus as the
--nproc_per_node argument:

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=<num_gpus> <nemo_git_repo_root>/examples/tts/tacotron2.py ...


.. _TTS Inference:

Inference
---------
For inference in English, please refer to our
`Tacotron Inference <https://github.com/NVIDIA/NeMo/blob/master/examples/tts/notebooks/1_Tacotron_inference.ipynb>`_
notebook.
