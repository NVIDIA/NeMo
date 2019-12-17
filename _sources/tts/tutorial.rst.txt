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

NeMo supports the following models:

1. `Tacotron 2 <https://arxiv.org/abs/1712.05884>`_ :a model that converts
   text to mel spectrograms
2. `Waveglow <https://arxiv.org/abs/1811.00002>`_: a model that converts mel
   spectrograms to audio

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

For more details on the Chinese Standard Mandarin Speech Copus, see :ref:`our docs here <csmsc>`.
    
Training
---------
NeMo supports training both Tacotron 2 and Waveglow. For the purposes of this
tutorial, we will be focusing on training Tacotron 2 as that determines the
majority of the characteristics of the trained audio such as the gender, and
prosody. Furthermore, in our experiments, Waveglow has been shown to work as
an unverisal vocoder. Our pretrained Waveglow, though trained on read female
English speech, can be used as vocoder for male voices as well as other languages
such as Mandarin.

Training Tacotron 2 can be done by running the `tacotron2.py` file inside 
NeMo/examples/tts. Assuming you are inside the NeMo/examples/tts directory,
you can run the following to start training:

.. code-block:: bash

    python tacotron2.py --train_dataset=<data_root>/ljspeech_train.json --eval_datasets <data_root>/ljspeech_eval.json --model_config=configs/tacotron.yaml --max_steps=30000

Training Tacotron 2 on Mandarin also can be done by running the `tacotron2.py` file.
You can run the following to start training:

.. code-block:: bash

    python tacotron2.py --train_dataset=<data_root>/databaker_csmsc_train.json --eval_datasets <data_root>/databaker_csmsc_eval.json --model_config=configs/tacotron_mandarin.yaml --max_steps=30000
    
.. tip::
    Tacotron 2 normally takes around 20,000 steps for attention to be learned.
    Once attention is learned, this is when you can use the model to generate
    audible speech.

Mixed Precision training
-------------------------
Enabling or disabling mixed precision training can be changed through a command
line argument --amp_opt_level. Recommended and default values for Tacotron 2
and Waveglow are O1. It can be:

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
torch.distributed.launch module and sepcifying the num_gpus as the
--nproc_per_node argument:

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=<num_gpus> <nemo_git_repo_root>/examples/tts/tacotron2.py ...


.. _TTS Inference:

Inference
---------
You can now to inference with either your own trained Tacotron 2, or you can
use our pre-trained Tacotron 2 model. `Please download our
pretrained model here  <https://ngc.nvidia.com/catalog/models/nvidia:tacotron2_ljspeech>`_.
Next create the texts that you want
to generate and add them to a json like the training dataset. They should
have lines like so:

.. code-block:: json

    {"audio_filepath": "", "duration": 1.0, "text": "Talk to me!"}
    {"audio_filepath": "", "duration": 1.0, "text": "Speech Synthensis is cool."}

For Mandarin, they should have lines like so:

.. code-block:: json

    {"audio_filepath": "", "duration": 1.0, "text": "jin1 tian1 tian1 qi4 bu2 cuo4."}
    {"audio_filepath": "", "duration": 1.0, "text": "ni3 kan4 bao4 zhi3 ma0"}

The "text" should contain the **pinyin** sequence in Mandarin. The digit behind each Chinese character's pinyin is the **tone**. 0 stands for soft tone.

Inference can be done with the tts_infer.py file under the
NeMo/examples/tts folder like so:

.. code-block:: bash

    python tts_infer.py --spec_model=tacotron2 --spec_model_config=configs/tacotron2.yaml --spec_model_load_dir=<directory_with_tacotron2_checkopints> --vocoder=waveglow --vocoder_model_config=configs/waveglow.yaml --vocoder_model_load_dir=<directory_with_waveglow_checkopints> --save_dir=<where_you_want_to_save_wav_files> --eval_dataset <mainfest_to_generate>

For Mandarin, remember to replace the config file of Tacotron 2 with tacotron2_mandarin.yaml.