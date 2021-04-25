Speaker Diarization (SD)
========================

Speaker Diarization (SD) is the task of segmenting audio recordings by speaker labels, meaning, who speaks when.

.. image:: images/sd_pipeline.png
        :align: center
        :alt: Speaker Diarization pipeline [todo]
        :scale: 50%

A diarization system consists of a **Voice Activity Detection (VAD)** model to get the timestamps of audio where speech is being 
spoken while ignoring the background noise and a **Speaker Embeddings** model to get speaker embeddings on speech segments obtained 
from VAD timestamps. These speaker embeddings are then clustered into clusters based on the number of speakers present in the audio 
recording.

NeMo supports both **oracle VAD** and **non-oracle VAD** diarization. 

The full documentation tree is as follows:

.. toctree::
   :maxdepth: 8

   models
   datasets
   results
   configs
   api

Resource and Documentation Guide
--------------------------------

Hands-on speaker diarization tutorial notebooks can be found under ``<NeMo_git_root>/tutorials/speaker_recognition/``.

There are tutorials for peformming inference using :ref:`MarbleNet_model` and :ref:`SpeakerNet_model`. These models also show how 
you can get ASR transcriptions combined with Speaker labels along with voice activity timestamps with NeMo ASR collections.

Most of the tutorials can be run on Google Colab by specifying the link to the notebooks' GitHub pages on Colab.

If you are looking for information about a particular model used for speaker diarization inference, or would like to find out more 
about the model architectures available in the ``nemo_asr`` collection, refer to the :doc:`Models <./models>` section.

Documentation on dataset preprocessing can be found in the :doc:`Datasets <./datasets>` section. NeMo includes preprocessing scripts 
for several common ASR datasets. This section contains instructions on running those scripts as well as guidance for creating your 
own NeMo-compatible dataset, if you have your own data.

Information about how to load model checkpoints (either local files or pretrained ones from NGC), perform inference, as well as a list
of the checkpoints available on NGC are located in the :doc:`Checkpoints <./results>` section.

Documentation for configuration files specific to the ``nemo_asr`` models can be found in the
:doc:`Configuration Files <./configs>` section.
