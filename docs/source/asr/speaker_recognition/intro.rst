Speaker Recognition (SR)
========================

Speaker Recognition (SR) is a broad research area which solves two major tasks: speaker identification (who is speaking?) and speaker verification (is the speaker who they claim to be?). 
We focus on far-field, text-independent speaker recognition when the identity of the speaker is based on how the speech is spoken, not necessarily in what is being said. 
Typically such SR systems operate on unconstrained speech utterances, which are converted into vectors of fixed length, called speaker embeddings. 
Speaker embeddings can also be used in automatic speech recognition (ASR) and speech synthesis.

As the goal of most speaker recognition systems is to get good speaker level embeddings that could help distinguish from other speakers, we shall first train these embeddings in end-to-end manner optimizing the QuatzNet based encoder model on cross-entropy loss. 
We modify the decoder to get these fixed size embeddings irrespective of the length of the input audio and employ a mean and variance based statistics pooling method to grab these embeddings.
   
In Speaker Identification we typically train on a larger training set with cross-entrophy loss and finetune later on your preferred set of labels where one would want to classify only known set of speakers. 
And in Speaker verification we train with Angular Softmax loss and compare embedings extracted from one audio file coming from a single speaker with 
embeddings extracted from another file of same or another speaker by employing backend scoring techniques like cosine similarity. 


The full documentation tree is as follows:

.. toctree::
   :maxdepth: 8

   models
   configs
   datasets
   results

Resource and Documentation Guide
--------------------------------

Hands-on speaker recognition tutorial notebooks can be found under
`the speaker recognition tutorials folder <https://github.com/NVIDIA/NeMo/tree/r1.0.0rc1/tutorials/speaker_recognition/>`_. This and most other tutorials can be run on Google Colab by specifying the link to the notebooks' GitHub pages on Colab.

If you are looking for information about a particular SpeakerNet model, or would like to find out more about the model
architectures available in the ``nemo_asr`` collection, check out the :doc:`Models <./models>` page.

Documentation on dataset preprocessing can be found on the :doc:`Datasets <./datasets>` page.
NeMo includes preprocessing and other scripts for speaker_recognition in <nemo/scripts/speaker_recognition/> folder, and this page contains instructions on running
those scripts. It also includes guidance for creating your own NeMo-compatible dataset, if you have your own data.

Information about how to load model checkpoints (either local files or pretrained ones from NGC), perform inference, as well as a list
of the checkpoints available on NGC are located on the :doc:`Checkpoints <./results>` page.

Documentation for configuration files specific to the ``nemo_asr`` models can be found on the
:doc:`Configuration Files <./configs>` page.


For a clear step-by-step tutorial we advice you to refer tutorials found in `folder <https://github.com/NVIDIA/NeMo/tree/r1.0.0rc1/tutorials/speaker_recognition/>`_.

