Speech Classification
==================================
Speech Classification refers to a set of tasks or problems of getting a program to automatically classify input utterance or audio segment into categories, 
such as Speech Command Recognition (multi-class), Voice Activity Detection (binary or multi-class), and Audio Sentiment Classification (typically multi-class), etc.

**Speech Command Recognition** is the task of classifying an input audio pattern into a discrete set of classes. 
It is a subset of Automatic Speech Recognition, sometimes referred to as Key Word Spotting, in which a model is constantly analyzing speech patterns to detect certain "command" classes. 
Upon detection of these commands, a specific action can be taken by the system. 
It is often the objective of command recognition models to be small and efficient so that they can be deployed onto low-power sensors and remain active for long durations of time.


**Voice Activity Detection (VAD)** also known as speech activity detection or speech detection, is the task of predicting which parts of input audio contain speech versus background noise.
It is an essential first step for a variety of speech-based applications including Automatic Speech Recognition. 
It serves to determine which samples to be sent to the model and when to close the microphone.


The full documentation tree is as follows:

.. toctree::
   :maxdepth: 8

   models
   datasets
   results
   configs


Resource and Documentation Guide
--------------------------------

Hands-on speech classification tutorial notebooks can be found under ``<NeMo_git_repo>/tutorials/asr/``.
There are training and offline & online microphone inference tutorials for Speech Command Detection and Voice Activity Detection tasks.
This and most other tutorials can be run on Google Colab by specifying the link to the notebooks' GitHub pages on Colab.

If you are looking for information about a particular Speech Classification model or would like to find out more about the model
architectures available in the `nemo_asr` collection, check out the :doc:`Models <./models>` page.

Documentation on dataset preprocessing can be found on the :doc:`Datasets <./datasets>` page.
NeMo includes preprocessing scripts for several common ASR datasets, and this page contains instructions on running
those scripts.
It also includes guidance for creating your own NeMo-compatible dataset, if you have your own data.

Information about how to load model checkpoints (either local files or pretrained ones from NGC), perform inference, as well as a list
of the checkpoints available on NGC are located on the :doc:`Checkpoints <./results>` page.

Documentation for configuration files specific to the ``nemo_asr`` models can be found on the
:doc:`Configuration Files <./configs>` page.
