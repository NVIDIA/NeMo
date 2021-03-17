Automatic Speech Recognition (ASR)
==================================

ASR, or Automatic Speech Recognition, refers to the problem of getting a program to automatically transcribe spoken language (speech-to-text). Our goal is usually to have a model that minimizes the Word Error Rate (WER) metric when transcribing speech input. In other words, given some audio file (e.g. a WAV file) containing speech, how do we transform this into the corresponding text with as few errors as possible?

Traditional speech recognition takes a generative approach, modeling the full pipeline of how speech sounds are produced in order to evaluate a speech sample. We would start from a language model that encapsulates the most likely orderings of words that are generated (e.g. an n-gram model), to a pronunciation model for each word in that ordering (e.g. a pronunciation table), to an acoustic model that translates those pronunciations to audio waveforms (e.g. a Gaussian Mixture Model).

Then, if we receive some spoken input, our goal would be to find the most likely sequence of text that would result in the given audio according to our generative pipeline of models. Overall, with traditional speech recognition, we try to model Pr(audio|transcript)*Pr(transcript), and take the argmax of this over possible transcripts.

Over time, neural nets advanced to the point where each component of the traditional speech recognition model could be replaced by a neural model that had better performance and that had a greater potential for generalization. For example, we could replace an n-gram model with a neural language model, and replace a pronunciation table with a neural pronunciation model, and so on. However, each of these neural models need to be trained individually on different tasks, and errors in any model in the pipeline could throw off the whole prediction.

Thus, we can see the appeal of end-to-end ASR architectures: discriminative models that simply take an audio input and give a textual output, and in which all components of the architecture are trained together towards the same goal. The model's encoder would be akin to an acoustic model for extracting speech features, which can then be directly piped to a decoder which outputs text. If desired, we could integrate a language model that would improve our predictions, as well.

And the entire end-to-end ASR model can be trained at once--a much easier pipeline to handle!

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

Hands-on speech recognition tutorial notebooks can be found under
`the ASR tutorials folder <https://github.com/NVIDIA/NeMo/tree/r1.0.0rc1/tutorials/asr/>`_.
If you are a beginner to NeMo, consider trying out the
`ASR with NeMo <https://github.com/NVIDIA/NeMo/tree/r1.0.0rc1/tutorials/asr/01_ASR_with_NeMo.ipynb>`_ tutorial.
This and most other tutorials can be run on Google Colab by specifying the link to the notebooks' GitHub pages on Colab.

If you are looking for information about a particular ASR model, or would like to find out more about the model
architectures available in the `nemo_asr` collection, check out the :doc:`Models <./models>` page.

Documentation on dataset preprocessing can be found on the :doc:`Datasets <./datasets>` page.
NeMo includes preprocessing scripts for several common ASR datasets, and this page contains instructions on running
those scripts.
It also includes guidance for creating your own NeMo-compatible dataset, if you have your own data.

Information about how to load model checkpoints (either local files or pretrained ones from NGC), as well as a list
of the checkpoints available on NGC are located on the :doc:`Checkpoints <./results>` page.

Documentation for configuration files specific to the ``nemo_asr`` models can be found on the
:doc:`Configuration Files <./configs>` page.
