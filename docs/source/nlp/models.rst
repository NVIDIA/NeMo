Models
======

NeMo's NLP collection supports the following models:

* BERT pretraining
* GLUE Benchmark
* Joint Intent and Slot Classification
* Text Classification
* Token Classification (Name Entity Recognition (NER))
    * :ref:`ner`
* Punctuation and Capitalization
    * :ref:`punctuation_and_capitalization`
* Question Answering
* Information Retrieval

Scripts for running these models, could be found under ``NeMo/example/nlp/``.
NLP tutorials are located under ``NeMo/tutorials/nlp/``.


All examples from NLP collection can be found under `NeMo/example/nlp/ <https://github.com/NVIDIA/NeMo/tree/main/examples/nlp>`__.
NLP notebook-tutorials are located under `NeMo/example/nlp/ <https://github.com/NVIDIA/NeMo/tree/main/tutorials/nlp>`__
(most notebooks work in Colab).

If you're just starting with NeMo, the best way to start is to take a look at:

* `NeMo Primer <https://github.com/NVIDIA/NeMo/blob/main/tutorials/00_NeMo_Primer.ipynb>`__ \
- introduces NeMo, PyTorch Lightning and OmegaConf, and shows how to use, modify, save, and restore NeMo Models
* `NeMo Models <https://github.com/NVIDIA/NeMo/blob/main/tutorials/01_NeMo_Models.ipynb>`__ - fundamental concepts of the NeMo Model
* `NeMo voice swap demo <https://github.com/NVIDIA/NeMo/blob/main/tutorials/NeMo_voice_swap_app.ipynb`__ - demonstrates how to
swap a voice in the audio fragment with a computer generated one using NeMo.


Pretraining BERT
-----------------
.. toctree::
   :maxdepth: 8

   bert_pretraining


Megatron-LM for Downstream tasks
--------------------------------
.. toctree::
   :maxdepth: 8

   megatron_finetuning

GLUE Benchmark
--------------
.. toctree::
   :maxdepth: 8

   glue

Intent and Slot filling
-----------------------
.. toctree::
   :maxdepth: 8

   .. joint_intent_slot_filling

Text Classification
-------------------
.. toctree::
   :maxdepth: 8

   .. text_classification

Token Classification (Named Entity Recognition)
-----------------------------------------------

.. toctree::
   :maxdepth: 8

   token_classification

Punctuation and Word Capitalization
-----------------------------------

.. toctree::
   :maxdepth: 8

   punctuation_capitalization

Question Answering
------------------
.. toctree::
   :maxdepth: 8

    .. question_answering



Dialogue State Tracking
-----------------------

.. toctree::
   :maxdepth: 8

