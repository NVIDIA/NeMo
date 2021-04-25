Natural Language Processing (NLP)
=================================

Significant advances in the NLP field have been made over the past few years with most of the advances sharing one common thread: 
dramatically larger models trained on more data. BERT-large, for example, has 340 million parameters and GPT-2 has 1.5 billion 
parameters. Models of this size make inference tasks on a CPU impractical today, necessitating a scalable inference framework for 
NLP tasks on a GPU.

.. toctree::
   :maxdepth: 2

   models
   megatron_finetuning
   api

Example scripts for the NLP collection can be found in `NeMo/example/nlp/ <https://github.com/NVIDIA/NeMo/tree/main/examples/nlp>`__.

NLP notebook-tutorials are located in `NeMo/tutorials/nlp/ <https://github.com/NVIDIA/NeMo/tree/main/tutorials/nlp>`__. Most NeMo 
tutorials can be run on `Google's Colab <https://colab.research.google.com/notebooks/intro.ipynb>`_.), with more details in the :ref:`tutorials` section.
