.. _megatron_finetuning:

Megatron-LM for Downstream Tasks
================================

Megatron-LM :cite:`nlp-megatron-shoeybi2019megatron` is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA.
More details could be found in `Megatron-LM github repo <https://github.com/NVIDIA/Megatron-LM>`_.


Fine-tuning
-----------

In order to fine-tune a pretrained Megatron-BERT language model on NLP downstream tasks from `NeMo/examples/nlp <https://github.com/NVIDIA/NeMo/tree/master/examples/nlp>`_, specify the ``model.language_model.pretrained_model_name`` parameter as follows:

.. code-block:: bash

    model.language_model.pretrained_model_name=megatron-bert-345m-uncased

Available pre-trained Megatron-LM models:

* `megatron-bert-345m-cased`
* `megatron-bert-345m-uncased`
* `biomegatron-bert-345m-uncased`
* `biomegatron-bert-345m-cased`

For example, to fine-tune SQuAD v1.1 with Megatron-LM, run:

.. code-block:: bash

    python question_answering_squad.py \
           model.train_ds.file=<TRAIN_JSON_FILE> \
           model.validation_ds=<VAL_JSON_FILE> \
           model.language_model.pretrained_model_name=megatron-bert-345m-uncased


If you have a different checkpoint or model configuration (pre-trained with `Megatron-LM github repo <https://github.com/NVIDIA/Megatron-LM>`_), use ``model.language_model.pretrained_model_name=megatron-bert-uncased`` \
or ``model.language_model.pretrained_model_name=megatron-bert-cased`` and specify ``--bert_config`` and ``--bert_checkpoint`` for your model.

.. note::
    Megatron-LM has its own set of training arguments (including tokenizer) that are ignored during fine-tuning in NeMo. Please use downstream task training scripts for all NeMo supported arguments.


BioMegatron
--------------

Examples of using BioMegatron on biomedical downstream tasks could be found at:
`NeMo/tutorials/nlp/Relation_Extraction-BioMegatron.ipynb <https://github.com/NVIDIA/NeMo/blob/main/tutorials/nlp/Relation_Extraction-BioMegatron.ipynb>`__ and `NeMo/tutorials/nlp/Token_Classification-BioMegatron.ipynb <https://github.com/NVIDIA/NeMo/blob/main/tutorials/nlp/Token_Classification-BioMegatron.ipynb>`__
(can be executed with `Google's Colab <https://colab.research.google.com/notebooks/intro.ipynb>`_).


References
----------

.. bibliography:: nlp_all.bib
    :style: plain
    :labelprefix: NLP-MEGATRON
    :keyprefix: nlp-megatron-