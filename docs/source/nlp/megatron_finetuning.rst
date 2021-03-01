.. _megatron_finetuning:

Megatron-LM for Downstream Tasks
================================

TODO: update based on the recent NeMo

Megatron :cite:`nlp-megatron-lm-shoeybi2020megatron` is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA.
More details could be found in `Megatron-LM github repo <https://github.com/NVIDIA/Megatron-LM>`_.

Requirements
------------

To use Megatron-LM models, please install NVIDIA APEX `from here <https://github.com/NVIDIA/apex>`_. 
We strongly recommend using one of NGC's recent PyTorch containers (has APEX pre-installed) or NeMo docker container with all dependencies pre-installed, \
more details could be found in the `Getting Started section <https://nvidia.github.io/NeMo/index.html#getting-started>`_.

Download pretrained models
--------------------------

Original and domain-specific Megatron-LM BERT models and model configuration files can be downloaded at following links.

Megatron-LM BERT 345M (~345M parameters):
`https://ngc.nvidia.com/catalog/models/nvidia:megatron_bert_345m <https://ngc.nvidia.com/catalog/models/nvidia:megatron_bert_345m>`__

BioMegatron-LM BERT Cased 345M (~345M parameters):
`https://ngc.nvidia.com/catalog/models/nvidia:biomegatron345mcased <https://ngc.nvidia.com/catalog/models/nvidia:biomegatron345mcased>`__

BioMegatron-LM BERT Uncased 345M (~345M parameters):
`https://ngc.nvidia.com/catalog/models/nvidia:biomegatron345muncased <https://ngc.nvidia.com/catalog/models/nvidia:biomegatron345muncased>`__


Fine-tuning
-----------

In order to fine-tune a pretrained Megatron BERT language model on NLP downstream tasks from `examples/nlp  <https://github.com/NVIDIA/NeMo/tree/master/examples/nlp>`_, specify the `model.language_model.pretrained_model_name` like this:

.. code-block:: bash

    # to used uncased model
    model.language_model.pretrained_model_name=megatron-bert-345m-uncased

    # to used cased model
    model.language_model.pretrained_model_name=megatron-bert-345m-cased

For example, to fine-tune SQuAD v1.1 with Megatron-LM, run:

.. code-block:: bash

    python question_answering_squad.py \
           model.train_ds.file=<TRAIN_JSON_FILE> \
           model.validation_ds=<VAL_JSON_FILE> \
           model.language_model.pretrained_model_name=megatron-bert-345m-uncased


If you have a different checkpoint or model configuration, use ``model.language_model.pretrained_model_name=megatron-bert-uncased`` \
or ``model.language_model.pretrained_model_name=megatron-bert-cased`` and specify ``--bert_config`` and ``--bert_checkpoint`` for your model.

.. note::
    Megatron-LM has its own set of training arguments (including tokenizer) that are ignored during fine-tuning in NeMo. Please use downstream task training scripts for all NeMo supported arguments.


BioMegatron
--------------

To use BioMegatron for biomedical downstream tasks please visit:

`https://github.com/NVIDIA/NeMo/blob/master/examples/nlp/biobert_notebooks <https://github.com/NVIDIA/NeMo/blob/master/examples/nlp/biobert_notebooks>`__


References
----------

.. bibliography:: nlp_all_refs.bib
    :style: plain
    :labelprefix: NLP-MEGATRON-LM
    :keyprefix: nlp-megatron-lm-