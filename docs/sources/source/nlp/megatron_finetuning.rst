.. _megatron_finetuning:

Megatron-LM for Downstream Tasks
================================

Megatron :cite:`nlp-megatron-lm-shoeybi2020megatron` is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA.
More details could be found in `Megatron-LM github repo <https://github.com/NVIDIA/Megatron-LM>`_.

Requirements
------------

To use Megatron-LM models, please install NVIDIA APEX `from here <https://github.com/NVIDIA/apex>`_. 
We strongly recommend using one of NGC's recent PyTorch containers (has APEX pre-installed) or NeMo docker container with all dependencies pre-installed, \
more details could be found in the `Getting Started section <https://nvidia.github.io/NeMo/index.html#getting-started>`_.

Fine-tuning
-----------

In order to finetune a pretrained Megatron BERT language model on NLP downstream tasks from `examples/nlp  <https://github.com/NVIDIA/NeMo/tree/master/examples/nlp>`_, specify the pretrained_model_name like this: 

.. code-block:: bash

    # to used uncased model
    --pretrained_model_name megatron-bert-345m-uncased

    # to used cased model
    --pretrained_model_name megatron-bert-345m-cased

For example, to finetune SQuAD v1.1 with Megatron-LM, run:

.. code-block:: bash

    python question_answering_squad.py  \
    --train_file PATH_TO_DATA_DIR/squad/v1.1/train-v1.1.json  \
    --eval_file PATH_TO_DATA_DIR/squad/v1.1/dev-v1.1.json \
    --pretrained_model_name megatron-bert-345m-uncased


If you have a different checkpoint or model configuration, use ``--pretrained_model_name megatron-bert-uncased`` \
or ``--pretrained_model_name megatron-bert-cased`` and specify ``--bert_config`` and ``--bert_checkpoint`` for your model.

.. note::
    Megatron-LM has its own set of training arguments (including tokenizer) that are ignored during finetuning in NeMo. Please use downstream task training scripts for all NeMo supported arguments.


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