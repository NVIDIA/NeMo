Megatron-LM for Downstream Tasks
================================

Megatron :cite:`nlp-megatron-lm-shoeybi2020megatron` is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA.
More details could be found in `Megatron-LM github repo <https://github.com/NVIDIA/Megatron-LM>`_.

In order to finetune a pretrained Megatron BERT language model on NLP downstream tasks from `examples/nlp  <https://github.com/NVIDIA/NeMo/tree/master/examples/nlp>`_, specify the pretrained_model_name like this: 

.. code-block:: bash

    --pretrained_model_name megatron-bert-345m-uncased

For example, to finetune SQuAD v1.1 with Megatron-LM, run:

.. code-block:: bash

    python question_answering_squad.py  \
    --train_file PATH_TO_DATA_DIR/squad/v1.1/train-v1.1.json  \
    --eval_file PATH_TO_DATA_DIR/squad/v1.1/dev-v1.1.json \
    --pretrained_model_name megatron-bert-345m-uncased


If you have a different checkpoint or model configuration, use ``--pretrained_model_name megatron-bert-uncased`` or ``--pretrained_model_name megatron-bert-cased`` and specify ``--bert_config`` and ``--bert_checkpoint`` for your model.

.. note::
    Megatron-LM has its own set of training arguments (including tokenizer) that are ignored during finetuning in NeMo. Please use downstream task training scripts for all NeMo supported arguments.



References
----------

.. bibliography:: nlp_all_refs.bib
    :style: plain
    :labelprefix: NLP-MEGATRON-LM
    :keyprefix: nlp-megatron-lm-