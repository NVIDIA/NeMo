Megatron-LM for Downstream Tasks
================================

Megatron :cite:`nlp-megatron-lm-shoeybi2020megatron` is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA.
More details could be found in `Megatron-LM github repo <https://github.com/NVIDIA/Megatron-LM>`_.

Most of the NLP downstream tasks currently support fine-tuning with Pretrained Megatron language model. 
In order to use Megatron language model, follow the steps below:

1. Download pretrained Megatron checkpoint as described `here <https://github.com/NVIDIA/Megatron-LM#downloading-checkpoints>`_
2. Download `configuration file <https://drive.google.com/file/d/123zDhg38Aat3gIFfX-ptpCAIOsGgSqr2/view?usp=sharing>`_. It determines model architecture: number of hidden layers, number of attention heads, etc
3. Download `vocabulary file <https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt>`_ used for model training

.. note::
    Megatron-LM has its own set of arguments, but training is done with Neural factory in NeMo, so all Megatron-LM training arguments
    are ignored. Please use downstream task training scripts for all NeMo supported arguments.

To finetune SQuAD v1.1 with Megatron-LM, run:

.. code-block:: bash

    python question_answering_squad.py  \
    --train_file PATH_TO_DATA_DIR/squad/v1.1/train-v1.1.json  \
    --eval_file PATH_TO_DATA_DIR/squad/v1.1/dev-v1.1.json \
    --pretrained_model_name megatron-uncased \
    --bert_config PATH_TO_MEGATRON_CONFIG/config.json \
    --bert_checkpoint PATH_TO_CHECKPOINT/model.pt \
    --vocab_file PATH_TO_VOCAB/vocab.txt \
    --do_lower_case 

References
----------

.. bibliography:: nlp_all_refs.bib
    :style: plain
    :labelprefix: NLP-MEGATRON-LM
    :keyprefix: nlp-megatron-lm-