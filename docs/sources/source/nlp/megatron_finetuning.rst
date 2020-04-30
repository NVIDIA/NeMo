Megatron-LM for Downstream Tasks
================================

Megatron :cite:`nlp-megatron-lm-shoeybi2020megatron` is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA.
More details could be found in `Megatron-LM github repo <https://github.com/NVIDIA/Megatron-LM>`_.

Most of the NLP downstream tasks currently support fine-tuning with Pretrained Megatron language model. 
In order to use Megatron language model, follow the steps below:
1. Download pretrained Megatron checkpoint as described `here <https://github.com/NVIDIA/Megatron-LM#downloading-checkpoints>`_
2. Download configuration file. It determines model architecture: number of hidden layers, number of attention heads, etc
3. Download vocabulary file used for model training

.. note::
    Megatron-LM has its own set of arguments, but training is done with Neural factory in Nemo, so all Megatron-LM training arguments
    are ignored. Please use downstream task training scripts for all NeMo supported arguments.

To run Multi GPU training with Megatron-LM, run:

.. code-block:: bash

    export NUM_GPUS=2
    python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS examples/nlp/token_classification/token_classification.py --num_gpus $NUM_GPUS \
    --data_dir path_to_data \
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