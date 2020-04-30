Megatron-LM for Donwstream Tasks
================================

Megatron is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA.
More details could be found `here`_.
.. _here: https://github.com/NVIDIA/Megatron-LM

Most of the NLP dowstream tasks currently support finetuning with Pretrained Megatron language model. 
In order to use Megatron language model, follow the steps below:

1. Download pretrained Megatron checkpoint as described `here`_
specify --pretrained-model-name (currently, only megatron-uncased is supported)
2. Download config file. It determines model architecture: number of hidden layers, number of attention heads, etc
3. Download vocabulary file used for model training

.. _here: https://github.com/NVIDIA/Megatron-LM#downloading-checkpoints

Note that Megatron-LM has its own set of arguments, but training is done with Neural factory in Nemo, so all Megatron training arguments
are ignored. Please use donwstream task training scripts for all Nemo supported arguments.

To run training with Megatron-LM:

.. code-block:: bash

    python examples/nlp/token_classification/punctuation_capitalization_infer.py --punct_labels_dict path_to_data/punct_label_ids.csv --capit_labels_dict path_to_data/capit_label_ids.csv --checkpoint_dir path_to_output_dir/checkpoints/

Note, punct_label_ids.csv and capit_label_ids.csv files will be generated during training and stored in the data_dir folder.

Multi GPU Training
------------------

To run training on multiple GPUs, run

.. code-block:: bash

    export NUM_GPUS=2
    python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS examples/nlp/token_classification/token_classification.py --num_gpus $NUM_GPUS \
    --data_dir path_to_data \
    --pretrained_model_name megatron-uncased \
    --bert_config PATH_TO_MEGATRON_CONFIG/config.json \
    --bert_checkpoint PATH_TO_CHECKPOINT/model.pt \
    --vocab_file PATH_TO_VOCAB/vocab.txt \
    --do_lower_case 
