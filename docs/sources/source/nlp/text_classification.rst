.. _text_classification:

Tutorial
========

In this tutorial, we are going to describe how to finetune a BERT-like model \
based on `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`_ :cite:`nlp-tc-devlin2018bert` \
on a task classification task. 
The code used in this tutorial is based on ``examples/nlp/glue_benchmark/glue_benchmark_with_bert.py``.

Task Description
----------------

Data Format
-----------

Model training
--------------
Use ``--task_name`` argument to run the training script on a specific task, use lower cased task name: ``cola, sst-2, mrpc, sts-b, qqp, mnli, qnli, rte, wnli``.

To run the script on MRPC task on a single GPU, run:
    
    .. code-block:: bash

        python glue_benchmark_with_bert.py  \
            --data_dir /path_to_data_dir/MRPC \
            --task_name mrpc \
            --work_dir /path_to_output_folder \
            --pretrained_model_name bert-base-uncased 
            

To use multi-gpu training on MNLI task, run:

    .. code-block:: bash

        export NUM_GPUS=4
        python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS glue_benchmark_with_bert.py \
            --data_dir=/path_to_data/MNLI \
            --task_name mnli \
            --work_dir /path_to_output_folder \
            --num_gpus=$NUM_GPUS \
            --pretrained_model_name bert-base-uncased \

More details about multi-gpu training could be found in the `Fast Training <https://nvidia.github.io/NeMo/training.html>`_ section.

For additional model training parameters, please see ``examples/nlp/glue_benchmark_with_bert.py``.

.. _sentiment_analysis:

Sentiment Analysis with BERT
============================

link to the notebook


References
----------

.. bibliography:: nlp_all_refs.bib
    :style: plain
    :labelprefix: NLP-TC
    :keyprefix: nlp-tc-