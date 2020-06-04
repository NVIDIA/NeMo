.. _text_classification:

Tutorial
========

In this tutorial, we are going to describe how to finetune a BERT-like model \
based on `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`_ :cite:`nlp-tc-devlin2018bert` \
on a text classification task. 

Task Description
----------------

Text classification is the task of assigning a predefined label to a given text based on its content. 
The text classification task applies to a broad range of problems: sentiment analysis, spam detection, intent detection, and many others.


Data Format
-----------

For the text classification task, NeMo requires the following format:

- the first line of each data file should contain a header with columns ``sentence`` and ``label``
- all subsequent lines in the file should contain some text in the first column and numerical label in the second column
- the columns are separated with tab

.. code-block::

    sentence [TAB] label
    text [TAB]  label_id
    text [TAB]  label_id
    text [TAB]  label_id

For example, that final data file could look like this:

.. code-block::

    sentence label
    the first sentence  0
    the second sentence 1
    the third sentence  2 

By default, the training script assumes that the training data is locating under the specified \
``--data_dir PATH_TO_DATA`` in ``train.tsv`` file, and evaluation file in ``dev.tsv`` file. 
Use ``--train_file_prefix`` and ``--eval_file_prefix`` to change the default names.

NeMo provides a conversion script from the original data format to the NeMo format \
for some of the well-known datasets including SST-2 and IMDB, see 
`examples/nlp/text_classification/data/import_datasets.py <https://github.com/NVIDIA/NeMo/blob/master/examples/nlp/text_classification/data/import_datasets.py>`_ for details.

Model training
--------------

The code used in this tutorial is based on `examples/nlp/text_classification/text_classification_with_bert.py <https://github.com/NVIDIA/NeMo/blob/master/examples/nlp/text_classification/text_classification_with_bert.py>`_.

.. note::

    The script supports multi-class tasks

To run the script on a single GPU, run:
    
    .. code-block:: bash

        python text_classification_with_bert.py  \
            --data_dir /path_to_data_dir \
            --work_dir /path_to_output_folder 
            
To use multi-gpu training on this task, run:

    .. code-block:: bash

        export NUM_GPUS=4
        python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS text_classification_with_bert.py \
            --data_dir=/path_to_data_dir \
            --work_dir /path_to_output_folder \
            --num_gpus=$NUM_GPUS 

More details about multi-gpu training could be found in the `Fast Training <https://nvidia.github.io/NeMo/training.html>`_ section.

For additional model training parameters, please see ``examples/nlp/text_classification_with_bert.py``.

Evaluating Checkpoints
----------------------

During training, the model is evaluated after every epoch and by default a folder named ``checkpoints`` would be created under the working folder specified by `--work_dir` and \
checkpoints would be stored there. To evaluate a pre-trained checkpoint on a dev set, \
run the same training script by passing ``--checkpoint_dir`` and setting ``--num_epochs`` as zero to avoid the training.

.. code-block:: bash

    python text_classification_with_bert.py  \
        --data_dir /path_to_data_dir/ \
        --work_dir /path_to_output_folder \
        --checkpoint_dir /path_to_output_folder/checkpoints \
        --num_epochs 0


.. _sentiment_analysis:

Sentiment Analysis with BERT
============================

Tutorial on how to finetune a BERT model on Sentiment Analysis task, could be found at
`examples/nlp/text_classification/sentiment_analysis_with_bert.ipynb <https://github.com/NVIDIA/NeMo/blob/master/examples/nlp/text_classification/sentiment_analysis_with_bert.ipynb>`_


References
----------

.. bibliography:: nlp_all_refs.bib
    :style: plain
    :labelprefix: NLP-TC
    :keyprefix: nlp-tc-