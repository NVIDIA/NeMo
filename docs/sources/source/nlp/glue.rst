.. _glue:

Tutorial
========

In this tutorial, we are going to describe how to finetune a BERT-like model based on `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`_ :cite:`nlp-glue-devlin2018bert` on `GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding <https://openreview.net/pdf?id=rJ4km2R5t7>`_ :cite:`nlp-glue-wang2018glue`. 
The code used in this tutorial is based on ``examples/nlp/glue_benchmark/glue_benchmark_with_bert.py``.

GLUE tasks description
----------------------

GLUE Benchmark includes 9 natural language understanding tasks:

Single-Sentence Tasks:

- **CoLA** The Corpus of Linguistic Acceptability :cite:`nlp-glue-warstadt2018neural` is a set of English sentences from published linguistics literature. The task is to predict whether a given sentence is grammatically correct or not.
- **SST-2** The Stanford Sentiment Treebank :cite:`nlp-glue-socher2013recursive` consists of sentences from movie reviews and human annotations of their sentiment. The task is to predict the sentiment of a given sentence: positive or negative.


Similarity and Paraphrase tasks:

- **MRPC** The Microsoft Research Paraphrase Corpus :cite:`nlp-glue-dolan-brockett-2005-automatically` is a corpus of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent.
- **QQP** `The Quora Question Pairs2 <https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs>`_ dataset is a collection of question pairs from the community question-answering website Quora. The task is to determine whether a pair of questions are semantically equivalent. 
- **STS-B** The Semantic Textual Similarity Benchmark :cite:`nlp-glue-cer2017semeval` is a collection of sentence pairs drawn from news headlines, video, and image captions, and natural language inference data. The task is to determine how similar two sentences are.


Inference Tasks:

- **MNLI** The Multi-Genre Natural Language Inference Corpus :cite:`nlp-glue-williams2017broad` is a crowdsourced collection of sentence pairs with textual entailment annotations. Given a premise sentence and a hypothesis sentence, the task is to predict whether the premise entails the hypothesis (entailment), contradicts the hypothesis (contradiction), or neither (neutral).  The task has the matched (in-domain) and mismatched (cross-domain) sections.
- **QNLI** The Stanford Question Answering Dataset (:cite: `nlp-glue-rajpurkar2016squad`) is a question-answering dataset consisting of question-paragraph pairs, where one of the sentences in the paragraph (drawn from Wikipedia) contains the answer to the corresponding question. The task is to determine whether the context sentence contains the answer to the question.
- **RTE** The Recognizing Textual Entailment (RTE) datasets come from a series of annual textual entailment challenges. The task is to determine whether the second sentence is the entailment of the first one or not.
- **WNLI** The Winograd Schema Challenge :cite:`nlp-glue-levesque2012winograd` is a reading comprehension task in which a system must read a sentence with a pronoun and select the referent of that pronoun from a list of choices.

All the tasks are classification tasks, except for the STS-B task which is a regression task.
All classification tasks are 2-class tasks, except for the MNLI task which is a 3-class task.

More details about GLUE benchmark could be found `here <https://gluebenchmark.com/tasks>`_.

Training the model
------------------
Before running ``examples/nlp/glue_benchmark/glue_benchmark_with_bert.py``, download the GLUE data with `this script <https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e>`_ by running:

.. code-block:: bash

    # download the script to get the GLUE data
    wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py
    # run the script to download the GLUE data
    python download_glue_data.py

After running the above commands, you will have a folder ``glue_data`` with data folders for every GLUE task. For example, data for MRPC task would be under ``glue_data/MRPC``.

The GLUE tasks can be fine-tuned on 4 pre-trained back-bone models supported in NeMo: Megatron-LM BERT, BERT, AlBERT and RoBERTa.
See the list of available pre-trained Huggingface models `here <https://huggingface.co/transformers/pretrained_models.html>`__. 
To get the list of all NeMo supported pre-trained models run:

.. code-block:: python
    
    import nemo.collections.nlp as nemo_nlp
    nemo_nlp.nm.trainables.get_pretrained_lm_models_list()

Specify the model to use for training with ``--pretrained_model_name``.

.. note::
    It's recommended to finetune the model on each task separately.
    Also, based on `GLUE Benchmark FAQ#12 <https://gluebenchmark.com/faq>`_,
    there are might be some differences in dev/test distributions for QQP task
    and in train/dev for WNLI task.

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
            --data_dir=/path_to_data_dir/MNLI \
            --task_name mnli \
            --work_dir /path_to_output_folder \
            --num_gpus=$NUM_GPUS \
            --pretrained_model_name bert-base-uncased \

More details about multi-gpu training could be found in the `Fast Training <https://nvidia.github.io/NeMo/training.html>`_ section.

For additional model training parameters, please see ``examples/nlp/glue_benchmark_with_bert.py``.

Model results
-------------

Results after finetuning on the specific task (average result after 3 runs) using different pre-trained models:
 
 .. code-block:: python
    
    # to reproduce BERT base paper results
    --pretrained_model_name bert-base-uncased 

    # Albert-large
    --pretrained_model_name albert-large-v2

    # Albert-xlarge
    --pretrained_model_name albert-xlarge-v2

    # Megatron
    --pretrained_model_name megatron-bert-345m-uncased

+-------+--------------------------+--------------+---------------+---------------+-----------------+------------------+
| Task  |         Metric           | Albert-large | Albert-xlarge | Megatron-345m | BERT base paper | BERT large paper |
+=======+==========================+==============+===============+===============+=================+==================+
| CoLA  | Matthew's correlation    |     54.94    |     61.72     |     64.56     |      52.1       |       60.5       |
+-------+--------------------------+--------------+---------------+---------------+-----------------+------------------+
| SST-2 | Accuracy                 |     92.74    |     91.86     |     95.87     |      93.5       |       94.9       |
+-------+--------------------------+--------------+---------------+---------------+-----------------+------------------+
| MRPC  | F1/Accuracy              |  92.05/88.97 |  91.87/88.61  |  92.36/89.46  |      88.9/-     |     89.3/-       |
+-------+--------------------------+--------------+---------------+---------------+-----------------+------------------+
| STS-B | Person/Spearman corr.    |  90.41/90.21 |  90.07/90.10  |  91.51/91.61  |     -/85.8      |      -/86.5      |
+-------+--------------------------+--------------+---------------+---------------+-----------------+------------------+
| QQP   | F1/Accuracy              |  88.26/91.26 |  88.80/91.65  |  89.18/91.91  |     71.2/-      |     72.1/-       |
+-------+--------------------------+--------------+---------------+---------------+-----------------+------------------+
| MNLI  | Matched /Mismatched acc. |  86.69/86.81 |  88.66/88.73  |  89.86/89.81  |    84.6/83.4    |     86.7/85.9    |
+-------+--------------------------+--------------+---------------+---------------+-----------------+------------------+
| QNLI  | Accuracy                 |     92.68    |     93.66     |     94.33     |      90.5       |       92.7       |
+-------+--------------------------+--------------+---------------+---------------+-----------------+------------------+ 
| RTE   | Accuracy                 |     80.87    |     82.86     |     83.39     |      66.4       |       70.1       |
+-------+--------------------------+--------------+---------------+---------------+-----------------+------------------+

WNLI task was excluded from the experiments due to the problematic WNLI set.
The dev sets were used for evaluation for Albert and Megatron models, and the test sets results for the BERT paper from :cite:`nlp-glue-devlin2018bert`.

Hyperparameters used to get the results from the above table, could be found in the table below.
Each cell in the table represents the following parameters:
Number of GPUs used/ Batch Size/ Learning Rate/ Number of Epochs. For not specified parameters, please refer to the default parameters in the training script.

+-------+--------------+---------------+---------------+
| Task  | Albert-large | Albert-xlarge | Megatron-345m |
+=======+==============+===============+===============+
| CoLA  | 1/32/1e-5/3  |  1/32/1e-5/10 |  4/16/2e-5/12 |
+-------+--------------+---------------+---------------+
| SST-2 | 4/16/2e-5/5  |  4/16/2e-5/12 |  4/16/2e-5/12 |
+-------+--------------+---------------+---------------+
| MRPC  |  1/32/1e-5/5 |  1/16/2e-5/5  |  1/16/2e-5/10 |
+-------+--------------+---------------+---------------+
| STS-B | 1/16/2e-5/5  |  1/16/4e-5/12 |  4/16/3e-5/12 |
+-------+--------------+---------------+---------------+
| QQP   |  1/16/2e-5/5 | 4/16/1e-5/12  |  4/16/1e-5/12 |
+-------+--------------+---------------+---------------+
| MNLI  |  4/64/1e-5/5 |  4/32/1e-5/5  |  4/32/1e-5/5  | 
+-------+--------------+---------------+---------------+
| QNLI  | 4/16/1e-5/5  |  4/16/1e-5/5  |  4/16/2e-5/5  | 
+-------+--------------+---------------+---------------+
| RTE   | 1/16/1e-5/5  | 1//16/1e-5/12 |  4/16/3e-5/12 |
+-------+--------------+---------------+---------------+

Evaluating Checkpoints
----------------------

During training, the model is evaluated after every epoch and by default a folder named "checkpoints" would be created under the working folder specified by `--work_dir` and \
checkpoints would be stored there. To do evaluation of a pre-trained checkpoint on a dev set, \
run the same training script by passing `--checkpoint_dir` and setting `--num_epochs` as zero to avoid the training.
For example, to evaluate a checkpoint trained on MRPC task, run:

.. code-block:: bash

    cd examples/nlp/glue_benchmark
    python glue_benchmark_with_bert.py  \
        --data_dir /path_to_data_dir/MRPC \
        --task_name mrpc \
        --work_dir /path_to_output_folder \
        --pretrained_model_name bert-base-uncased \
        --checkpoint_dir /path_to_output_folder/checkpoints \
        --num_epochs 0

References
----------

.. bibliography:: nlp_all_refs.bib
    :style: plain
    :labelprefix: NLP-GLUE
    :keyprefix: nlp-glue-