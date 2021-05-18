.. _megatron_finetuning:

Megatron-LM for Downstream Tasks
================================

Megatron-LM :cite:`nlp-megatron-shoeybi2019megatron` is a large, powerful transformer developed by the Applied Deep Learning Research 
team at NVIDIA. Unlike BERT, the position of the layer normalization and the residual connection in the model architecture (similar to 
GPT-2 architucture) are swapped, which allows the models to continue to improve as they were scaled up. This model reaches higher 
scores compared to BERT on a range of Natural Language Processing (NLP) tasks. More details on efficient, model-parallel and multi-node 
pre-training of GPT and BERT using mixed precision can be found in the `Megatron-LM GitHub repo <https://github.com/NVIDIA/Megatron-LM>`_.


Fine-tuning
-----------

Pre-trained Megatron-LM (BERT) can be used for most of the NLP downstream tasks from `NeMo/examples/nlp <https://github.com/NVIDIA/NeMo/tree/master/examples/nlp>`_. 
Specify the ``model.language_model.pretrained_model_name`` parameter:

.. code-block:: bash

    model.language_model.pretrained_model_name=megatron-bert-345m-uncased

Available pre-trained Megatron-LM models:

- ``megatron-bert-345m-cased``
- ``megatron-bert-345m-uncased``
- ``biomegatron-bert-345m-uncased``
- ``biomegatron-bert-345m-cased``

For example, to fine-tune SQuAD v1.1 with Megatron-LM, run:

.. code-block:: bash

    python question_answering_squad.py \
           model.train_ds.file=<TRAIN_JSON_FILE> \
           model.validation_ds=<VAL_JSON_FILE> \
           model.language_model.pretrained_model_name=megatron-bert-345m-uncased

If you have a different checkpoint or model configuration (pre-trained with `Megatron-LM GitHub repo <https://github.com/NVIDIA/Megatron-LM>`_), 
use ``model.language_model.pretrained_model_name=megatron-bert-uncased`` or ``model.language_model.pretrained_model_name=megatron-bert-cased`` 
and specify ``--bert_config`` and ``--bert_checkpoint`` for your model.

.. note::
    Megatron-LM has its own set of training arguments (including tokenizer) that are ignored during fine-tuning in NeMo. Use downstream 
    task training scripts for all NeMo supported arguments.

BioMegatron
-----------

BioMegatron has the same network architecture as the Megatron-LM, but is pretrained on a different dataset - `PubMed <https://catalog.data.gov/dataset/pubmed>`_, 
a large biomedical text corpus, which achieves better performance in biomedical downstream tasks than the original Megatron-LM.

Examples of using BioMegatron on biomedical downstream tasks can be found at (can be executed with `Google's Colab <https://colab.research.google.com/notebooks/intro.ipynb>`_): 
`NeMo/tutorials/nlp/Relation_Extraction-BioMegatron.ipynb <https://github.com/NVIDIA/NeMo/blob/main/tutorials/nlp/Relation_Extraction-BioMegatron.ipynb>`__ and `NeMo/tutorials/nlp/Token_Classification-BioMegatron.ipynb <https://github.com/NVIDIA/NeMo/blob/main/tutorials/nlp/Token_Classification-BioMegatron.ipynb>`__.

Model Parallelism
-----------------

`Megatron-LM <https://github.com/NVIDIA/Megatron-LM>`_ is a highly optimized and efficient library for training large language models.
With Megatron model parallelism, language models can be trained with billions of weights and then used in NeMo for downstream tasks.

NeMo handles pretrained model parallel checkpoints from Megatron-LM automatically and model parallel models in NeMo have the all 
the same features as other NeMo Models.

.. note::

    Currently, NeMo only supports tensor model parallelism.

Training
^^^^^^^^

All of the necessary logic to train model parallel models in NeMo with PyTorch Lightning is contained in the ``NLPDDPPlugin``. 
The ``NLPDDPPlugin`` subclasses the PyTorch Lightning training type plugin ``DDPPlugin``.
See `plugins <https://pytorch-lightning.readthedocs.io/en/latest/extensions/plugins.html>`_ for more information on PyTorch Lightning Plugins.

To enable model parallel training in NeMo:

.. code-block:: python

    trainer = Trainer(plugins=[NLPDDPPlugin()], **cfg.trainer)

Megatron-LM checkpoints have a specific format. One checkpoint is saved for each model parallel rank:

.. code-block:: bash

    iter_0080000/
    ├── mp_rank_00
    │   └── model_optim_rng.pt
    └── mp_rank_01
        └── model_optim_rng.pt


To start fine-tuning from a Megatron-LM checkpoint, simply pass the path to the Megatron-LM checkpoint 
via the language model config:

.. code-block:: bash 

    model.language_model.lm_checkpoint=/raid/megatron/bert/iter_0080000 \

We also need to input the model configuration. This can be done via json:

.. code-block:: json

    {
    "hidden-size": 1024, 
    "num-attention-heads": 16, 
    "num-layers": 24, 
    "max-seq-length": 512
    }

And input via command line:

.. code-block:: bash

    model.language_model.config_file=/raid/data/megatron/bert/config.json \

Or the model configuration can be input via YAML:

.. code-block:: YAML

    model:
        language_model:
            config:
                hidden_size: 1024
                num_attention_heads: 16
                num_layers: 24
                max_position_embeddings: 512

Additionally, Megatron-LM requires a vocab file:

.. code-block:: bash

    model.tokenizer.vocab_file=/path/to/vocab.txt

If using the Megatron-LM default tokenizer for training BERT the vocab file can be omitted:

.. code-block:: bash

    # uncased model
    model.tokenizer.tokenizer_name=megatron-bert-uncased

.. code-block:: bash

    # cased model 
    model.tokenizer.tokenizer_name=megatron-bert-uncased

Auto-Resume
^^^^^^^^^^^

Resuming training with NeMo experiment manager and PyTorch Lightning works exactly the same as other NeMo models.
While training with PTL, model parallel checkpoint will be saved and loaded properly.

.. code-block:: bash

    checkpoints/
    ├── mp_rank_00
    │   ├── mp_autoresume-last.ckpt
    │   ├── mp_autoresume---val_loss=0.35-epoch=0.ckpt
    │   ├── mp_autoresume---val_loss=0.38-epoch=1.ckpt
    │   └── mp_autoresume---val_loss=0.39-epoch=2.ckpt
    └── mp_rank_01
        ├── mp_autoresume-last.ckpt
        ├── mp_autoresume---val_loss=0.35-epoch=0.ckpt
        ├── mp_autoresume---val_loss=0.38-epoch=1.ckpt
        └── mp_autoresume---val_loss=0.39-epoch=2.ckpt

Save and Restore
^^^^^^^^^^^^^^^^

Model parallel .nemo files behave the same as all other .nemo files. Calling ``.save_to`` will save 
a checkpoint for each model parallel rank inside the .nemo file:

.. code-block:: bash

    text_class_350m
    ├── megatron-bert-uncased_encoder_config.json
    ├── megatron_checkpoint_version.json
    ├── model_config.yaml
    ├── mp_rank_00
    │   └── model_weights.ckpt
    ├── mp_rank_01
    │   └── model_weights.ckpt
    ├── tokenizer_vocab_dict.json
    └── tokenizer.vocab_file

When restoring a model parallel .nemo file, we must pass in the ``Trainer`` as model parallel requires DDP:

.. code-block:: python

    model = TokenClassificationModel.restore_from(cfg.pretrained_model, trainer=trainer)

Evaluation
^^^^^^^^^^

Since model parallel models always require more than one GPU, the ``Trainer`` is needed for evaluation:

.. code-block:: python

    trainer = pl.Trainer(plugins=[NLPDDPPlugin()], **cfg.trainer)

    model = TextClassificationModel.restore_from(cfg.model.nemo_path, trainer=trainer)
    model.setup_test_data(test_data_config=cfg.model.test_ds)

    trainer.test(model=model, ckpt_path=None)





References
----------

.. bibliography:: nlp_all.bib
    :style: plain
    :labelprefix: NLP-MEGATRON
    :keyprefix: nlp-megatron-