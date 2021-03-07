Conversational AI
=================

`NVIDIA NeMo <https://github.com/NVIDIA/NeMo>`_ is a toolkit for building new State-of-the-Art 
Conversational AI models. NeMo has separate collections for Automatic Speech Recognition (ASR), 
Natural Language Processing (NLP), and Text-to-Speech (TTS) models. Each collection consists of 
prebuilt modules that include everything needed to train on your data. 
Every module can easily be customized, extended, and composed to create new Conversational AI 
model architectures.

Conversational AI architectures are typically very large and require a lot of data  and compute 
for training. NeMo uses PyTorch Lightning for easy and performant multi-GPU/multi-node 
mixed-precision training. 


NeMo Models
-----------
NeMo Models contain everything needed to train and reproduce state of the art Conversational AI
research and applications, including:

- neural network architectures 
- datasets/data loaders
- data preprocessing/postprocessing
- data augmentors
- optimizers and schedulers
- tokenizers
- language models

NeMo uses `Hydra <https://hydra.cc/>`_ for configuring both NeMo models and the PyTorch Lightning Trainer.
Depending on the domain and application, many different AI libraries will have to be configured
to build the application. Hydra makes it easy to bring all of these libraries together
so that each can be configured from .yaml or the Hydra CLI.

.. note:: Every NeMo model has an example configuration file and a corresponding script that contains all configurations needed for training.

The end result of using NeMo, Pytorch Lightning, and Hydra is that
NeMo models all have the same look and feel. This makes it easy to do Conversational AI research
across multiple domains. NeMo models are also fully compatible with the PyTorch ecosystem.

Model Configuration
-------------------
Hydra is an open-source Python framework that simplifies configuration for complex applications
that must bring together many different software libraries. Conversational AI is great examples of such an application.
To build a Conversational AI application, we must be able to configure the neural network architectures, training and optimization algorithms, 
data pre/post processing, data augmentation, experiment logging/visualization, and model checkpointing.   

With Hydra we can configure everything needed for NeMo with three interfaces: CLI, YAML files, and Python Dataclasses.

NeMo provides YAML configuration files for all of our `example <https://github.com/NVIDIA/NeMo/tree/r1.0.0rc1/examples>`_ training scripts.
YAML files make it easy to experiment with different model and training configurations.

Every NeMo example YAML has the same underlying configuration structure:

.. code-block:: yaml

    # PyTorch Lightning Trainer configuration
    # any argument of the Trainer object can be set here
    trainer:
        gpus: 1 # number of gpus per node
        num_nodes: 1 # number of nodes
        max_epochs: 10 # how many training epochs to run
        val_check_interval: 1.0 # run validation after every epoch

    # Experiment logging configuration
    exp_manager:
        exp_dir: /path/to/my/nemo/experiments
        name: name_of_my_experiment
        create_tensorboard_logger: True
        create_wandb_logger: True

    # Model configuration
    # model network architecture, train/val/test datasets, data augmentation, and optimization
    model:
        train_ds:
            manifest_filepath: /path/to/my/train/manifest
            batch_size: 256
            shuffle: True
        validation_ds:
            manifest_filepath: /path/to/my/validation/manifest
            batch_size: 32
            shuffle: False
        test_ds:
            manifest_filepath: /path/to/my/test/manifest
            batch_size: 32
            shuffle: False
        optim:
            name: novograd
            lr: .01
            betas: [0.8, 0.5]
            weight_decary: 0.001
        # network architecture can vary greatly depending on the domain
        encoder:
            ...
        decoder:
            ...
        

Dataclasses allow NeMo to ship model configurations as part of the NeMo library and 
also enables pure Python configuration of NeMo models.

As an example, see the code block below for an Attenion is All You Need machine translation model:

.. code-block:: Python

    from nemo.collections.nlp.models.machine_translation.mt_enc_dec_config import AAYNBaseConfig

    cfg = AAYNBaseConfig()

    # modify the number of layers in the encoder
    cfg.encoder.num_layers = 8

    # modify the training batch size
    cfg.train_ds.tokens_in_batch = 8192





.. note:: Configuration with Hydra always has the following precedence CLI > YAML > Dataclass

Please see the `Hydra Tutorials <https://hydra.cc/docs/tutorials/intro>`_ for a great introduction to using Hydra.

PyTorch Lightning
-----------------


Experiment Manager
------------------
NeMo's Experiment Manager leverages PyTorch Lightning for model checkpointing, 
TensorBoard Logging, and Weights and Biases logging. The Experiment Manager is included by default
in all NeMo example scripts.

.. code-block:: python

    exp_manager(trainer, cfg.get("exp_manager", None))

And is configurable via .yaml with Hydra.

.. code-block:: bash

    exp_manager:
        exp_dir: null
        name: *name
        create_tensorboard_logger: True
        create_checkpoint_callback: True

Optionally launch Tensorboard to view training results in ./nemo_experiments (by default).

.. code-block:: bash

    tensorboard --bind_all --logdir nemo_experiments

..
    TODO: add auto resume docs here

Neural Module
-------------
Neural Modules are building blocks for Models.
They accept (typed) inputs and return (typed) outputs. *All Neural Modules inherit from ``torch.nn.Module`` and, therefore, compatible with PyTorch ecosystem.* There are 3 types on Neural Modules:

    - Regular modules
    - Dataset/IterableDataset
    - Losses

Neural Types
------------
Neural Types perform semantic checks for modules and models inputs/outputs. They contain information about:

    - Semantics of what is stored in the tensors. For example, logits, logprobs, audiosignal, embeddings, etc.
    - Axes layout, semantic and (optionally) dimensionality. For example: [Batch, Time, Channel]