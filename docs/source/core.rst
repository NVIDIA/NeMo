NeMo Basics
===========


NeMo Models
-----------

NeMo Models contain everything needed to train and reproduce Conversational AI model:

- neural network architectures 
- datasets/data loaders
- data preprocessing/postprocessing
- data augmentors
- optimizers and schedulers
- tokenizers
- language models

NeMo uses `Hydra <https://hydra.cc/>`_ for configuring both NeMo models and the PyTorch Lightning Trainer.

.. note:: Every NeMo model has an example configuration file and training script that can be found `here <https://github.com/NVIDIA/NeMo/tree/r1.0.0rc1/examples>`_.

The end result of using NeMo, Pytorch Lightning, and Hydra is that
NeMo models all have the same look and feel and are also fully compatible with the PyTorch ecosystem. 


Model Training
--------------
NeMo leverages `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ for model training.
PyTorch Lightning lets NeMo decouple the Conversational AI code from the PyTorch training code. 
This means that NeMo users can focus on their domain (ASR, NLP, TTS) and building complex AI applications
without having to rewrite boiler plate code for PyTorch training.

When using PyTorch Lightning, NeMo users can automatically train with:

- multi-GPU/multi-node
- mixed precision
- model checkpointing
- logging
- early stopping
- and more

The two main aspects of the Lightning API are the `LightningModule <https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#>`_ 
and the `Trainer <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_.

PyTorch Lightning LightningModule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every NeMo model is a ``LightningModule`` which is an ``nn.module``. 
This means that NeMo models are compatible with the PyTorch ecosystem and
can be plugged into existing PyTorch workflows.

Creating a NeMo Model is similar to any other PyTorch workflow.
We start by initializing our model architecture and then define the forward pass:

.. code-block:: python

    class TextClassificationModel(NLPModel, Exportable):
        ...
        def __init__(self, cfg: DictConfig, trainer: Trainer = None):
            """Initializes the BERTTextClassifier model."""
            ...
            super().__init__(cfg=cfg, trainer=trainer)

            # instantiate a BERT based encoder
            self.bert_model = get_lm_model(
                pretrained_model_name=cfg.language_model.pretrained_model_name,
                config_file=cfg.language_model.config_file,
                config_dict=cfg.language_model.config,
                checkpoint_file=cfg.language_model.lm_checkpoint,
                vocab_file=cfg.tokenizer.vocab_file,
            )

            # instantiate the FFN for classification
            self.classifier = SequenceClassifier(
                hidden_size=self.bert_model.config.hidden_size,
                num_classes=cfg.dataset.num_classes,
                num_layers=cfg.classifier_head.num_output_layers,
                activation='relu',
                log_softmax=False,
                dropout=cfg.classifier_head.fc_dropout,
                use_transformer_init=True,
                idx_conditioned_on=0,
            )

.. code-block:: python

        def forward(self, input_ids, token_type_ids, attention_mask):
            """
            No special modification required for Lightning, define it as you normally would
            in the `nn.Module` in vanilla PyTorch.
            """
            hidden_states = self.bert_model(
                input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
            )
            logits = self.classifier(hidden_states=hidden_states)
            return logits


The LightningModule organizes PyTorch code so that across all NeMo models we have a similar look and feel.
For example, the training logic can be found in ``training_step``:

.. code-block:: python

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        input_ids, input_type_ids, input_mask, labels = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        train_loss = self.loss(logits=logits, labels=labels)

        lr = self._optimizer.param_groups[0]['lr']

        self.log('train_loss', train_loss)
        self.log('lr', lr, prog_bar=True)

        return {
            'loss': train_loss,
            'lr': lr,
        }

While validation logic can be found in ``validation_step``:

.. code-block:: python

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        if self.testing:
            prefix = 'test'
        else:
            prefix = 'val'

        input_ids, input_type_ids, input_mask, labels = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        val_loss = self.loss(logits=logits, labels=labels)

        preds = torch.argmax(logits, axis=-1)

        tp, fn, fp, _ = self.classification_report(preds, labels)

        return {'val_loss': val_loss, 'tp': tp, 'fn': fn, 'fp': fp}

PyTorch Lightning then handles all of the boiler plate code needed for training.
Virtually any aspect of training can be customized via PyTorch Lightning `hooks <https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#hooks>`_, 
`Plugins <https://pytorch-lightning.readthedocs.io/en/stable/extensions/plugins.html>`_, 
`callbacks <https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html>`_, 
or by overriding `methods <https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#methods>`_. 

Please see the NeMo ASR, NLP, TTS, collections sections for domain-specific documentation.


PyTorch Lightning Trainer
~~~~~~~~~~~~~~~~~~~~~~~~~

Since every NeMo Model is a ``LightningModule``, we can automatically take advantage of the PyTorch Lightning ``Trainer``.
Every NeMo `example <https://github.com/NVIDIA/NeMo/tree/r1.0.0rc1/examples>`_ training script uses the ``Trainer`` object
to fit the model.

First instantiate the model and trainer and then call ``.fit``:

.. code-block:: python
    
    # We first instantiate the trainer based on the model configuration.
    # See the model configuration documentation for details.    
    trainer = pl.Trainer(**cfg.trainer)

    # Then pass the model configuration and trainer object into the NeMo model
    model = TextClassificationModel(cfg.model, trainer=trainer)

    # Now we can train with by calling .fit
    trainer.fit(model)

    # Or we can run the test loop on test data by calling
    trainer.test(model=model)

All `trainer flags <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags>`_ 
can be set from from the NeMo Configuration, see below for more details on model configuration.
    

Model Configuration
-------------------

Hydra is an open-source Python framework that simplifies configuration for complex applications
that must bring together many different software libraries. 
Conversational AI model training is a great example of such an application.
To train a Conversational AI model, we must be able to configure:

- neural network architectures
- training and optimization algorithms 
- data pre/post processing
- data augmentation
- experiment logging/visualization
- model checkpointing   

Please see the `Hydra Tutorials <https://hydra.cc/docs/tutorials/intro>`_ for an introduction to using Hydra.

With Hydra we can configure everything needed for NeMo with three interfaces:

- Command Line (CLI) 
- Configuration Files (YAML)
- Dataclasses (Python)

YAML
~~~~
NeMo provides YAML configuration files for all of our `example <https://github.com/NVIDIA/NeMo/tree/r1.0.0rc1/examples>`_ training scripts.
YAML files make it easy to experiment with different model and training configurations.

Every NeMo example YAML has the same underlying configuration structure:

- trainer
- exp_manager
- model

Model configuration always contain train_ds, validation_ds, test_ds, and optim. 
Model architectures vary across domains so please see the ASR, NLP, and TTS Collections documentation for 
more detailed information on Model architecture configuration.

A NeMo configuration file should look something like this:

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
            manifest_filepath: /path/to/my/train/manifest.json
            batch_size: 256
            shuffle: True
        validation_ds:
            manifest_filepath: /path/to/my/validation/manifest.json
            batch_size: 32
            shuffle: False
        test_ds:
            manifest_filepath: /path/to/my/test/manifest.json
            batch_size: 32
            shuffle: False
        optim:
            name: novograd
            lr: .01
            betas: [0.8, 0.5]
            weight_decay: 0.001
        # network architecture can vary greatly depending on the domain
        encoder:
            ...
        decoder:
            ...

More specific details about configuration files for each collection can be found on the following pages:

* :doc:`asr/configs`
        
CLI
~~~
With NeMo and Hydra, every aspect of model training can modified from the command line. 
This is extremely helpful for running lots of experiments on compute clusters or 
for quickly testing parameters while developing.

All NeMo `examples <https://github.com/NVIDIA/NeMo/tree/r1.0.0rc1/examples>`_ come with instructions on how to 
run the training/inference script from the command line, see `here <https://github.com/NVIDIA/NeMo/blob/4e9da75f021fe23c9f49404cd2e7da4597cb5879/examples/asr/speech_to_text.py#L24>`_
for an example.

With Hydra, arguments are set using the ``=`` operator:

.. code-block:: bash

    python examples/asr/speech_to_text.py \
        model.train_ds.manifest_filepath=/path/to/my/train/manifest.json \
        model.validation_ds.manifest_filepath=/path/to/my/validation/manifest.json \
        trainer.gpus=2 \
        trainer.max_epochs=50

We can use the ``+`` operator to add arguments from the CLI:

.. code-block:: bash

    python examples/asr/speech_to_text.py \
        model.train_ds.manifest_filepath=/path/to/my/train/manifest.json \
        model.validation_ds.manifest_filepath=/path/to/my/validation/manifest.json \
        trainer.gpus=2 \
        trainer.max_epochs=50 \
        +trainer.fast_dev_run=true

We can use the ``~`` operator to remove configurations:

.. code-block:: bash

    python examples/asr/speech_to_text.py \
        model.train_ds.manifest_filepath=/path/to/my/train/manifest.json \
        model.validation_ds.manifest_filepath=/path/to/my/validation/manifest.json \
        ~model.test_ds \
        trainer.gpus=2 \
        trainer.max_epochs=50 \
        +trainer.fast_dev_run=true

We can specify configuration files using the ``--config-path`` and ``--config-name`` flags:

.. code-block:: bash

    python examples/asr/speech_to_text.py \
        --config-path=conf \
        --config-name=quartznet_15x5 \
        model.train_ds.manifest_filepath=/path/to/my/train/manifest.json \
        model.validation_ds.manifest_filepath=/path/to/my/validation/manifest.json \
        ~model.test_ds \
        trainer.gpus=2 \
        trainer.max_epochs=50 \
        +trainer.fast_dev_run=true


Dataclasses
~~~~~~~~~~~
Dataclasses allow NeMo to ship model configurations as part of the NeMo library and 
also enables pure Python configuration of NeMo models. 
With Hydra, dataclasses can be used to create `structured configs <https://hydra.cc/docs/tutorials/structured_config/intro>`_ 
for the Conversational AI application. 

As an example, see the code block below for an Attenion is All You Need machine translation model.
The model configuration can be instantiated and modified like any Python `Dataclass <https://docs.python.org/3/library/dataclasses.html>`_.


.. code-block:: Python

    from nemo.collections.nlp.models.machine_translation.mt_enc_dec_config import AAYNBaseConfig

    cfg = AAYNBaseConfig()

    # modify the number of layers in the encoder
    cfg.encoder.num_layers = 8

    # modify the training batch size
    cfg.train_ds.tokens_in_batch = 8192

.. note:: Configuration with Hydra always has the following precedence CLI > YAML > Dataclass

Experiment Manager
------------------
NeMo's Experiment Manager leverages PyTorch Lightning for model checkpointing, 
TensorBoard Logging, and Weights and Biases logging. The Experiment Manager is included by default
in all NeMo example scripts.

To use the experiment manager simply call it and pass in the PyTorch Lightning ``Trainer``.

.. code-block:: python

    exp_manager(trainer, cfg.get("exp_manager", None))

And is configurable via YAML with Hydra.

.. code-block:: bash

    exp_manager:
        exp_dir: /path/to/my/experiments
        name: my_experiment_name
        create_tensorboard_logger: True
        create_checkpoint_callback: True

Optionally launch Tensorboard to view training results in ./nemo_experiments (by default).

.. code-block:: bash

    tensorboard --bind_all --logdir nemo_experiments

..

If ``create_checkpoint_callback`` is set to ``True`` then NeMo will automatically create checkpoints during training
using PyTorch Lightning's `ModelCheckpoint <https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint>`_
We can configure the ``ModelCheckpoint`` via YAML or CLI.

.. code-block:: yaml

    exp_manager:
        ...
        # configure the PyTorch Lightning ModelCheckpoint using checkpoint_call_back_params
        # any ModelCheckpoint argument can be set here

        # save the best checkpoints based on this metric
        checkpoint_callback_params.monitor=val_loss 
        
        # choose how many total checkpoints to save
        checkpoint_callback_params.save_top_k=5

We can auto-resume training as well by configuring the exp_manager. 
Being able to auto-resume is important when doing long training runs that are premptible or 
may be shut down before the training procedure has completed.
To auto-resume training set the following via YAML or CLI:

.. code-block:: yaml

    exp_manager:
        ...
        # resume training if checkpoints already exist
        resume_if_exists: True

        # to start training with no existing checkpoints
        resume_ignore_no_checkpoint: True

        # by default experiments will be versioned by datetime
        # we can set our own version with
        exp_manager.version: my_experiment_version


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
