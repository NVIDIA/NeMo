NeMo fundamentals
=================

On this page we will go a little bit deeper on how NeMo works, to give you a good foundation for using NeMo for your :ref:`desired usecase <where_next>`.

NeMo Models
-----------

A NeMo "model" includes all of the below components wrapped into a singular, cohesive unit:

* neural network architecture,

* dataset & data loaders,

* preprocessing of input data & postprocessing of model outputs,

* loss function, optimizer & schedulers,

* any other supporting infrastructure: tokenizers, language model configuration, data augmentation etc.

NeMo models are based on PyTorch. Many of their components are subclasses of ``torch.nn.Module``. NeMo models use PyTorch Lightning (PTL) for training, thus reducing the amount of boilerplate code needed.

NeMo models are also designed to be easily configurable; often this is done with YAML files. Below we show simplified examples of a NeMo model defined in pseudocode, and a config defined in YAML. We highlight the lines where the Python config parameter is read from the YAML file.

.. list-table:: Simplified examples of a model and config.
    :widths: 1 1
    :header-rows: 0

    * - .. code-block:: python
	  :caption: NeMo model definition (Python pseudocode)
	  :linenos:
	  :emphasize-lines: 4, 7, 10, 13, 16, 20

	  class ExampleEncDecModel:
	      # cfg is passed so it only contains "model" section
	      def __init__(self, cfg, trainer):
	          self.tokenizer = init_from_cfg(cfg.tokenizer)


	          self.encoder = init_from_cfg(cfg.encoder)


	          self.decoder = init_from_cfg(cfg.decoder)


	          self.loss = init_from_cfg(cfg.loss)


		  # optimizer configured via parent class


	      def setup_training_data(self, cfg):
	          self.train_dl = init_dl_from_cfg(cfg.train_ds)

	      def forward(self, batch):
	          # forward pass defined,
		  # as is standard for PyTorch models
	          ...

	      def training_step(self, batch):
	          log_probs = self.forward(batch)
	          loss = self.loss(log_probs, labels)
	          return loss


      - .. code-block:: yaml
	  :caption: Experiment config (YAML)
	  :linenos:
	  :emphasize-lines: 4, 7, 10, 13, 16, 20

	  #
	  # desired configuration of the NeMo model
	  model:
	      tokenizer:
	       ...

	      encoder:
	       ...

	      decoder:
	       ...

	      loss:
	       ...

	      optim:
	       ...


	      train_ds:
	       ...

	  # desired configuration of the
	  # PyTorch Lightning trainer object
	  trainer:
	      ...


Configuring and training NeMo models
------------------------------------

During initialization of the model, a lot of key parameters are read from the config (``cfg``), which gets passed in to the model construtor (left panel above, line 2).

The other object that passed into the constructor is a PyTorch Lightning ``trainer`` object, which handles the training process. The trainer will take care of the standard training `boilerplate <https://lightning.ai/docs/pytorch/stable/common/trainer.html#under-the-hood>`__. For things that are not-standard, PTL will refer to any specific methods that we may have defined in our NeMo model. For example, PTL requires every model to have a specified ``training_step`` method (left panel above, line 15).

The configuration of the trainer is also specified in the config (right panel above, line 20 onwards). This will include parameters such as (number of) ``devices``, ``max_steps``, (numerical) ``precision`` and `more <https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api>`__.


Example training script
-----------------------

Putting it all together, here is an example training script for our ``ExampleEncDecModel`` model. We highlight the 3 most important lines, which put together everything we discussed in the previous section.

.. code-block:: python
	:caption: run_example_training.py
	:linenos:
	:emphasize-lines: 10, 11, 12

	import pytorch_lightning as pl
	from nemo.collections.path_to_model_class import ExampleEncDecModel
	from nemo.core.config import hydra_runner

	@hydra_runner(
		config_path="config_file_dir_path",
		config_name="config_file_name"
	)
	def main(cfg):
		trainer = pl.Trainer(**cfg.trainer)
		model = ExampleEncDecModel(cfg.model, trainer)
		trainer.fit(model)

	if __name__ == '__main__':
		main(cfg)


Let's go through the code:

* *Lines 1-3*: import statements (second one is made up for the example).
* *Lines 5-8*: a decorator on lines 5-8 of ``run_example_training.py`` will look for a config file at ``{config_path}/{config_name}.yaml``, and load its contents into the ``cfg`` object that is passed into the ``main`` function. This functionality is provided by `Hydra <https://hydra.cc/docs/intro/>`__. Instead of a YAML file, we could also have specified the default config as a dataclass, and passed that into the ``@hydra_runner`` decorator.
* *Line 7*: initialize a PTL trainer object, using the parameters specified in the ``trainer`` section of the config.
* *Line 8*: initialize a NeMo model, passing in both the parameters in the ``model`` section of the config, and a PTL trainer.
* *Line 9*: call ``trainer.fit`` on the model. This one unassuming line will carry out our entire training process. PTL will make sure we iterate over our data and call the ``training_step`` we define for each batch (as well as any other PTL `callbacks <https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html>`__ that may have been defined).



Overriding configs
------------------

The ``cfg`` object in the script above is a dictionary-like object that contains our configuration parameters. Specifically, it is an `OmegaConf <https://omegaconf.readthedocs.io/en/2.3_branch/usage.html>`__ ``DictConfig`` object. These objects have special features such as dot-notation `access <https://omegaconf.readthedocs.io/en/latest/usage.html#access>`__, `variable interpolation <https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation>`__ and ability to set `mandatory values <https://omegaconf.readthedocs.io/en/latest/usage.html#mandatory-values>`__.

We can run the script above like this:

.. code-block:: bash

	python run_example_training.py

This will use the default config file specified inside the ``@hydra_runner`` decorator.

We can specify a different config file to use by calling the script like this:

.. code-block:: diff

	 python run_example_training.py \
	+    --config_path="different_config_file_dir_path" \
	+    --config_name="different_config_file_name"

We can also override, delete or add elements to the config when we call the script like this:


.. code-block:: diff

	 python run_example_training.py \
	     --config_path="different_config_file_dir_path" \
	     --config_name="different_config_file_name" \
	+    model.optim.lr=0.001 \                                     # overwriting
	+    model.train_ds.manifest_filepath="your_train_data.json" \  # overwriting
	+    ~trainer.max_epochs \                                      # deleting
	+    +trainer.max_steps=1000                                    # adding

Running NeMo scripts
--------------------

NeMo scripts typically take on the form shown above, where the Python script relies on a config object which has some specified default values that you can choose to override.

The NeMo `examples <https://github.com/NVIDIA/NeMo/tree/main/examples/>`__ directory contains many scripts for training and inference of various existing NeMo models. Note that this includes default configs whose default values for model, optimizer and trainer parameters were tuned over the course of many GPU-hours of the NeMo team's experiments. We thus recommend using these as a starting point for your own experiments.

.. note::
	**NeMo inference scripts**

	The examples scripts directory also contains many inference scripts, e.g. `transcribe_speech.py <https://github.com/NVIDIA/NeMo/blob/main/examples/asr/transcribe_speech.py>`_. These normally have a different structure to the training scripts, as they have a lot of additional utilities for reading and saving files. The inference scripts also use configs, but these naturally do not require the ``trainer``, ``model``, ``exp_manager`` sections. Additionally, due to having fewer elements, the default configs for inference scripts are normally specified as dataclasses rather than separate files. Elements also can be overwritten/added/deleted via the command line.


Specifying training data
------------------------

NeMo will handle creation of data loaders for you, as long as you put your data into the expected input format. You may also need to train a tokenizer before starting training. Learn more about data formats for :doc:`LLM <../nlp/nemo_megatron/gpt/gpt_training>`, :doc:`Multimodal <../multimodal/mllm/datasets>`, :ref:`Speech AI <section-with-manifest-format-explanation>`, and :doc:`Vision models <../vision/datasets>`.


Model checkpoints
-----------------

Throughout training, model :doc:`checkpoints <../checkpoints/intro>` will be saved inside ``.nemo`` files. These are archive files containing all the necessary components to restore a usable model, e.g.:

* model weights (``.ckpt`` files),
* model configuration (``.yaml`` files),
* tokenizer files

The NeMo team also releases pretrained models which you browse on `NGC <https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC>`_ and `HuggingFace Hub <https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia>`_.


Finetuning
----------

NeMo allows you to finetune models as well as train them from scratch.

You can do this by initializing a model with random weights, replacing some/all the weights with those of a pretrained model, and then continuing training as normal, potentially with some small changes such as reducing your learning rate or freezing some model parameters.


.. _where_next:

Where next?
-----------

Here are some options:

* dive in to `examples <https://github.com/NVIDIA/NeMo/tree/main/examples>`_ or :doc:`tutorials <./tutorials>`
* read docs of the domain (e.g. :doc:`LLM <../nlp/nemo_megatron/intro>`, :doc:`Multimodal <../multimodal/mllm/intro>`, :doc:`ASR <../asr/intro>`, :doc:`TTS <../tts/intro>`, :doc:`Vision Models <../vision/intro>`) you want to work with
* learn more about the inner workings of NeMo:

  * `NeMo Primer <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/00_NeMo_Primer.ipynb>`_ notebook tutorial

    * hands-on intro to NeMo, PyTorch Lightning, and OmegaConf
    * shows how to use, modify, save, and restore NeMo models

  * `NeMo Models <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/01_NeMo_Models.ipynb>`__ notebook tutorial

    * explains the fundamentals of how NeMo models are created

  * :doc:`NeMo Core <../core/core>` documentation

