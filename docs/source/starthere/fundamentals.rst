NeMo fundamentals
=================

On this page we will go a little bit deeper on how NeMo works, to give you a good foundation for using NeMo for your :ref:`desired usecase <where_next>`.

.. _nemo_model:

NeMo Models
-----------

For NeMo developers, a "model" is the neural network(s) as well as all the infrastructure supporting those network(s), wrapped into a singular, cohesive unit. As such, all NeMo models are constructed to contain the following out of the box:

* neural network architecture,

* dataset & data loaders,

* data preprocessing & postprocessing,

* optimizer & schedulers,

* any other supporting infrastructure: tokenizers, language model configuration, data augmentation etc.

For your convenience, NeMo contains many pre-defined models that have successfully been trained on various tasks. We make available the architectures of the models (which you can see in their config files). For many tasks & languages, we also release the model weights, which are contained within ``.nemo`` files which you can browse on `NGC <https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC>`_ and `HuggingFace <https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia>`_.

You can run inference with the pretrained models as they are, or you can train them on your own data: feel free to finetune from the provided model weights, or start from scratch; reuse the architectures as they are, or make changes; or define your own models!

NeMo & PyTorch Lightning & Hydra
--------------------------------

NeMo makes use of `PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/starter/introduction.html>`_ (PTL) to abstract away all of the model training details.

Most training & inference scripts (available `here <https://github.com/NVIDIA/NeMo/tree/main/examples>`_) in the NeMo repository make use of `Hydra <https://hydra.cc/docs/intro/>`_ to define the configurations of the scripts, and to allow you to easily overwrite the configurations if you wish.

You will see how PTL and Hydra interact with NeMo in the sections to follow.

.. _nemo_training_script:

Anatomy of a NeMo training script
---------------------------------

For the rest of this page, we will show how to use NeMo by looking at an ASR finetuning script. All of the NeMo training scripts (accessible `here <https://github.com/NVIDIA/NeMo/tree/main/examples>`_) follow a similar structure.

Let's call the script at `NeMo/examples/asr/asr_transducer/speech_to_text_rnnt.py <https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_transducer/speech_to_text_rnnt.py>`_. We set some parameters to make sure we do training using a FastConformer architecture (and using most of its default optimization parameters), finetuning from an existing model, and will use our own training and validation data.

.. note:: In the code block below we have added some comments - if you wish to run the code, you will need to remove any text after the ``\`` characters.

.. _nemo_fundamentals_cli:

.. code-block:: bash
	:linenos:

	python NeMo/examples/asr/asr_transducer/speech_to_text_rnnt.py \
	    --config_path="../fastconformer/" \	   # (1. CONFIG) can use alternative YAML file 
	    --config_name="conformer_transducer" \ # for the default config
	    model.optim.lr=0.001 \					 # (1. CONFIG) can 
	    model.train_ds.manifest_filepath="your_train_data.json" \    # overwrite, add,
	    model.validation_ds.manifest_filepath="your_val_data.json" \ # or delete lines
	    +init_from_nemo_model="model_to_finetune_from.nemo"          # from the config


The script that we are calling looks like the code block below. We have removed some logging and model testing code, and added some brief comments. The comments also contain the names of the section headings (like ``(1. CONFIG)``) in which we will explain how that aspect of NeMo works in more detail.

.. _nemo_fundamentals_python:

.. code-block:: python
	:linenos:

	# simplified contents of NeMo/examples/asr/asr_transducer/speech_to_text_rnnt.py

	import pytorch_lightning as pl

	from nemo.collections.asr.models import EncDecRNNTBPEModel  # (2. MODEL CLASS)
	from nemo.core.config import hydra_runner
	from nemo.utils.exp_manager import exp_manager

	@hydra_runner(
	    config_path=".../contextnet_rnnt", # (1. CONFIG) default base config,
	    config_name="config_rnnt" .        # can be overwritten by command line
	)
	def main(cfg):                         # (1. CONFIG) passed in as `cfg` variable

	    trainer = pl.Trainer(**cfg.trainer)                 # (3. TRAINER) handles number
   								# of GPUs, epochs etc.

	    exp_manager(trainer, cfg.get("exp_manager", None))  # (4. EXPERIMENT MANAGER)
	    							# handles checkpoints, 
								# W&B etc.
	    asr_model = EncDecRNNTModel(cfg=cfg.model, trainer=trainer) # (2. MODEL CLASS)
	    								# init model object

	    asr_model.maybe_init_from_pretrained_checkpoint(cfg)# (5. CHECKPOINT) init from
	    							# pretrained checkpoint if 
								# provided
	    trainer.fit(asr_model)				# (6. TRAINING) run training

	    # (7. .nemo FILE) at the end of training, you will have a .nemo file containing
	    # the weights of your trained model

	if __name__ == '__main__':
    		main() 


We encourage you to take a moment to digest the script and the comments.

Every NeMo training script takes in some specified config, and uses some specific NeMo model class. The config and the model class are the only things that vary between the task and the architecture. The rest of the training script handles the training setup & execution with just a few lines of code. 

.. tip:: Almost every training script example in NeMo is essentially identical to this example except for the config & model class.

Now let's break down what happens in the script, following the sections numbered in the comments.

1. Config
---------

The config that we use in our ``main`` function is passed in as the variable ``cfg``. 

Configs are an important part of NeMo. They are a nested dictionary-like data structure containing the information needed to run a NeMo training or inference script.

Specifically, ``cfg`` (the config that is passed into our :ref:`Python training script<nemo_fundamentals_python>` on line 13) is an `OmegaConf <https://omegaconf.readthedocs.io/en/2.3_branch/usage.html>`_ ``DictConfig`` object.

.. collapse:: What exactly is an OmegaConf DictConfig object?

	.. note:: OmegaConf DictConfig objects provide extra functionality beyond plain dictionaries such as [TODO] , which are helpful for NeMo ...[TODO]

		Due to their similarity, it is easy to `convert <https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#omegaconf-to-container>`_ from a DictConfig object to a dictionary, (and `vice versa <https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-a-dictionary>`_).

		[TODO: maybe handle fact that omegaconf docs are not explicit about how an 'omegaconf object' can be either a dictconfig or listconfig, or structuredconfig]


 In the case of training scripts, all configs are expected to contain the entries ``model``, ``trainer`` and ``exp_manager``:

.. code-block:: yaml

	# outline of a config needed for training a NeMo model

	model:        # contains desired configuration of the NeMo model
	  
	  # NEEDED BY ALL MODELS
	  train_ds:      # specify train dataset
	    ...

	  validation_ds: # specify validation dataset
	    ...

	  test_ds:	 # specify test dataset (not strictly necessary)
	    ...

	  # MODEL-DEPENDENT
	  encoder:
	    ...

	  decoder:
	    ... 

	  optim:
	    ...

	trainer:     # contains desired configuration of the PyTorch Lightning trainer
	  ...

	exp_manager: # contains desired configuration of the NeMo Experiment Manager
	  ...

You can see that all of these elements are used in our Python script, in lines 15, 18 and 21.

Specifying the content of the config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We specify the content of the config using YAML files. It is also possible to overwrite parts of the YAML files when calling the script on the command line.

On lines 10 & 11, our :ref:`Python training script<nemo_fundamentals_python>` specifies a default YAML file (located at ``{config_path}/{config_name}``). This file will be used as the config for the script if we do not specify an alternative file, nor overwrite (or add, or delete) any of its elements. In our example, we actually do specify an alternative YAML file for the config (on lines 2 & 3 of the :ref:`CLI command <nemo_fundamentals_cli>`). We also override some elements of that config (on lines 4-6) and add a new element (on line 7). The ability to override the config like this makes it easier to run experiments with different parameters.

All of this logic is implemented by Hydra. This is why we use the ``@hydra_runner`` decorator on line 9 of the :ref:`Python training script<nemo_fundamentals_python>`.

.. collapse:: More info about Hydra & how to use it

	``@hydra_runner`` is NeMo's slightly-modified version of Hydra's decorator ``@hydra.main``. [add more info on what is the difference?]

	Learn more about Hydra `here <https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/>`_.

The flow for specifying the content of the config is summarized in the diagram below.

.. code-block:: txt

	TODO: make this an actual diagram instead of text

	default YAML file (passed to @hydra_runner) |
	or                                          | ----------> config object -----> Python script
	other YAML file (passed via CLI)            |      ^
		                                           |
	                                	     any overrides,
						 additions or deletions
						   (specified via CLI)

NeMo-provided config files
^^^^^^^^^^^^^^^^^^^^^^^^^^

Within the examples directory of the NeMo repository, you will find battle-tested config files for various NeMo models. For example, the ASR config files are `here <https://github.com/NVIDIA/NeMo/tree/main/examples/asr/conf>`_.

If possible, we recommend using these config files as a starting point for your own experiments as these files specify model architecture and optimization parameters that have been proven to work after many GPU-hours of the NeMo team's experiments. 


Specifying input data in configs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
NeMo will handle creation of data loaders for you, as long as you put your data into the expected input format. You may also need to train a tokenizer before starting training. Learn more about data formats for :ref:`ASR <section-with-manifest-format-explanation>`, :doc:`NLP <../nlp/language_modeling>`, :doc:`TTS <../tts/datasets>`.

Configs for inference scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The examples scripts directory also contains many inference scripts, e.g. `transcribe_speech.py <https://github.com/NVIDIA/NeMo/blob/main/examples/asr/transcribe_speech.py>`_. These normally have a different structure to the training scripts, as they have a lot of additional utilities for reading and saving files. The inference scripts also use configs, but these naturally do not require the ``trainer``, ``model``, ``exp_manager`` sections. Additionally, due to having fewer elements, the default configs for inference scripts are normally specified as dataclasses rather than separate files. Elements also can be overwritten/added/deleted via the command line.

2. Model class
--------------

The beating heart [TODO: better phrasing?] of NeMo are its "models", which we implement as classes which will glue together the elements defined in the ``model`` section of the config. This includes the neural network architecture, the dataset & data loaders, the data preprocessing & postprocessing, the optimizer & schedulers, and any other supporting infrastructure.

The model classes are made to be as general as possible, allowing you to specify the architecture and the optimization parameters simply by changing the config, rather than any Python code.

[TODO: maybe add something about the config containing 'neural modules'?]

The model classes tend only to vary by nature of the modelling task and its loss (? and whether a tokenizer is needed, at least in the case of ASR?)

Below are some examples of model classes provided in NeMo. As you become a more advanced user, you may want to create your own.
[TODO: we will probably want a complete list somewhere?]


+---------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| ASR                                                                                                           | NLP                                                                                                                                    | TTS                                                                                                    |
+===============================================================================================================+========================================================================================================================================+========================================================================================================+
| `ASRModel <https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/models/asr_model.py>`_               | `TransformerLMModel <https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/transformer_lm_model.py>`_ | `SpectrogramGenerator <https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/tts/models/base.py>`_ |
+---------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| `EncDecCTCModel <https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/models/ctc_models.py>`_        | `MegatronGPTModel <https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py>`_     | `Tacotron2Model <https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/tts/models/tacotron2.py>`_  |
+---------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| `EncDecCTCModelBPE <https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/models/ctc_bpe_models.py>`_ | `MegatronNMTModel <https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/machine_translation/megatron_nmt_model.py>`_   | `HifiGanModel <https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/tts/models/hifigan.py>`_      |
+---------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+


Because NeMo utilizes PyTorch Lightning, the amount of boilerplate code involved in defining a NeMo model class is minimal. It also means that once we have specified the necessary methods in the model class, running training is as easy as running ``trainer.fit(model)``, as we do in line 27 of our :ref:`Python training script<nemo_fundamentals_python>`.

That method call is able to work because we defined a ``training_step()`` method for the ``EncDecRNNTModel`` class [improve this phrasing?] which looks like this:

.. code-block:: python

	# simplified code from NeMo/nemo/collections/asr/models/rnnt_models.py

	class EncDecRNNTModel(ASRModel, ASRModuleMixin, ExportableEncDecModel):

	    ...

	    def training_step(self, batch, batch_nb):

	        signal, signal_len, transcript, transcript_len = batch

	        encoded, encoded_len = self.forward(...)
	        decoder, target_length, states = self.decoder(...)
	        joint = self.joint(...)
	        loss_value = self.loss(...)

	        return {'loss': loss_value}

3. Trainer
----------

We use Pytorch Lightning to handle the training setup & execution. We first instantiate a ``pl.Trainer`` `object <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_ with the parameters specified in ``cfg.trainer`` (line 15 of our :ref:`Python training script<nemo_fundamentals_python>`). These parameters include number of steps or epochs of training, number of GPUs or nodes for training, and `many others <https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags>`_.  We will later call ``trainer.fit(asr_model)`` (line 27 of our :ref:`Python training script<nemo_fundamentals_python>`) to run the training.

4. Experiment Manager
---------------------

On line 18 of our `Python training script <nemo_fundamentals_python>`_, we call the ``exp_manager`` helper function which enables saving and resuming from checkpoints as the training progresses, logging experiment results (including e.g. `TensorBoard <https://www.tensorflow.org/tensorboard>`_ and `Weights & Biases <https://wandb.ai/site>`_) and :doc:`more <../core/exp_manager>`. 


5. Checkpoint
-------------

So far in the code (up to line 21 of our :ref:`Python training script<nemo_fundamentals_python>`), we have specified a model with an architecture as specified in ``cfg.model``, with model weights randomly initialized (according to the distribution specified in each part of the neural network).

NeMo allows you to start training with the weights of a pretrained model. This is accomplished by ``model.maybe_initiate_from_pretrained_checkpoint(cfg)`` (line 24 of our :ref:`Python training script<nemo_fundamentals_python>`). This method will check if a pretrained checkpoint is specified in the config, and if so, will load the weights from that checkpoint into the model. If no pretrained checkpoint is specified, the model will be left as-is (i.e. with the initialized random weights).

6. Training
-----------

Training is handled by PyTorch Lightning when we call ``trainer.fit(asr_model)`` (line 27 of our :ref:`Python training script<nemo_fundamentals_python>`).

7. ``.nemo`` file
-----------------

At the end of training, you will have a ``.nemo`` file saved in the directory at ``cfg.exp_manager.exp_dir`` (if that it is not specified, it will be saved in the directory ``"./nemo_experiments"`` inside the directory where you called the NeMo script [TODO: verify].

``.nemo`` files can be used for inference, or for further training. They are archive files containing all the necessary components to restore a usable model:

* model weights (``.ckpt`` files),
* other necessary :ref:`artifacts <core-register-artifacts>` such as tokenizers

.. _where_next:

Where next?
-----------

You have a few options:

* dive in to `examples <https://github.com/NVIDIA/NeMo/tree/main/examples>`_ or :doc:`tutorials <./tutorials>`
* read docs of the collection (:doc:`ASR <../asr/intro>`, :doc:`NLP <../nlp/nemo_megatron/intro>`, :doc:`TTS <../tts/intro>`) you want to work with
* learn more about the inner workings of NeMo:

  * `NeMo Primer <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/00_NeMo_Primer.ipynb>`_ notebook tutorial

    * hands-on intro to NeMo, PyTorch Lightning, and OmegaConf
    * shows how to use, modify, save, and restore NeMo models

  * `NeMo Models <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/01_NeMo_Models.ipynb>`__ notebook tutorial 

    * explains the fundamentals of how NeMo models are created

  * :doc:`NeMo Core <../core/core>` documentation

