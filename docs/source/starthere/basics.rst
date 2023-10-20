NeMo basics
===========

On this page we will go a little bit deeper on how NeMo works, to give you a good foundation for using NeMo for your :ref:`desired usecase <where_next>`.

NeMo models
-----------

For NeMo developers, a "Model" is the neural network(s) as well as all the infrastructure supporting those network(s), wrapped into a singular, cohesive unit. As such, all NeMo models are constructed to contain the following out of the box:

* Neural Network architecture

* Dataset + Data Loaders

* Data preprocessing + postprocessing

* Optimizer + Schedulers

* Any other supporting infrastructure: tokenizers, language model configuration, data augmentation etc.

For your convenience, NeMo contains many pre-defined models that have successfully been trained on various tasks. We make available the architectures of the models (which you can see in their config files). For many tasks & languages, we also release the model weights, which are contained within ``.nemo`` files which you can browse on NGC [link] and hugging face [link].

You can run inference with the pretrained models as they are, or you can train them on your own data: feel free to finetune from the provided model weights, or start from scratch; reuse the architectures as they are, or make changes; or define your own models!

NeMo & PyTorch Lightning & Hydra
--------------------------------

NeMo makes use of `PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/starter/introduction.html>`_ (PTL) to abstract away all of the model training details.

Most training & inference scripts (available here [link]) provided for NeMo make use of `Hydra <https://hydra.cc/docs/intro/>`_ to allow you to overwrite the default configurations of the scripts.

You can see an example of the role that PTL and Hydra play in the next section.

Anatomy of a NeMo training script
---------------------------------

All of the NeMo training scripts (accessible here[link]) follow a similar structure.

In the video below, we will explain the workings of an ASR training script below, but the same principles apply to all other training scripts provided in the NeMo repository - the main difference between them being the model class that they instantiate.

.. raw:: html

    <div style="position: relative; padding-bottom: 3%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <video allowfullscreen controls>
		<source src="../../../../new_docs_diagrams/train_overview.mov" type="video/mp4">
	</video>
    </div>

The overall final diagram is shown here:

.. figure:: ../../../../new_docs_diagrams/train_overview.png
   :alt: overview of NeMo training script 

   An overview of how training scripts work in NeMo.


You can use this script as a template to specify your own training scripts, or refer to the examples in the examples dir[link].

You can read more about how you can modify the ASR training script to suit your needs  :doc:`here <../asr/training>`.

Using NeMo examples
-------------------

The examples scripts [link] in the NeMo repository allow you to quickly get started with training and inference for various tasks. They make heavy use of 'configs', which we will explain briefly below.

Understanding the role of configs in NeMo
-----------------------------------------

Section below needs to be completed - rough outline below.

NeMo has various pre-defined model classes... all of their constructors expect a 'config' (OmegaConf) (specificlaly model config).

Our example scripts also use also use 'configs', and we recommend you do the same.

All of the (~=)main functions in the example scripts rely on some config being specified, (which impacts e.g. model architecture, data used, details of model training).

Base config - comes from YAML file or dataclass. Can be overwritten when the main function is called (this is implemented using Hydra).

Learn more... (link to some hydra and NeMo docs)

.. _where_next

Where next?
-----------

You have a few options:

* dive in to examples [link] or tutorials [link]
* read docs of the collection [links] you want to work with
* learn more about 'nemo core' [link]


