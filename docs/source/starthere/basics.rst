NeMo basics
===========

On this page we will go a little bit deeper on how NeMo works, to give you a good foundation for using NeMo for your :ref:`desired usecase <where_next>`.

NeMo models
-----------

For NeMo developers, a "model" is the neural network(s) as well as all the infrastructure supporting those network(s), wrapped into a singular, cohesive unit. As such, all NeMo models are constructed to contain the following out of the box:

* Neural Network architecture

* Dataset + Data Loaders

* Data preprocessing + postprocessing

* Optimizer + Schedulers

* Any other supporting infrastructure: tokenizers, language model configuration, data augmentation etc.

For your convenience, NeMo contains many pre-defined models that have successfully been trained on various tasks. We make available the architectures of the models (which you can see in their config files). For many tasks & languages, we also release the model weights, which are contained within ``.nemo`` files which you can browse on `NGC <https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC>`_ and `HuggingFace <https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia>`_.

You can run inference with the pretrained models as they are, or you can train them on your own data: feel free to finetune from the provided model weights, or start from scratch; reuse the architectures as they are, or make changes; or define your own models!

NeMo & PyTorch Lightning & Hydra
--------------------------------

NeMo makes use of `PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/starter/introduction.html>`_ (PTL) to abstract away all of the model training details.

Most training & inference scripts (available `here <https://github.com/NVIDIA/NeMo/tree/main/examples>`_) in the NeMo repository make use of `Hydra <https://hydra.cc/docs/intro/>`_ to define the configurations of the scripts, and to allow you to easily overwrite the configurations if you wish.

You can see an example of the role that PTL and Hydra play in the next section.

.. _nemo_training_script:

Anatomy of a NeMo training script
---------------------------------

All of the NeMo training scripts (accessible `here <https://github.com/NVIDIA/NeMo/tree/main/examples>`_) follow a similar structure.

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


You can use this script as a template to specify your own training scripts, or refer to the examples in the `examples directory <https://github.com/NVIDIA/NeMo/tree/main/examples>`_.

You can read more about how you can modify the ASR training script to suit your needs  :doc:`here <../asr/training>`.

Using NeMo examples
-------------------

The `examples scripts <https://github.com/NVIDIA/NeMo/tree/main/examples>`_ in the NeMo repository allow you to quickly get started with training and inference for various tasks. They make heavy use of 'configs', which we will explain briefly below.

Understanding the role of configs in NeMo
-----------------------------------------

Almost every "main" function in the NeMo `examples scripts <https://github.com/NVIDIA/NeMo/tree/main/examples>`_ requires a 'config' to be specified. This is a dictionary-like data structure containing information such as:

* NeMo model config, PTL trainer parameters, NeMo experiment manager parameters (in the case of training scripts)
* pretrained model name or filepath, batch size, etc. (in the case of inference scripts)

The default parameters of the 'config' can be specified in a dataclass in the file, or as a separate YAML file.

The default parameters can be overwritten in various ways, e.g.:

* by specifying a different YAML file when calling the script
* by overwriting individual elements of the config 

Both of the above can be seen in the :ref:`Anatomy of a NeMo training script <nemo_training_script>` section above.

The functionality to specify and overwrite configs is implemented using `Hydra <https://hydra.cc/docs/intro/>`_.

You can learn more about configs in NeMo :ref:`here <nemo_configuration>`.

.. _where_next:

Where next?
-----------

You have a few options:

* dive in to `examples <https://github.com/NVIDIA/NeMo/tree/main/examples>`_ or :doc:`tutorials <./tutorials>`
* read docs of the collection (:doc:`ASR <../asr/intro>`, :doc:`NLP <../nlp/nemo_megatron/intro>`, :doc:`TTS <../tts/intro>`) you want to work with
* learn more about the :doc:`fundamentals <../core/core>` of NeMo


