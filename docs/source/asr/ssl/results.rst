Checkpoints
===========

Refer to `ASR Checkpoints <../results.rst>` for details about loading pre-trained checkpoints in NeMo.
For SSL specific checkpoints, refer to the section below.

NGC Pretrained Checkpoints
--------------------------

The SSL collection has checkpoints of several models trained on various datasets. These checkpoints are 
obtainable via NGC `NeMo Automatic Speech Recognition collection <https://catalog.ngc.nvidia.com/orgs/nvidia/collections/nemo_asr>`_.
The model cards on NGC contain more information about each of the checkpoints available.

The tables below list the SSL models available from NGC. The models can be accessed via the :code:`from_pretrained()` method inside
the ASR Model class. In general, you can load any of these models with code in the following format:

.. code-block:: python

  import nemo.collections.asr as nemo_asr
  model = nemo_asr.models.ASRModel.from_pretrained(model_name="<MODEL_NAME>")

Where the model name is the value under "Model Name" entry in the tables below.

For example, to load the conformer Large SSL checkpoint, run:

.. code-block:: python

  model = nemo_asr.models.ASRModel.from_pretrained(model_name="ssl_en_conformer_large")

You can also call :code:`from_pretrained()` from the specific model class (such as :code:`SpeechEncDecSelfSupervisedModel`
for Conformer) if you need to access a specific model functionality.

If you would like to programatically list the models available for a particular base class, you can use the
:code:`list_available_models()` method.

.. code-block:: python

  nemo_asr.models.<MODEL_BASE_CLASS>.list_available_models()


Fine-tuning on Downstream Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are multiple ASR tutorials provided in the :ref:`Tutorials <tutorials>` section. Most of these tutorials explain how to instantiate a pre-trained checkpoint, prepare the model for fine-tuning on some dataset as a demonstration.

Inference Execution Flow Diagram
--------------------------------

When preparing your own inference scripts after downstream fine-tuning, please follow the execution flow diagram order for correct inference, found at the `examples directory for ASR collection <https://github.com/NVIDIA/NeMo/blob/stable/examples/asr/README.md>`_.

SSL Models
-----------------------------------

Below is a list of all the SSL models that are available in NeMo.


.. csv-table::
   :file: data/benchmark_ssl.csv
   :align: left
   :widths: 40, 10, 50
   :header-rows: 1

-----------------------------