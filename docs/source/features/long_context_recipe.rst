.. _long_context_recipe:

Long Context Recipe
------------------------

Long context model training enhances the capability of large language models to handle long context inputs. Longer sequence lengths are beneficial for many NLP tasks, such as document-level summarization, long document classification, and long document question answering. NeMo Framework provides a recipe to train long context models like Llama-3, Mixtral, and Nemotron.

Access Long Context Recipe
========================

NeMo 2.0 is providing tested recipes to train long context models. `The recipe is available in the NeMo LLM GPT Trainer Directory <https://github.com/NVIDIA/NeMo/tree/main/nemo/collections/llm/recipes>`_ directory.

The following charts for the Llama-3, Mixtral, and Nemotron models show the different sequence lengths supported by each model at various sizes.

Llama-3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Llama-3
   :widths: 30 25 25
   :header-rows: 1

   * - Sequence Length
     - 8B
     - 70B
   * - 16k
     - Yes
     - Yes
   * - 64k
     - Yes
     - Yes
   * - 128k
     - TODO
     - TODO

Mixtal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Mixtral
   :widths: 30 25 25 25
   :header-rows: 1

   * - Sequence Length
     - 8x3B
     - 8x7B
     - 8x22B
   * - 16k
     - Yes
     - Yes
     - TODO
   * - 64k
     - Yes
     - Yes
     - TODO
   * - 128k
     - TODO
     - TODO
     - TODO

Nemotron (Not yet supported in NeMo 2.0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Nemotron
   :widths: 30 25 25
   :header-rows: 1

   * - Sequence Length
     - 15B
     - 340B
   * - 16k
     - TODO
     - TODO
   * - 64k
     - TODO
     - TODO
   * - 128k
     - TODO
     - TODO


Context Parallelism
========================

Context Parallelism (CP) is a method for parallelizing the processing of neural network activations across multiple GPUs by partitioning the input tensors along the sequence dimension.Unlike Sequence Parallelism (SP) that partitions the activations of specific layers, CP divides the activations of all layers.

CP is critical for training long context models, as it allows the model to handle longer sequences by distributing the sequence activations across multiple GPUs. This method reduces the memory footprint and computational cost of processing long sequences.

Enable Context Parallelism
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To activate CP in the NeMo Framework, set the context_parallel_size parameter in the model configuration. This parameter specifies the number of GPUs across which the model’s sequence activations are distributed.

Set ``context_parallel_size`` to a value greater than ``1`` to enable sequence-wide model parallelism.

   .. code-block:: yaml

       context_parallel_size: 1  # Example to enable Context Parallelism

The configuration can be found and modified here: `NeMo Megatron Core Context Config <https://docs.nvidia.com/Megatron-Core/developer-guide/latest/api-guide/context_parallel.html>`_.

Implement Context Parallelism
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NeMo Framework leverages functionalities from both Megatron Core and Transformer Engine to implement CP efficiently. During forward propagation, each GPU handles a segment of the sequence, storing only the necessary Key and Value (KV) pairs. In the backward pass, these KV pairs are reassembled across GPUs using advanced communication schemes like all-gather and reduce-scatter transformed into point-to-point communications in a ring topology. This method reduces the memory footprint significantly while maintaining computational efficiency.

Visit our source code for more insights into the implementation:

  - `Megatron Core wrappers for Transformer Engine <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/custom_layers/transformer_engine.py>`_.
  - `Transformer Engine attention modules <https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention.py>`_.
