Scripts
=======

preprocess_data_for_megatron.py
-------------------------------

This script is designed for preprocessing text data to be used with NVIDIA's Megatron language models. It handles the conversion of raw text files into formats suitable for training these large language models.

**Key Arguments**:

- ``--input``: Path to the input text file.
- ``--output_prefix``: Prefix for the output files.
- ``--vocab_file``: Path to the vocabulary file.
- ``--dataset_impl``: Implementation of the dataset to use (e.g., 'mmap').
- ``--tokenizer_model``: Path to the tokenizer model.

**Example Usage**:

.. code-block:: bash

    python preprocess_data_for_megatron.py --input your_dataset.txt --output_prefix processed_data --vocab_file vocab.json --dataset_impl mmap --tokenizer_model tokenizer.model

For complete details and more options, refer to the full script documentation on GitHub at `preprocess_data_for_megatron.py <https://github.com/NVIDIA/NeMo/blob/main/scripts/nlp_language_modeling/preprocess_data_for_megatron.py>`_.

merge_lora_weights.py
----------------------

The `merge_lora_weights.py` script is used to integrate LoRA-based model weights with a base model's weights. This is particularly useful for users who need to merge adaptations from LoRA into standard model architectures for enhanced performance or specific capabilities.

**Key Arguments**:

- ``--model_path``: Path to the directory containing the base model.
- ``--lora_path``: Path to the directory containing the LoRA weights.
- ``--output_path``: Path where the merged model will be saved.
- ``--config_file``: Configuration file specifying the parameters for merging.

**Example Usage**:

.. code-block:: bash

    python merge_lora_weights.py --model_path /path/to/base_model --lora_path /path/to/lora_weights --output_path /path/to/output_model --config_file merge_config.json

For complete details and more options, refer to the full script documentation on GitHub at `merge_lora_weights.py <https://github.com/NVIDIA/NeMo/tree/main/scripts/nlp_language_modeling/merge_lora_weights>`_.

megatron_change_num_partitions.py
---------------------------------

The `megatron_change_num_partitions.py` script is utilized for adjusting the tensor parallelism partitions of Megatron-LM models. This functionality is critical when scaling models across different numbers of GPUs, enabling efficient model parallel training.

**Key Arguments**:

- ``--input_model_file``: Path to the input model file that needs partition adjustments.
- ``--output_model_file``: Path for saving the modified model file.
- ``--tensor_model_parallel_size``: The number of partitions to split the model into, adjusting the tensor model parallel size.

**Example Usage**:

.. code-block:: bash

    python megatron_change_num_partitions.py --input_model_file path/to/your_model.pt --output_model_file path/to/your_modified_model.pt --tensor_model_parallel_size 8

This script supports changes in the tensor parallelism configuration by re-assigning weights across different partitions. Make sure to refer to the full script documentation on GitHub at `megatron_change_num_partitions.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/megatron_change_num_partitions.py>`_ for complete details and advanced options.
