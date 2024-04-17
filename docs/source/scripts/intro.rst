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

The ``megatron_change_num_partitions.py`` script is used to convert the model parallel size configurations for NVIDIA's NeMo Megatron models. It supports the conversion between different parallelism configurations including tensor model parallelism, pipeline model parallelism, and virtual pipeline parallelism. This allows fine-tuning the model's parallel configuration to better fit the available compute resources.

Important Arguments
^^^^^^^^^^^^^^^^^^^

- **--model_file**: Path to the source model file.
- **--target_file**: Path to the target model file where the converted model will be saved.
- **--model_class**: The class path of the model if using models other than the default MegatronGPTModel. E.g., ``nemo.collections.nlp.models.language_modeling.megatron_t5_model.MegatronT5Model``.
- **--tensor_model_parallel_size** and **--target_tensor_model_parallel_size**: Current and target sizes for tensor model parallelism.
- **--pipeline_model_parallel_size** and **--target_pipeline_model_parallel_size**: Current and target sizes for pipeline model parallelism.
- **--precision**: The precision mode of the model; typically set to bf16 for reduced memory footprint.

The arguments ``--tensor_model_parallel_size`` and ``--pipeline_model_parallel_size`` can be set to -1 to automatically infer the size from the model configuration.

Additional optional arguments include ``--num_gpu_per_node`` to specify the number of GPUs per node and ``--tokenizer_model_path`` to override the tokenizer model path used in the model configuration.

Usage
^^^^^

The script can be executed with different parameters depending on the specific model and the desired parallelism configuration. Below are example usages for converting parallelism settings for both GPT and T5 models, as well as other specific use cases.

**Tensor and Pipeline Parallelism Conversion for GPT:**

.. code-block:: bash

    python megatron_change_num_partitions.py \
        --model_file=PATH_TO_SRC_FILE \
        --target_file=PATH_TO_TGT_FILE \
        --tensor_model_parallel_size=-1 \
        --target_tensor_model_parallel_size=1 \
        --pipeline_model_parallel_size=-1 \
        --target_pipeline_model_parallel_size=1 \
        --precision=bf16

**Conversion for T5 Model:**

.. code-block:: bash

    python megatron_change_num_partitions.py \
        --model_file=PATH_TO_SRC_FILE \
        --target_file=PATH_TO_TGT_FILE \
        --model_class="nemo.collections.nlp.models.language_modeling.megatron_t5_model.MegatronT5Model" \
        --tensor_model_parallel_size=-1 \
        --target_tensor_model_parallel_size=1 \
        --pipeline_model_parallel_size=-1 \
        --target_pipeline_model_parallel_size=1 \
        --target_pipeline_model_parallel_split_rank=0 \
        --precision=bf16

 Make sure to refer to the full script documentation on GitHub at `megatron_change_num_partitions.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/megatron_change_num_partitions.py>`_ for complete details and advanced options.
