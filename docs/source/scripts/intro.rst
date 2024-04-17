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
