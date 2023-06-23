.. _megatron_onnx_export:

ONNX Export of Megatron Models
====================================

This guide demonstrates the usage of the ONNX export functionality for Megatron models.

Installation
-----------------
Set up the development environment by launching the latest NeMo container

.. code-block:: bash 

    docker run -it nvcr.io/nvidia/nemo:23.04

ONNX Export of Megatron Models requires the installation of the latest versions of NeMo and TransformerEngine. The snippet below can be used to perform both installations

.. code-block:: bash 

    https://github.com/NVIDIA/TransformerEngine.git && cd TransformerEngine
    git submodule update --init --recursive
    python3 -m pip install -e .

    cd ..
    git clone https://github.com/NVIDIA/NeMo.git && cd NeMo/
    python3 -m pip install -e .

Export to ONNX
-----------------
The export script supports the ONNX export of models with .nemo and .ckpt file extensions. The script also supports the export of the following types of models: GPT, T5, BERT, BART, NMT, RETRO.
Commands for both file formats are discussed in the following sections. The model type used for the examples is GPT.


Export using .nemo file
^^^^^^^^^^^^^^^^^^^^^^^^
A model with .nemo file extension can be exported using the command below

.. code-block:: bash 

    python3 examples/nlp/language_modeling/megatron_export.py \
        model_type=gpt \
        onnx_model_file=gpt_126m.onnx \
        gpt_model_file=gpt_126m.nemo

Export using .ckpt file
^^^^^^^^^^^^^^^^^^^^^^^^
A model with .ckpt file extension can be exported using the command below

.. code-block:: bash 

    python3 examples/nlp/language_modeling/megatron_export.py \
        model_type=gpt \
        onnx_model_file=gpt_126m.onnx \
        checkpoint_dir=./gpt_126m/ \
        checkpoint_name=model_weights.ckpt \
        hparams_file=./gpt_126m/hparams.yaml