Checkpoints
===========

Checkpoints
===========

In this section, we present key functionalities of NVIDIA NeMo related to checkpoint management.

Understanding Checkpoint Formats
--------------------------------

A ``.nemo`` checkpoint is fundamentally a tar file that bundles the model configurations (given as a YAML file), model weights, and other artifacts like tokenizer models or vocabulary files. This consolidated design streamlines sharing, loading, tuning, evaluating, and inference.

Contrarily, the ``.ckpt`` file, created during PyTorch Lightning training, encompasses both the model weights and the optimizer states, usually employed to pick up training from a pause.

Sharded Model Weights
---------------------

Within ``.nemo`` or ``.ckpt`` checkpoints, the model weights could be saved in either a regular format (one file called ``model_weights.ckpt`` inside model parallelism folders) or a sharded format (a folder called ``model_weights``).

With sharded model weights, you can save and load the state of your training script with multiple GPUs or nodes more efficiently and avoid the need to change model partitions when you resume tuning with a different model parallelism setup.

NeMo supports the distributed (sharded) checkpoint format from Megatron-Core. In Megatron-Core, it supports two backends: Zarr-based and PyTorch-based.

Community Checkpoint Converter
-----------------------------
We provide easy-to-use tools that enable users to convert community checkpoints into the NeMo format. These tools facilitate various operations, including resuming training, Sparse Fine-Tuning (SFT), Parameter-Efficient Fine-Tuning (PEFT), and deployment. For detailed instructions and guidelines, please refer to our documentation.

We offer comprehensive guides to assist both end users and developers:

- **User Guide**: Detailed steps on how to convert community model checkpoints for further training or deployment within NeMo. For more information, please see our :doc:`user_guide`.

- **Developer Guide**: Instructions for developers on how to implement converters for community model checkpoints, allowing for broader compatibility and integration within the NeMo ecosystem. For development details, refer to our :doc:`dev_guide`.

- **Megatron-LM Checkpoint Conversion**: NVIDIA NeMo and NVIDIA Megatron-LM share several foundational technologies. You can convert your GPT-style model checkpoints trained with Megatron-LM into the NeMo Framework using our scripts, see our :doc:`convert_mlm`.

Access the user and developer guides directly through the links below:

.. toctree::
   :maxdepth: 1
   :caption: Conversion Guides

   user_guide
   dev_guide
   convert_mlm
