Core Concepts
=============

Neural Module
~~~~~~~~~~~~~
Neural Modules are building blocks for Models.
They accept (typed) inputs and return (typed) outputs. *All Neural Modules inherit from ``torch.nn.Module`` and, therefore, compatible with PyTorch ecosystem.* There are 3 types on Neural Modules:

    * Regular modules
    * Dataset/IterableDataset
    * Losses

Model
~~~~~
NeMo Model is an entity which contains 100% of information necessary to invoke training/fine-tuning.
It is based on Pytorch Lightning's LightningModule and as such contains information on:

    * Neural Network architecture, including necessary pre- and post- processing
    * How data is handled for training/validation/testing
    * Optimization, learning rate schedules, scaling, etc.

Neural Types
~~~~~~~~~~~~

Neural Types perform semantic checks for modules and models inputs/outputs. They contain information about:

    * Semantics of what is stored in the tensors. For example, logits, logprobs, audiosignal, embeddings, etc.
    * Axes layout, semantic and (optionally) dimensionality. For example: [Batch, Time, Channel]