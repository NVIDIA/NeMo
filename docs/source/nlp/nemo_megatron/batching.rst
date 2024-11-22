.. _batching:

Batching
--------

Batch size is one of the first parameters you should play with. For efficiency and convergence reasons we recommend you first try maximizing your batch size per GPU so that your GPU RAM usage is maximized.

NeMo Framework uses the following concepts:

===========================  ==================================================================================================================================
Parameter                    Description
===========================  ==================================================================================================================================
Micro Batch Size             The number of examples per data parallel rank.
Global Batch Size            ``global batch size = micro_batch_size * data_parallel_size * gradient_accumulation_steps``. For details on ``data_parallel_size`` see `this page <https://github.com/NVIDIA/NeMo/blob/main/docs/source/features/parallelisms.rst>`_.
Gradient Accumulation        Supports training with large batch sizes with fixed memory footprint at the cost of additional compute. ``accumulate_grad_batches`` is automatically handled by PyTorch Lightning.
===========================  ==================================================================================================================================

Usage
^^^^^

       .. code-block:: python

              from nemo.collections import llm
              from functools import partial

              # Load train recipe
              recipe = partial(llm.llama3_8b.pretrain_recipe)()
              
              # Set micro and global batch size
              recipe.data.micro_batch_size = 4
              recipe.data.global_batch_size = 16
              
              # Set accumulate_grad_batches
              recipe.trainer.accumulate_grad_batches = 1

Set batching parameters directly from CLI:

       .. code-block:: bash

              nemo llm pretrain --factory llama3_8b data.micro_batch_size=4 data.global_batch_size=16 trainer.accumulate_grad_batches=1

