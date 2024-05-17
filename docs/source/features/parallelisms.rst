.. _parallelisms:

Parallelisms
------------

NeMo Megatron supports 5 types of parallelisms (which can be mixed together arbitrarily):

Data Parallelism
^^^^^^^^^^^^^^^^

Data Parallelism (DP) creates identical copies of the model across
multiple GPUs. Data batches are distributed between GPUs so that the
GPUs can process them independently. While compute is efficiently
distributed between GPUs, communication is required in order to keep
the model copies consistent with each other.

Distributed Data Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Distributed Data Parallelism (DDP) keeps model copies consistent by
synchronizing parameter gradients before each optimization step. More
specifically it sums gradients over all model copies using an
all-reduce communication collective.

.. image:: ../nlp/nemo_megatron/images/ddp.gif
    :align: center
    :width: 800px
    :alt: Distributed Data Parallel

Distributed Optimizer (ZeRO-1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ZeRO-1 algorithm keeps model copies consistent by sharding the
optimizer state between GPUs. During each optimization step, the
parameter gradients are first summed and sharded (with a
reduce-scatter collective), each GPU applies an optimization to its
local shard of the parameters, and the updated parameter shards are
broadcast to update all of the model copies (with an all-gather
collective). This approach is attractive for large models since
sharding the optimizer state can significantly reduce its memory
footprint on individual GPUs. It also has, in theory, the same
communication volume as DDP and its communication pattern has more
opportunities for overlapping with compute.

Enable Data Parallelism
~~~~~~~~~~~~~~~~~~~~~~~

DDP is the default parallelism scheme when NeMo is run on multiple
GPUs. Enabling other parallelism schemes in the model configuration
will decrease the size of the DP group, i.e. the number of identical
model copies.

In order to enable the distributed optimizer, set
``model.optim.name=distributed_fused_adam`` in the model
configuration. It can be configured with the following options:

| Option                      | Default   | Description                                  |
|-----------------------------|-----------|----------------------------------------------|
| ``dtype``                   | fp32      | Optimizer state datatype                     |
| ``grad_sync_dtype``         | ``dtype`` | Gradient reduce-scatter datatype             |
| ``overlap_grad_sync``       | True      | Overlap gradient reduce-scatter with compute |
| ``overlap_param_sync``      | False     | Overlap parameter all-gather with compute    |
| ``bucket_cap_mb``           | 100       | Buffer size (in MiB) for internal state and workspaces. Larger buckets have lower runtime overheads but may increase memory usage. |
| ``contiguous_param_buffer`` | False     | Allocate parameters as views into a large buffer. Helps avoid some data copies. |
| ``contiguous_grad_buffer``  | True      | Allocate parameter gradients as views into a large buffer. Helps avoid some data copies. |

See the keyword arguments in `Apex DistributedFusedAdam <https://github.com/NVIDIA/apex/blob/master/apex/contrib/optimizers/distributed_fused_adam.py>`_ and `NeMo MegatronDistributedFusedAdam <https://github.com/NVIDIA/NeMo/blob/main/nemo/core/optim/distributed_adam.py>`_ for a full list of distributed optimizer options.

Implement Data Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~

NeMo's DDP either uses PyTorch
`DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_
(default) or a custom implementation (if custom multi-precision
training is enabled with ``megatron_amp_O2``).

NeMo's distributed optimizer is built on top of
`DistributedFusedAdam <https://github.com/NVIDIA/apex/blob/master/apex/contrib/optimizers/distributed_fused_adam.py>`_
from Apex.

Tensor Parallelism
^^^^^^^^^^^^^^^^^^
With Tensor Paralellism (TP) a tensor is split into non-overlapping pieces and
different parts are distributed and processed on separate GPUs.

.. image:: ../nlp/nemo_megatron/images/tp.gif
    :align: center
    :width: 800px
    :alt: Tensor Parallel

Pipeline Parallelism
^^^^^^^^^^^^^^^^^^^^
With Pipeline Paralellism (PP) consecutive layer chunks are assigned to different GPUs.

.. image:: ../nlp/nemo_megatron/images/pp.gif
    :align: center
    :width: 800px
    :alt: Pipeline Parallel

Sequence Parallelism
^^^^^^^^^^^^^^^^^^^^

.. image:: ../nlp/nemo_megatron/images/sp.gif
    :align: center
    :width: 800px
    :alt: Sequence Parallel

Expert Parallelism
^^^^^^^^^^^^^^^^^^
Expert Paralellim (EP) distributes experts across GPUs.


.. image:: ../nlp/nemo_megatron/images/ep.png
    :align: center
    :width: 800px
    :alt: Expert Parallelism

Parallelism nomenclature
^^^^^^^^^^^^^^^^^^^^^^^^

When reading and modifying NeMo Megatron code you will encounter the following terms.

.. image:: ../nlp/nemo_megatron/images/pnom.gif
    :align: center
    :width: 800px
    :alt: Parallelism nomenclature
