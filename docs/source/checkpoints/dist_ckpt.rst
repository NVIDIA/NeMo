Distributed checkpoints
=======================

This guide provides details about the distributed checkpoints format from Megatron-Core.


Introduction
------------

Model parallel training requires parallelism-aware checkpointing.
Megatron-Core provides a checkpointing library capable of handling all types of parallelisms used in LLMs training.
Although distributed checkpointing library is targeted at Megatron-Core model, it can be used with other models as well, provided an appropriate integration.

The library provides two main entrypoints: `dist_checkpointing.save` and `dist_checkpointing.load` which are meant to replace the `torch.save` and `torch.load` in the regular checkpointing flow.
Apart from that it provides mechanism to define different types of local tensors placement in the global checkpoint.


Basic sharding
--------------

The main way to define relationship of a plain local PyTorch tensor to tensors on other ranks is by wrapping it in a `ShardedTensor` class.
This allows to express the fact that a given local tensor is part of a larger *grid* of tensors of a given shape at a given offset.
Instead of saving a simple state dict with `torch.Tensor`s, we save a *sharded* state dict with `dist_checkpointing.ShardedTensor`s.

Example: assume we have a 128 elements tensor divided equally across the whole workload which we want to save and load with different number of ranks.

.. code-block:: python
    from pathlib import Path

    import torch

    from megatron.core import dist_checkpointing

    # Setup
    ckpt_root = Path('/tmp/checkpoints')
    native_ckpt_root = ckpt_root / 'native'
    native_ckpt_root.mkdir(exist_ok=True, parents=True)
    dist_ckpt_root = ckpt_root / 'dist_ckpt'
    dist_ckpt_root.mkdir(exist_ok=True, parents=True)

    torch.distributed.init_process_group()
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # Local tensor to save
    assert 128 % world_size == 0
    num_elems_per_rank = 128 // world_size
    local_ten = torch.arange(start=num_elems_per_rank * rank,
                             end=num_elems_per_rank * (rank + 1))

    # Native checkpoint save
    state_dict = {
        'weight': local_ten
    }
    torch.save(state_dict, native_ckpt_root / f'ckpt_{rank}.pt')

    # Distributed checkpoint save
    sharded_state_dict = {
        'weight': dist_checkpointing.ShardedTensor.from_rank_offsets('weight', local_ten, (0, rank, world_size))
    }
    dist_checkpointing.save(sharded_state_dict, dist_ckpt_root)

During load the distributed checkpoint can be easily read even if the job size changes (contrary to native checkpoints that require the same number of ranks).
The main difference wrt. `torch.load` is that the user has to provide the definition of the sharded state dict that needs to be loaded.

.. code-block:: python
    from pathlib import Path

    import torch

    from megatron.core import dist_checkpointing

    ckpt_root = Path('/tmp/checkpoints')
    dist_ckpt_root = ckpt_root / 'dist_ckpt'

    torch.distributed.init_process_group()
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    assert 128 % world_size == 0
    num_elems_per_rank = 128 // world_size

    # Local tensor to load
    local_ten = torch.empty(num_elems_per_rank)
    sharded_state_dict = {
        'weight': dist_checkpointing.ShardedTensor.from_rank_offsets('weight', local_ten, (0, rank, world_size))
    }
    loaded_state_dict = dist_checkpointing.load(sharded_state_dict, dist_ckpt_root)
    expected_local_ten = torch.arange(start=num_elems_per_rank * rank, end=num_elems_per_rank * (rank + 1))
    assert torch.all(loaded_state_dict['weight'] == expected_local_ten)

    # With torch.save and torch.load, we would have to load all files that contain
    # parts of the desired tensor in new configuration and concatenate appropriate fragments.
    # For some distributed checkpoint backends this is actually what happens underneath.


# TODO: regular vs flattened?


Supported entities
==================
The distributed checkpointing library supports saving and loading of different objects in different configurations.

A sharded state dict is a (possibly nested) Python dictionary or list with the following elements:

1. ShardedBase
    a. ShardedTensor
    #. ShardedObject
    #. ShardedTensorFactory
#. LocalNonpersitentObject
#. Arbitrary object


ShardedBase
-----------
Base class for expressing any kind of sharding.
Each sharded entity must be uniquely identified by its `key`, carry some `data` to be saved or loaded and define `replica_id` which helps identify data redundancy.


ShardedTensor
-------------
It's the primary use case of distributed checkpointing - tensors sharding.
Allows to define how PyTorch tensors are sharded across the workload.

# TODO: expand?

ShardedObject
-------------
Sometimes there is a need to save arbitrary objects across the ranks.
ShardedObject allows to structure those objects into arrays of objects with a fixed `global_shape` and save/load parts of the arrays on specific ranks.

ShardedTensorFactory
--------------------
This class allows to defer tensors transformations until the actual saving.
A factory can expand a tensor into an arbitrary sub state dict (including all supported entities listed above).
The need for such deferral will be explained in the `Optimizers`_ section.

LocalNonpersitentObject
-----------------------
This is a simple wrapper that allows to express the fact that the object wrapped with this class should end up in the final loaded state dict during loading.
During saving such objects are ignored.

Arbitrary object
----------------
All objects different than dicts, lists and the instances of the classes listed above are treated as "common" objects.

During saving, all such objects in the sharded state dict passed to `dist_checkpointing.save` are assumed to be duplicated across ranks and therefore saved only by a single coordinator rank (rank 0).

During loading, all such objects in the sharded state dict passed to `dist_checkpointing.load` are simply ignored - the loaded state dict contains only "common" objects that are were actually saved in the checkpoint.




Entrypoints
===========
There are several useful user entrypoints for checkpoint saving and loading.

dist_checkpointing.save
-----------------------
The only entrypoint for checkpoint saving.
Requires providing a sharded state dict to save and saving strategies for handling different entities (see `Save and load strategies`_ for detailed explanation).
The sharded state dict is processed in the following way:

1. The ShardedTensorFactories are applied
2. LocalNonPersistentObject are extracted from the sharded state dict and ignored
3. ShardedBase objects are extracted
4. All other objects are treated as "common" and saved according to a TODO strategy
5. All ShardedObjects are extracted from point (3) objects and saved with TODO strategy
6. All ShardedTensors are saved.
7. `metadata.json` file with backend and version metadata is saved to the checkpoint directory.

dist_checkpointing.load
-----------------------
The main entrypoint for checkpoint loading.
Requires providing a sharded state dict (in order to implicitly define mappings between local tensors and checkpoint tensors) and loading strategies.
In practice, the same sharded state dict can be usually used for both saving and loading (the sharded state dict for loading will just contain tensors with uninitialized data).

The sharded state dict provided as an input is processed in the following way:

1. "common" state dict is loaded from the checkpoint. This forms the base of the resulting state dict
2. The ShardedTensorFactories from the input sharded state dict are applied
3. LocalNonPersistentObject are extracted from the input sharded state dict, unwrapped and added to the resulting state dict
4. ShardedObjects are extracted and loaded from the checkpoint into the resulting state dict
5. ShardedTensors are extracted and loaded from the checkpoint into the resulting state dict
6. Factory merges are applied (see `Optimizers`_ for explanation)

This results in a *regular* state dict with plain tensors that can be furter processed by the application (which usually means running `model.load_state_dict(state_dict)`.


dist_checkpointing.load_common_state_dict
-----------------------------------------
TODO

dist_checkpointing.load_tensors_metadata
----------------------------------------
TODO

dist_checkpointing.load_plain_tensors
-------------------------------------
TODO

Save and load strategies
========================
There are multiple ways to save a sharded state dict into a serialized checkpoint and they can be provided by the user as saving and loading strategies.

