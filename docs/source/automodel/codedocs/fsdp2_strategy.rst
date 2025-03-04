FSDP2Strategy Documentation
==========================

Overview
--------
The `FSDP2Strategy` class implements Fully Sharded Data Parallel (FSDP) training using PyTorch's distributed tensor APIs. This strategy is designed to optimize large-scale model training by sharding model parameters across devices.

Key Features
------------
- Data parallel and tensor parallel sharding.
- Mixed precision policy support.
- Optimizer state checkpointing and deferred restoration.
- Automatic distributed environment setup.
- Customizable model parallelization.
- Loss reduction for training, validation, and test steps.

Initialization
--------------
.. code-block:: python

    FSDP2Strategy(
        data_parallel_size="auto",
        tensor_parallel_size="auto",
        data_sampler=None,
        checkpoint_io=None,
        mp_policy=None,
        parallelize_fn=None,
        **kwargs
    )

**Parameters:**
- **data_parallel_size** (Union[Literal["auto"], int]): Number of data-parallel replicas.
- **tensor_parallel_size** (Union[Literal["auto"], int]): Number of tensor-parallel groups.
- **data_sampler** (optional): Custom data sampler.
- **checkpoint_io** (optional): Checkpoint I/O handler.
- **mp_policy** (optional): Mixed precision policy.
- **parallelize_fn** (optional): Function to parallelize the model.

Environment Setup
-----------------
.. code-block:: python

    strategy.setup_environment()

Automatically sets up the distributed environment and device mesh.

Parallelization
---------------
.. code-block:: python

    strategy.parallelize()

Applies model sharding using the specified parallelization function.

Optimizer State Restoration
---------------------------
.. code-block:: python

    strategy.load_optimizer_state_dict(checkpoint)

Stores the optimizer state from a checkpoint and defers its restoration.

Loss Reduction
--------------
.. code-block:: python

    strategy._get_loss_reduction("training")

Retrieves the appropriate loss reduction function for a given step type.

Training, Validation, and Testing Steps
---------------------------------------
.. code-block:: python

    strategy.training_step(batch, batch_idx)
    strategy.validation_step(batch, batch_idx)
    strategy.test_step(batch, batch_idx)

Defines the respective step logic with loss reduction and logging.

Checkpointing
-------------
.. code-block:: python

    strategy.save_checkpoint(checkpoint, filepath)
    strategy.load_checkpoint(filepath)

Handles model checkpoint saving and loading.

Tensor Initialization Context
-----------------------------
.. code-block:: python

    with strategy.tensor_init_context():
        # Initialize tensors
        pass

Context manager for tensor initialization.

DataLoader Processing
---------------------
.. code-block:: python

    strategy.process_dataloader(dataloader)

Applies data sampling transformations to the DataLoader.

Sharded State Dictionary
------------------------
.. code-block:: python

    strategy.load_model_state_dict(ckpt)

Shards the model's state dictionary for distributed storage.

Current Epoch Step
------------------
.. code-block:: python

    strategy.current_epoch_step

Returns the current step index within the epoch.

Removing Checkpoints
--------------------
.. code-block:: python

    strategy.remove_checkpoint(filepath)

Deletes checkpoint files from the filesystem.

Lightning Module State Dictionary
---------------------------------
.. code-block:: python

    strategy.lightning_module_state_dict()

Returns the state dictionary of the Lightning module.


