****************
MSC Checkpointing
****************

MSCCheckpointIO
==============

This checkpoint_io is used for saving and loading files to and from MSC (Multi-Storage Client).
Initializing this checkpoint_io requires the dirpath be an MSC dirpath (starting with "msc://").

**Example Usage:**

.. code-block:: bash

    dirpath = "msc://profile/path/checkpoints"

    msc_checkpoint_io = MSCCheckpointIO(dirpath=dirpath)

    strategy = DDPStrategy(
        checkpoint_io=msc_checkpoint_io,
    )


Dependencies
===========

The MSCCheckpointIO depends on:

1. multi-storage-client

If this dependency is missing, the MSC checkpointing functionality cannot be used. 