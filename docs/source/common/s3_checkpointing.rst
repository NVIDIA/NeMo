*********
Callbacks
*********

S3CheckpointIO
==============

This checkpoint_io is used for saving and loading files to and from S3. 
Initializing this checkpoint_io requires 

Example Usage:
.. code-block:: bash
    async_checkpointing = self.cfg.s3_checkpointing.get('enable_async_checkpointing', False)
    chunk_size_MB = self.cfg.s3_checkpointing.get('chunk_size_MB')
    max_read_concurrency = self.cfg.s3_checkpointing.get('max_read_concurrency')
    max_write_concurrency = self.cfg.s3_checkpointing.get('max_write_concurrency')
    dirpath = self.cfg.exp_manager.checkpoint_callback_params.get('dirpath')

    s3_checkpoint_io = S3CheckpointIO(dirpath=dirpath, chunk_size_MB=chunk_size_MB, max_read_concurrency=max_read_concurrency, max_write_concurrency=max_write_concurrency, async_checkpointing=async_checkpointing)

    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,
        checkpoint_io=s3_checkpoint_io,
        gradient_as_bucket_view=self.cfg.model.gradient_as_bucket_view,
        find_unused_parameters=False,
        nccl_communicator_config_path=self.cfg.model.get('nccl_communicator_config_path', None),
        sharp=self.cfg.model.get('sharp', False),
    )


Config changes:
.. code-block:: bash
    checkpoint_callback_params:
    dirpath: s3://mstar-eks-dev-us-east-2/alxzhang/nemo123/1n/checkpoints
    
    ...

    s3_checkpointing:
        # write_concurrency * tp * pp * 1.15 (buffer) should be within 3500 S3 TPS limit per partition
        max_write_concurrency: 10
        # read_concurrency * tp * pp * 1.15 (buffer) should be within 5500 S3 TPS limit per partition
        max_read_concurrency: 15
        chunk_size_MB: 64
        # enables asynchronous checkpoint writing to S3
        enable_async_checkpointing: False


S3Utils and Dependencies
========================

This utility class is used by the S3CheckpoinIO and the exp_manager to do S3-related operations. 
It has dependencies on 
1. boto3[crt]
2. s3fs==0.4.2
3. tenacity

If any of these are missing, this class can't be used. 



s3_dirpath_utils
================

Used to operate on strings by checking if they are S3 dirpaths, or convert a bucket and key into an s3 dirpath. 
This has NO dependencies on what's required for the S3Utils class, and can be used with without any new dependencies. 


S3 Demands and ExpManager Details When Running at Scale
=======================================================

When there are many ranks loading from S3, there can be slowdown or throttling errors. 
To avoid overloading S3, when resuming from a checkpoint only rank 0 needs to identify the checkpoint path and find the correct resumption file. 

.. code-block:: bash
    trainer._checkpoint_connector = NeMoCheckpointConnector(trainer)

The NeMoCheckpointConnector is defined in the exp_manager.py file, and uses the broadcasted checkpoint path founds by rank 0 on all ranks when resuming training. 

The NeMoModelCheckpoint setup() method is updated to broadcast the checkpoint path. 