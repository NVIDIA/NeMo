.. _logging-checkpointing-label:

NeMo 2.0: Logging and Checkpointing
==================

There are three main classes in NeMo 2.0 that are responsible for configuring logging and checkpointing directories. They are:

1. :class:`~nemo.lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint`: Handles the logic that determines when to save a checkpoint. In addition, this class provides the ability to perform asynchronous checkpointing
2. :class:`~nemo.lightning.nemo_logger.NeMoLogger`: Responsible for setting logging directories and (optionally) configuring loggers.
3. :class:`~nemo.lightning.resume.AutoResume`: Sets the checkpointing directory and determines whether there is an existing checkpoint to resume from. 

Each of these classes is described in detail below. 

:class:`~nemo.lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NeMo's ``ModelCheckpoint`` callback is a wrapper around Pytorch Lightning's ``ModelCheckpoint`` and is responsible for handling the logic of when to
save and clean up checkpoints. Additionally, ``ModelCheckpoint`` supports saving a checkpoint on train_end, and the callback provides the necessary support for
asynchronous checkpointing. Below is an example of instantiating a ``ModelCheckpoint`` callback:

.. code-block:: python

    checkpoint_callback = ModelCheckpoint(
        save_best_model=True,
        save_last=True,
        monitor="reduced_train_loss",
        save_top_k=2,
        every_n_train_steps=30,
        enable_nemo_ckpt_io=False,
        dirpath='my_model_directory',
    )

Refer to NeMo Lightning's and Pytorch Lightning's ``ModelCheckpoint`` documentation for the full list of arguments that are supported by the
``ModelCheckpoint`` class. Note that ``dirpath`` is optional. If not provided, it will be determined automatically by ``AutoResume``, described
in detail in the subsequent sections. Also, note that asynchronous checkponting is set using the ``ckpt_async_save`` argument in ``MegatronStrategy``.
This attribute is then accessed by the checkpoint callback to perform async checkpointing as requested.

The ``ModelCheckpoint`` callback instance can be passed to the trainer in two ways:

1. adding the callback to the set of callbacks, and passing the callbacks directly to the trainer:

    .. code-block:: python

        callbacks = [checkpoint_callback]
        ### add any other desired callbacks...

        trainer = nl.Trainer(
            ...
            callbacks = callbacks,
            ...
        )

2. passing the callback to the ``NeMoLogger``, as described below.


:class:`~nemo.lightning.pytorch.nemo_logger.NeMoLogger`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``NeMoLogger`` class is responsible for setting the logging directories for NeMo runs. There are
a variety of arguments supported by the logger, described in detail in the ``NeMoLogger`` documentation,
but below is an example of creating a new ``NeMoLogger`` instance:

.. code-block:: python

    logger = nemo_logger = NeMoLogger(
        dir='my_logging_dir',
        name='experiment1',
        use_datetime_version=False,
        update_logger_directory=True,
    )


By default, the directory to which logs are written is ``dir / name / version``. If an
explicit version is not provided and ``use_datetime_version`` is False, the directory will instead become
``dir / name``. The  ``update_logger_directory`` argument controls whether to update the directory of the PTL loggers
to match the NeMo log dir. If set to ``True``, the PTL logger will also write to the same log directory.

As mentioned above, you can optionally pass your ``ModelCheckpoint`` instance in here as well, and the logger
will automatically set the checkpoint callback in your trainer:

.. code-block:: python

    logger = nemo_logger = NeMoLogger(
        ...
        ckpt=checkpoint_callback,
        ...
    )

Once your trainer has been initialized, the ``NeMoLogger`` can be setup using the following command:


.. code-block:: python

    nemo_logger.setup(
        trainer,
        resume_if_exists,
    )

``resume_if_exists`` is a boolean indicating whether to resume from the latest checkpoint if
one is available. The value of ``resume_if_exists`` should match the value passed into ``AutoResume``
as described below.

:class:`~nemo.lightning.pytorch.resume.AutoResume`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``AutoResume`` class is responsible for setting checkpoint paths and checking whether there
are existing checkpoints to restore from. Example usage is as follows:

.. code-block:: python

    resume = AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
        dirpath="checkpoint_dir_to_resume_from"
    )

If no ``dirpath`` is provided, the directory to resume from will become ``log_dir / checkpoints``, where
``log_dir`` is set from the nemo logger instance as described in the previous section. ``resume_ignore_no_checkpoint``
determines the behavior if ``resume_if_exists`` is ``True`` but no checkpoint is found in the checkpointing
directory. ``resume_if_exists`` should  match the argument passed into the nemo logger's setup. 

``AutoResume`` should be set up in a similar fashion to ``NeMoLogger``.

.. code-block:: python

    resume.setup(trainer, model)


Passing a model into the setup is optional. It is required only when importing a checkpoint from HF or other non-NeMo checkpoint formats.


Putting it all together
^^^^^^^^^^^^^^^^^^^^^^^

Putting it all together, setting up loggers and checkpointers in NeMo 2.0 looks something like 

.. code-block:: python

    checkpoint_callback = ModelCheckpoint(
        save_best_model=True,
        save_last=True,
        monitor="reduced_train_loss",
        save_top_k=2,
        every_n_train_steps=30,
        enable_nemo_ckpt_io=False,
        dirpath='my_model_directory',
    )

    logger = nemo_logger = NeMoLogger(
        dir='my_logging_dir',
        name='experiment1',
        use_datetime_version=False,
        update_logger_directory=True,
        ckpt=checkpoint_callback,
    )

    resume = AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
    )

    ### setup your trainer here ###

    nemo_logger.setup(
        trainer,
        etattr(resume, "resume_if_exists", False),
    )
    resume.setup(trainer)
