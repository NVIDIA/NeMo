Callbacks
=========
NeMo uses callbacks to do a variety of helper functions during training.
NeMo comes with three useful callbacks: SimpleLossLoggerCallback, 
CheckpointCallback, and EvaluatorCallback. Callbacks are defined prior to
calling the train() function, and are passed to the train() function.
For example, a common training script will look like:

.. code-block:: python

    loggercallback = nemo.core.SimpleLossLoggerCallback(...)
    savercallback = nemo.core.CheckpointCallback(...)
    evalcallback = nemo.core.EvaluatorCallback(...)

    nf.train(
        callbacks=[loggercallback, savercallback, evalcallback],
        ...)

SimpleLossLoggerCallback
------------------------
SimpleLossLoggerCallback is used to
log training metrics such as loss and time per step to screen and to
tensorboard. SimpleLossLoggerCallback has one required arguments, and two
arguments that we recommend to overwrite. It requires tensors which is a list
of NMTensors that print_func(), get_tb_values(), and log_to_tb_func will
receive as input during
training. The two reccomended arguments to override are print_func(), and
either get_tb_values() or log_to_tb_func().

print_func() should be used to log values to screen. We recommend using
nemo.logging.info() in place
of logging.info(). For example, it can be used to print the loss value:

.. code-block:: python

    def my_print_func(tensors):
        nemo.logging.info(f"Loss {tensors[0]}")

We provide two methods to log to tensorboard: get_tb_values() and
log_to_tb_func(). For simple use case of logging scalars, we recommend
get_tb_values(). For advanced use cases such as pictures or audio, we
recommend log_to_tb_func().

get_tb_values() is used to return values to be logged to tensorboard. It should
return a list of 2-element tuples, where the first element is a string
representing the tensorboard tag, and the second element is the scalar value to
log. Note we currently only support scalar values. Note to use get_tb_values(),
tb_writer should also be defined.

.. code-block:: python

    def my_get_tb_values(tensors):
        return [("Train_Loss", tensors[0])]

log_to_tb_func() takes three arguments: the
`tensorboardX.SummaryWriter <https://tensorboardx.readthedocs.io/en/latest/tensorboard.html>`_
, a list of evaluated tensors, and the current step. The user can then use the
SummaryWriter class to add images, audio, and more. For example:

.. code-block:: python

    def log_to_tb_func(swriter, tensors, step):
        swriter.add_scalar("Train_Loss", tensors[0], step)
        swriter.add_audio("Train_Sample", tensors[1][0], step)

SimpleLossLoggerCallback can be constructed as follows:

.. code-block:: python

    from functools import partial

    loggercallback = nemo.core.SimpleLossLoggerCallback(
        # Define tensors that we want to pass to print_func, and get_tb_values
        tensors=[train_loss],
        # Pass the print function that we want to use
        print_func=my_print_func,
        # Pass the function that returns tensorboard tags and scalars
        get_tb_values=my_get_tb_values,
        # How often we want to call this callback
        step_freq=500,
        # The tensorboard writer object we want to use, it should be
        # automatically created by neural_factory if create_tb_writer was
        # set to True during neural_factory construction
        tb_writer=neural_factory.tb_writer)
    )

CheckpointCallback
------------------
CheckpointCallback is used to checkpoint models during training so that
they can be reloaded later for inference or finetuning. CheckpointCallback
is simple to use:

.. code-block:: python

    from functools import partial

    loggercallback = nemo.core.CheckpointCallback(
        # The folder to save checkpoints
        # Note: Neural Factory automatically creates a checkpoint folder
        folder=neural_factory.checkpoint_dir,
        # If None, CheckpointCallback will attempt to load from folder
        # at the beginning of training.
        # Else, CheckpointCallback will attempt to load from load_from_folder
        load_from_folder=None,
        # Checkpointing frequency in steps
        step_freq=-1,
        # Checkpointing frequency in epochs
        epoch_freq=-1,
        # Number of checkpoints to keep
        checkpoints_to_keep=4,
        # If True, CheckpointCallback will raise an Error if restoring fails
        force_load=False
    )

EvaluatorCallback
-----------------
EvaluatorCallback is used during evaluation to log evaluation
metrics to screen and tensorboard. EvaluatorCallback requires three arguments:
eval_tensors, user_iter_callback, user_epochs_done_callback. Similar to
SimpleLossLoggerCallback, eval_tensors is a list of NMTensors whose values
we want to obtain during evaluation.

user_iter_callback is a function that is called after each batch during
evaluation. It is always passed two arguments: values_dict, and global_var_dict.
values_dict is a dictionary containing NMTensor names as keys, and the evaluated
tensor as values for that batch. It's main job is to copy the relevant evaluated
tensors from values_dict to global_var_dict as global_var_dict is saved
between batches and passed to the final user_epochs_done_callback function.

user_epochs_done_callback is a function that accepts global_var_dict. It's job
is to log relevant information to the screen such as the evaluation loss.

For simple logging of scalar values to tensorboard, user_epochs_done_callback
should return a dictionary with strings as keys and scalar tensors as values.
This tag -> value dictionary will be parsed and each element will be logged
to tensorboard if a tensorboard writter object is declared.

To enable more complex tensorboard logging such as images or audio,
EvaluatorCallback must be passed tb_writer_func at initialization. This
function must accept a
`tensorboardX.SummaryWriter <https://tensorboardx.readthedocs.io/en/latest/tensorboard.html>`_
, whatever is returned from user_epochs_done_callback, and the current step.
We recommend for user_epochs_done_callback to simply return the global_var_dict
for tb_writer_func to consume. The user must log all data of interest inside
tb_writer_func including scalars that would otherwise be logged if
tb_writer_func was not passed to EvaluatorCallback.

For an example, please see the scripts inside <nemo_dir>/examples.