Callbacks
=========
NeMo has a callback system that can be used to inject user code and logic inside its training loop. NeMo's callbacks
defines the following events that users can inject into:

.. code-block::

    -- on_action_start
    ---- on_epoch_start
    ------ on_step_start
    -------- on_batch_start
    -------- on_batch_end
    ------ on_step_end
    ---- on_epoch_end
    -- on_action_end

At a high level, the NeMo training loop looks like this:

.. code-block:: python

    def train():
        ...  # Do initialization of optimizers, amp, ddp, etc
        # Initialize the state passed to callbacks and the tensor state object to be empty
        state = {}
        state["tensors"] = TrainingState({})

        callbacks.on_action_start(state)  # For all callbacks, trigger the on_action_start event
        for epoch in range(num_epochs):  # Or until max_steps
            callbacks.on_epoch_start(state)  # Trigger the on_epoch_start event
            batch_counter = 0
            for data in dataloader:  # Fetch batches of data
                if batch_counter == 0:
                    callbacks.on_step_start(state)  # Trigger the on_step_start event
                callbacks.on_batch_start(state)  # Trigger the on_batch_start event

                ...  # Forward pass
                final_loss.backwards()
                # Set the `loss` key inside the tensor state object to be the loss that we call backwards() on
                state["tensors"]["loss"] = final_loss

                callbacks.on_batch_end(state)  # Trigger the on_batch_end event
                batch_counter += 1
                if batch_counter == gradient_accumulation_steps:
                    batch_counter = 0  # Reset batch counter
                    # By default, gradient_accumulation_steps = 1. Note this is passed to train() as batches_per_step
                    # and sometimes exposed as args.iter_per_step
                    optimizer.step()
                    callbacks.on_step_end(state)  # Trigger the on_step_end event

                # Clear the tensor state object
                clear state["tensors"]
            callbacks.on_epoch_end(state)  # Trigger the on_epoch_end event
        callbacks.on_action_end(state)  # Trigger the on_action_end event

.. note::
    NeMo's callbacks were updated in version 0.11. For documentation on the old callbacks, please see
    :ref:`callbacks0.10`. For an update guide from version 0.10 to version 0.11, please see
    :ref:`callbacks0.10update`.

Built-in Callbacks
------------------

NeMo offers the following built-in callbacks that users can use:

    - :class:`nemo.core.callbacks.SimpleLogger` is a simple callback that prints information to screen. By default,
      it logs the training loss every 100 steps.
    - :class:`nemo.core.callbacks.TensorboardLogger` is a callback that logs information to tensorboard. Note that the
      ``TensorboardSummaryWriter`` class needs to be passed to this callback. Be default, it logs the training loss and
      learning rate every 100 steps, the number of epochs completed, and the epoch training time.
    - :class:`nemo.core.callbacks.WandBLogger` is a callback that logs information to
      `Weights & Biases <https://docs.wandb.com/>`_. Make sure wandb is installed and you did ``wandb login``. It is
      recommended to pass the ``wandb_name`` and ``wandb_project`` arguments to the constructor. By default, it logs
      the training loss and learning rate every 100 steps, the number of epochs completed, and the epoch training time.
    - :class:`nemo.core.callbacks.CheckpointCallback` is a callback that saves trainer and module checkpoints every
      ``step_freq`` or ``epoch_freq``. The directory that the callback will save to must be passed as ``folder``.

In order to log additional tensors to screen, and additional tensor scalars to tensorboard and Weights and Biases, one
can simply add these tensors to the ``tensors_to_log`` parameters of the relevant callbacks. For example:

.. code-block:: python

    ...
    # Assuming that you have a network defined above that produces a predictions tensor and instantiated
    # a MyLossModule and a MyMetricModule.
    loss = MyLossModule(targets=targets, predictions=predictions)
    precision, recall, F1 = MyMetricModule(targets=targets, predictions=predictions)

    # If desired, users can assign a string name to tensors for easy reference in callbacks
    precision.rename("p")
    # Note that the name "loss" is reserved for the training loss

    callbacks = [
        # Create a callback that prints the loss to screen every 10 steps
        # By default tensors_to_log is ["loss"], so there is no need to pass that
        SimpleLogger(step_freq=10),
        # Create the tensorboard callback by passing the Tensorboard SummaryWriter object, and telling it to log
        # loss and precision.
        TensorboardLogger(nf.tb_writer, tensors_to_log=["loss", "p"]),
        # Create the Weights and Biases callback by giving it a name and project, and tell it to log the loss, F1
        # and recall scores. Note that tensors_to_log also accepts the NmTensors themselves or their unique_names
        # in addition to any renaming that users do
        WandBLogger(wandb_name="my_exp", wandb_project="my_proj", tensors_to_log=[loss, F1, recall.unique_name])
    ]

    nf.train(
        tensors_to_optimize=[loss],
        callbacks=callbacks,
        ...  # Other train() parameters
    )

.. tip::
    For more advanced logging of non-scalars such as images and audio to tensorboard, please take a look at the
    documentation and code for :class:`nemo.core.callbacks.TensorboardLogger`

.. _callback-creation:

Creating Your Own Callback
--------------------------
For more advanced user-cases where users want to inject their own logic not offered by NeMo's built-in callbacks, NeMo
allows users to defined their own callbacks via two methods. The first method is to create a child of the
:class:`nemo.core.callbacks.NeMoCallback` and define any of the methods
(``on_action_start, on_epoch_start, ..., on_action_end``) inside the child class. The second method is to use our
function decorators for each of those events such as :meth:`nemo.core.callbacks.on_step_start`. Regardless of the method
chosen, both interact with the NeMo trainer through the ``state`` dictionary. We will first detail the ``state``
dictionary and then provide examples for creating a callback through the decorator method and through the child
class method.

NeMo provides callbacks with access to the ``state`` dictionary as defined in the StateWrapper class inside of
nemo.backends.pytorch.actions.py. The dictionary contains the following key-value pairs:

    - "step" (int): the current step number
    - "epoch" (int): the current epoch
    - "local_rank" (int): the local_rank of the current process. Defaults to None for single-gpu or cpu runs
    - "global_rank" (int): the global_rank of the current process. Defaults to None for single-gpu or cpu runs
    - "optimizers" (list of pytorch.optimizers): a list of the current pytorch optimizers used in the train action.
      In most cases, it is a list of 1 optimizer. Note that the current learning rate can be extracted from the
      optimizer. See the tensorboard callback to see how this is done.
    - "tensors": a :class:`nemo.core.actions.TrainingState` instance. This class has the
      :meth:`nemo.core.actions.TrainingState.get_tensor` function that takes ``name``: either the user-renamed string,
      a NmTensor's unique_name, or a NmTensor and returns the associated pytorch tensor.

Users can use NeMo's callback function decorators to easily inject logic inside the training process that doesn't need
to keep state. For example, let's say we want to compute the confusion_matrix every 150 steps using
``sklearn.metrics.confusion_matrix``:

.. code-block:: python

    # Assume 'labels' is defined before this
    inputs, targets = MyDataLayerNM()
    predictions = MyNeuralNetworkNM(inputs=inputs)
    loss = MyLossNM(inputs=inputs, targets=targets)

    # Use the callback function decorator
    @nemo.core.callbacks.on_step_end
    # Define your function that accepts the input argument 'state'
    def print_confusion_matrix(state):
        if state["step"] % 150 == 0:  # Log once every 150 steps
            # Use the get_tensor method of state["tensors"] to get the pytorch tensor associated with the
            # `target` NmTensor
            targets_value = state["tensors"].get_tensor(target)
            predictions_value = state["tensors"].get_tensor(predictions)
            confusion_matrix = sklearn.metrics.confusion_matrix(targets_value, predictions_value, labels)
            logging.info(confusion_matrix)

    nf.train(callbacks=[print_confusion_matrix], ...)  # Pass the function to the callbacks arg of train()

Users can also create a child class of :class:`nemo.core.callbacks.NeMoCallback`. This method is useful when users want
to store state inside a class variable that they can access from multiple callback hooks. For example, here is a
callback that keeps the exponential moving average of the step time:

.. code-block:: python

    class StepTimeTracker(nemo.core.callbacks.NeMoCallback):
        def __init__(self, decay=0.99):
            self._decay = decay
            self._ema_step_time = 0
            self._step_start_time = 0

        # Note that even you do not use state, your functions must accept 1 positional argument
        def on_step_start(self, state):
            # Store current starting time in `self._step_start_time`
            self._step_start_time = time.time()

        def on_step_end(self, state):
            # Calculate step duration
            step_duration = time.time() - self._step_start_time

            # Apply exponential moving average
            self._ema_step_time = self._decay * self._ema_step_time + (1 - self._decay) * step_duration

    nf.train(callbacks=[StepTimeTracker()], ...)  # Pass your callback class to train()
