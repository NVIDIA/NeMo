.. _rampup_batch_size:

Ramp Up Batch Size
------------------

Ramp up batch size is a feature that allows training to start with a smaller global batch size and linearly increase to a target global batch size over a given number of training samples with specified incremental steps.

Usage
-----

To enable global batch size rampup during training, set the rampup_batch_size parameter under the model section of training configuration. This parameter should be a list of three values:

* ``start_batch_size``: The initial batch size.
* ``batch_size_increment``: The amount by which the batch size will increase at each step.
* ``rampup_samples``: The number of training samples over which the batch size will be ramped up.

Example Configuration
---------------------

``global_batch_size:1024<br />rampup_batch_size: [256, 128, 50000000]``
In this example, the training will start with a batch size of 256, increment by 128, and reach the target global batch size of 1024 over 50,000,000 training samples.

Ramp Up Stages and Training Interruption
----------------------------------------

Once the next rampup stage is reached (the point in training when the global batch size increases), NeMo will stop the training. This pause allows you to rerun the job with a larger number of GPUs or nodes for the next stage of ramp up batch size.

Automatic Node Scheduling
-------------------------

In the `NeMo-Framework-Launcher <https://github.com/NVIDIA/NeMo-Framework-Launcher>`_, when using rampup batch size, a node scheduler is created automatically. This scheduler allows the use smaller number of nodes for smaller batch size stages and scales up according to the ``training.trainer.num_nodes`` parameter. This parameter corresponds to the maximum number of nodes you want to use for the maximum global batch size.

