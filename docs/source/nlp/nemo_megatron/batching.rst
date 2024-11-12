.. _batching:

Batching
--------

Batch size is one of the first parameters you should play with. For efficiency and convergence reasons we recommend you first try maximizing your batch size per GPU so that your GPU RAM usage is maximized.

NeMo Framework uses the following concepts.

===========================  ==================================================================================================================================
Parameter                    Description
===========================  ==================================================================================================================================
Micro Batch Size             The number of examples per data parallel rank.
Global Batch Size            ``global batch size = micro_batch_size * data_parallel_size * gradient_accumulation_steps``. For details on ``data_parallel_size`` see :doc:`../../features/parallelisms`, but typically it is equal to the number of GPUs being used.
Gradient Accumulation        Overlap gradient reduce-scatter with compute
===========================  ==================================================================================================================================


*Gradient Accumulation*

    * Idea: Train with large batch sizes with fixed memory footprint at the cost of additional compute.
    * Do k forward and backward passes through the network with different batches, do not perform parameter updates until after k passes.
    * Update paramters

