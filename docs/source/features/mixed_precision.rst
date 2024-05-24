.. _mix_precision:

Mixed Precision Training
------------------------

Mixed precision training significantly enhances computational efficiency by conducting operations in half-precision and fp8 formats, while selectively maintaining minimal data in single-precision to preserve critical information throughout key areas of the network. NeMo now supports FP16, BF16, and FP8 (via Transformer Engine) across most models. Further details will be provided shortly.


FP8 usage
=========

Overview
^^^^^^^^

NVIDIA's H100 GPU introduced support for a new datatype, FP8 (8-bit floating point), enabling higher throughput of matrix multiplies and convolutions. NeMo uses NVIDIA's `TransformerEngine <https://github.com/NVIDIA/TransformerEngine>`_ (TE) in order to leverage speedups from FP8. The following table summarizes the FP8 related arguments that can be configured in NeMo (`example config setting <https://github.com/NVIDIA/NeMo/blob/2e1814c9f031ad2aeeebad44597365e97253d2c4/examples/nlp/language_modeling/conf/megatron_gpt_config.yaml/#L192-L200>`_). For a more detailed overview, refer to the TE `documentation <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html>`_, specifically the FP8 `format <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html#transformer_engine.common.recipe.Format>`_ and `recipe <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html#transformer_engine.common.recipe.DelayedScaling>`_.

.. list-table:: FP8 arguments
   :widths: 25 25 50
   :header-rows: 1

   * - Argument
     - Description
   * - transformer_engine
     - Boolean argument that needs to be set in order to enable TE and related functionality. All following arguments are ignored if this is not set to True.
   * - fp8
     - Enables FP8 training. For transformer networks, the QKV, projection, FC1, and FC2 matrix multiplications are executed using the 4th generation H100 tensor cores with FP8 support.
   * - fp8_e4m3
     - Training recipe format for FP8. Activations, weights, and gradient tensors use the E4M3 format.
   * - fp8_hybrid
     - Training recipe format for FP8. Activations and weight tensors use the E4M3 format, whereas gradient use the E5M2 format to satisfy the additional dynamic range requirement for backward tensors. This is the default setting.
   * - fp8_margin
     - The scaling factor for FP8 tensors can be shifted by a factor of $2 ^ {margin}$ using this argument.
   * - fp8_amax_history_len
     - Window size for the amax history. This many instances of most recent amaxes (absolute max) are stored per tensor.
   * - fp8_amax_compute_algo
     - "max" or "most_recent". Specifies how to choose an amax from the given history.
   * - reduce_amax
     - Whether or not to perform an allreduce on the amax (absolute max) values for the FP8 tensors. Since the amax is directly used to compute the scaling factor for FP8 tensors, setting this ensures that the scaling factors for a tensor remain synchronized across devices in multi-GPU training configurations.
   * - fp8_params
     - Whether or not to store module level parameters in FP8. Setting this may result in a lower memory consumption since this eliminates the need to store a copy of weights in higher precision (> half) for cases in which these are maintained externally anyway, such as master parameters in the optimizer. For details, refer to the `fp8_model_init <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html#transformer_engine.pytorch.fp8_model_init>`_ API in TE.

Resources
^^^^^^^^^

- `TE documentation <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html>`_.
- `Intro to FP8, floating point formats, and mixed precision training <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html#Introduction-to-FP8>`_.
- `Performance optimizations <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/advanced_optimizations.html>`_ that are natively supported in NeMo by enabling FP8 training with TE.
- `TE installation <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html>`_.
