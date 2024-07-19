# NeMo Lightning

This directory provides custom PyTorch Lightning-compatible objects for seamlessly training NeMo 2.0 models using PTL. NeMo 2.0 models
are implemented in [Megatron Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core), and NeMo Lightning provides the bridge between higher-level object-oriented PTL APIs and lower-level Megatron APIs. 
For detailed tutorials and documentation of NeMo 2, please refer to the NeMo 2.0 [docs](../../../docs/source/2.0) (## TODO: update this hyperlink).

Some of the useful classes provided here are:
- `MegatronStrategy`: A PTL strategy that enables training of Megatron models on NVIDIA GPUs. More information on `MegatronStrategy` can be found here (TODO: link to doc).
- `MegatronParallel`: Class which sets up and manages Megatron's distributed model parallelism. Refer to this document for more information (TODO: Link to doc).
- `MegatronMixedPrecision`: A specialized precision plugin for training Megatron-based models in PTL. More information can be found here (TODO: link to doc).
- `Trainer`: A lightweight wrapper around PTL's `Trainer` object which provides some additional support for serializing train states.