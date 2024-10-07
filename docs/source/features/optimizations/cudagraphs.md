# Cudagraphs for NeMo

###  Background and Motivation
Cudagraphs are a performance feature used mainly to eliminate reduce the presence of GPU idling due to host overhead and jitter. Commonly, if the host cannot issue kernels fast enough, the GPU may empty its work queue and begin idling. Cudagraphs solve this by encapsulating a sequence of many kernels into single unit. Thus, the host can launch multiple kernels by calling a single cudagraph, which greatly reduces the amount of work required by the host in comparison to eagerly launching each kernel individually.
The current cudagraph implementation in NeMo maps each transformer layer’s forward and backpasses to a cudagraph. On the first step, `CudaGraphManager` intercepts the forward pass to each transformer layer and uses it to trace the transformer layer via stream capture. On subsequent steps, `CudagraphManager`, rather than of executing the layer eagerly, reroutes calls into a single cudagraph, 
Currently, cudagraphs increases the memory usage of activations by ~20%. End to end, this is roughly an increase of about 10-20GB. The reason for this increase is due to cudagraphs preallocating any memory used by temporary tensors in the forward pass. As a result, temporary tensors that normally are deallocated when dereferenced are kept allocated for the lifespan of the cudagraph.
The implementation of cudagraphs for NeMo can be found in [ megatron/core/transformer/cuda_graphs.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/cuda_graphs.py).

### When to use
Cudagraphs are recommended to be used when host overheads are expected to be significant, for instance systems with weak single-threaded CPU performance.
As a demonstration we show that with a GPT3 20B model, cudagraphs can improve performance by 11.4%. The increase in memory usage due to cudagraphs was observed to be 15GB.
| Setting         | TFLOP/s |
| -----         | ------ |
| No Cudagraphs      | 750     |
| Cudagraphs     | 836     |
- Setup: 16x GH200, MBS=2, TP=4, FP8 precision




### Usage
As of the NeMo 24.09 release, Cudagraphs is currently supported only for the pretraining of dense models. 
To enable please add the following configs:
`model.enable_cuda_graph=True `
`model.use_te_rng_tracker=True `
`model.get_attention_mask_from_fusion=True`

