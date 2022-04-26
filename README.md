# BigNLP HP Tool

This tool searches for the Hyper-Parameters (HPs) that achieve the highest throughput for training 
Large Language Models (LLMs) using NeMo-Megatron. It also searches for the inference HPs that 
achieve the highest throughput and the lowest latency.

## Table of contents
- [1. HP Tool Capabilities](#1-hp-tool-capabilities)


## 1. HP Tool Capabilities

The Hyper-Parameter (HP) tool is intended to quickly iterate over different model configurations, 
to quickly find the best configuration with minimal time and money spending. To achieve that, our 
tool goes through different steps, as shown in the table below:

| Feature                    | GPT-3 | T5 | mT5  |
| -------------------------- | ----------------- |
| Model Size Recommendation  | Yes   | Yes | Yes |
| Base Config Generation     | Yes   | Yes | Yes |
| Training HP Search         | Yes   | Yes | Yes |
| Inference HP Search        | Yes   | No  | No  |

### 1.1. Model Size Recommendation
For users who do not know what model size they wish to train, our tool is capable of recommending 
a model size, given the hardware and training constraints. If the number of GPUs, the TFLOPS per GPU, 
the maximum time to train, and the number of tokens to train for are known, then our tool can 
recommend a model size that can be trained with the specified hardware and time constraints.

For example, if the user has 20 NVIDIA DGX nodes available (80GB GPU memory), and wants to train a 
GPT-3 model for a maximum of 5 days, the tool will recommend using a 5B parameter GPT-3 model. 
The tool will perform a best effort guess using heuristics, so the results might not be perfect.


### 1.2. Base Config Generation
If the model size is provided by the user, or after the model size is generated (as shown in section 1.1), 
the tool will generate a base configuration for the given model. This configuration will be a valid, 
runnable configuration in YAML format, which can be trained using NeMo-Megatron. However, this config 
will not be optimized at this stage.


### 1.3. Training HP Search
Given the input model size (generated in step 1.1) and the base configuration (generated in step 1.2), 
the tool will now search over four different critical Hyper-Parameters, that have great impact on the 
training throughput. These parameters are: Tensor Parallelism (TP), Pipeline Parallelism (PP), Micro 
Batch Size (MBS), and Activation Checkpointing Layers (ActCkpt).

First, the tool will use heuristics to choose good candidates for those four parameters to generate 
the grid of candidate configurations. All the candidate configs will be saved to the results directory, 
and will include YAML files with the corresponding config. NOTE: some of these configs might not work, 
due to high memory usage or for other reasons. The next step will determine which configs are valid.

Once all the candidate configs are generated, the tool will use heuristics to sort the most promising 
candidate configs. Then, the tool will launch the top `limit_search_runs` most promising candidates, 
to perform a grid search over the four training parameters. This search will launch the jobs using 
NeMo-Megatron, and it will train each config for `max_minutes_per_run` minutes, on the target cluster. 
During this search, the jobs will run in the minimum number of nodes required, using Data Parallelism 
of 1 (DP=1) in most cases.


### 1.4. Inference HP Search
A


## 2. Usage
A

### 2.1. Pre-Defined Configs
The pre-defined configs we provide have been well tested.


### 2.2. Custom Model Size
A


### 2.3. Base Config Generation
A

### 2.4. Training HP Search
A

### 2.5. Inference HP Search
A



