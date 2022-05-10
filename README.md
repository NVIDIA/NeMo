# BigNLP HP Tool

This tool searches for the Hyper-Parameters (HPs) that achieve the highest throughput for training 
Large Language Models (LLMs) using NeMo-Megatron. It also searches for the inference HPs that 
achieve the highest throughput and the lowest latency.

## Table of contents
- [1. HP Tool Capabilities](#1-hp-tool-capabilities)
    - [1.1. Model Size Recommendation](#11-model-size-recommendation)
    - [1.2. Base Config Generation](#12-base-config-generation)
    - [1.3. Training HP Search](#13-training-hp-search)
    - [1.4. Inference HP Search](#14-inference-hp-search)
- [2. Usage](#2-usage)
    - [2.1. General Configuration](#21-general-configuration)
    - [2.2. Running Pre-Defined Configs](#22-running-pre-defined-configs)
        - [2.2.1. Model Config](#221-model-config)
        - [2.2.2. Base Config Generation](#222-base-config-generation)
        - [2.2.3. Training HP Search](#223-training-hp-search)
        - [2.2.4. Inference HP Search](#224-inference-hp-search)
    - [2.3. Running Custom Model Size Configs](#23-running-custom-model-size-configs)
    - [2.4. Interpreting the Results](#24-interpreting-the-results)
    - [2.5. Logging Runs with Weights and Biases](#25-logging-runs-with-weights-and-biases)

## 1. HP Tool Capabilities
<a id="markdown-hp-tool-capabilities" name="hp-tool-capabilities"></a>

The Hyper-Parameter (HP) tool is intended to quickly iterate over different model configurations, 
to quickly find the best configuration with minimal time and money spending. To achieve that, our 
tool provides several different capabilities, as shown in the table below:

| Feature                              | GPT-3 | T5  | mT5 |
| ------------------------------------ | ----- | --- | --- |
| Model Size Recommendation            | Yes   | Yes | Yes |
| Base Config Generation               | Yes   | Yes | Yes |
| Training HP Search                   | Yes   | Yes | Yes |
| Parallel Training HP Search          | Yes   | Yes | Yes |
| Inference HP Search                  | Yes   | No  | No  |
| Parallel Inference HP Search         | No    | No  | No  |
| SLURM Based Clusters                 | Yes   | Yes | Yes |
| Base Command Platform Based Clusters | No    | No  | No  |

### 1.1. Model Size Recommendation
<a id="markdown-model-size-recommendation" name="model-size-recommendation"></a>

For users who do not know what model size they wish to train, our tool is capable of recommending 
a model size, given the hardware and training constraints. If the number of GPUs, the TFLOPS per GPU, 
the maximum time to train, and the number of tokens to train for are known, then our tool can 
recommend a model size that can be trained with the specified hardware and time constraints.

For example, if the user has 20 NVIDIA DGX nodes available (80GB GPU memory), and wants to train a 
GPT-3 model for a maximum of 5 days, the tool will recommend using a 5B parameter GPT-3 model. 
The tool will perform a best effort guess using heuristics, so the results might not be perfect.


### 1.2. Base Config Generation
<a id="markdown-base-config-generation" name="base-config-generation"></a>

If the model size is provided by the user, or after the model size is generated (as shown in section 1.1), 
the tool will generate a base configuration for the given model. This configuration will be a valid, 
runnable configuration in YAML format, which can be trained using NeMo-Megatron. However, this config 
will not be optimized at this stage.


### 1.3. Training HP Search
<a id="markdown-training-hp-search" name="training-hp-search"></a>

Given the input model size (generated in step 1.1) and the base configuration (generated in step 1.2), 
the tool will now search over four different critical Hyper-Parameters, that have great impact on the 
training throughput: Tensor Parallelism (TP), Pipeline Parallelism (PP), Micro Batch Size (MBS), 
and Activation Checkpointing Layers (ActCkpt).

First, the tool will use heuristics to choose good candidates for those four parameters to generate 
the grid of candidate configurations. All the candidate configs will be saved to the results directory, 
and will include YAML files with the corresponding config. NOTE: some of these configs might not work, 
due to high memory usage or for other reasons. The next step will determine which configs are valid.

Once all the candidate configs are generated, the tool will use heuristics to sort the most promising 
candidate configs. Then, the tool will launch the top `limit_search_runs` most promising candidates 
in parallel, to perform a grid search over the four training parameters. This search will launch the 
jobs using NeMo-Megatron, and it will train each config for `max_minutes_per_run` minutes, on the 
target cluster. During this search, the jobs will run in the minimum number of nodes required, using 
Data Parallelism of 1 (DP=1) in most cases.


### 1.4. Inference HP Search
<a id="markdown-inference-hp-search" name="inference-hp-search"></a>

The tool can also search the best HPs for inference purposes. It will empirically measure the 
throughput and latency for each given configuration in the grid search space, and return a comprehensive 
table with all the numbers. The tool will search over three different critical HPs, which have great 
impact on the inference throughput and latency: Tensor Parallelism (TP), Pipeline Parallelism (PP), and 
Batch Size (BS). Technically, the tool is also capable of searching over different input/outpu sequence 
lengths. However, we do not recommend adding multiple different sequence lengths to the same search, 
since the model that uses the shortest sequence lengths will always achieve higher throughput and lower 
latency. Therefore, we recommend performing several different inference searches for different sequence 
lengths.

Once the search space has been defined, the tool will launch a job for each config, and measure the 
throughput and latency.i This search will launch the jobs using NeMo-Megatron on the target cluster. 
Once all the jobs have finished running, the final result will be summarized in a CSV file.


## 2. Usage
<a id="markdown-usage" name="usage"></a>

In this section, we will explain how to run each of the stages described in section 1. 

### 2.1. General Configuration
<a id="markdown-general-configuration" name="general-configuration"></a>

First, our configuration setup assumes that the `/opt/bignlp` directory has been copied from the container 
to the local file system. And we assume that bignlp-hp-tool and BigNLP-Inference-Scripts are both 
located inside the `bignlp` directory.

The first parameter that must be set is the `bignlp_hp_tool_path` parameter inside the `conf/config.yaml` 
file. This parameter must point to the absolute path where the `bignlp-hp-tool` repository is stored in 
the file system. Additionally, if using a Slurm based cluster, the config file in the 
`conf/cluster/bcm.yaml` subfolder has the parameters to set the generic cluster related information, 
such as the `partition` or `account` parameters.

The `bignlp_hp_tool_path` parameter will automatically be mounted to the container at the same path as 
in the local file system. Any additional directories that should be mounted must be specified using the
`container_mounts` parameter. If the paths contain the colon character (`:`), the code will assume both 
the source and destination paths are provided. Otherwise, the given paths will be mounted to the same 
path inside the container.

The `bignlp_inference_path` must point to the path where BigNLP-Inference-Scripts is located. The location 
specified in the default config should be valid if `/opt/bignlp` was extracted correctly. Next, the 
`data_dir` value must point to the path where the training dataset is located. Note that the dataset 
for GPT-3, T5 and mT5 values will be different, so modify this parameter accordingly. Follow the data 
preparation steps to learn how to download and preprocess the datasets for each model. The dataset in 
this path does not need to be the full size dataset; only a small representative sample of the dataset 
is needed, since the HP tool does not train the models to convergence. Finally, the `base_results_dir` 
parameter can be modified to point to the location where the results will be stored. See all the 
parameters for the `conf/config.yaml` file below:

```yaml
defaults:
  - _self_
  - cluster: bcm
  - search_config: gpt3/5b
  - override hydra/job_logging: stdout

bignlp_hp_tool_path: ???  # Path should end with bignlp-hp-tool.
bignlp_inference_path: ${bignlp_hp_tool_path}/../BigNLP-Inference-Scripts
data_dir: ${bignlp_hp_tool_path}/../bignlp-scripts/data
base_results_dir: ${bignlp_hp_tool_path}/results

training_container: nvcr.io/ea-bignlp/bignlp-training:22.04-py3
container_mounts:
    - null
```

### 2.2. Running Pre-Defined Configs
<a id="markdown-running-pre-defined-configs" name="running-pre-defined-configs"></a>

The pre-defined configs we provide have been well tested, and the outputs produced by the HP tool 
have been verified manually. Running one of these configs will first generate a base config file for 
the specified model size. Then, it will launch the training and inference grid search jobs. When 
all the jobs have finished, a final recommendation will be produced for both training and inference, 
which will show the optimal hyper-parameters for the given model.

The pre-defined configs can be found in the `conf/search_config` directory. Each YAML file shows one 
model type (GPT-3, T5 or mT5) and one model size (up to 175B parameters for GPT-3 and up to 42B 
parameters for T5/mT5). To run the desired config, we will need to modify the `search_config` 
parameter in the `conf/config.yaml` file. For example, if we wish to run a 5B GPT-3 model, we can 
set this value to `gpt3/5b` (the .yaml ending should not be included). 

The tool will always generate the base configuration for the given model first. Then, the 
`run_training_hp_search` and `run_inference_hp_search` parameters can be set to `True`, 
to run the training and inference HP searches, respectively. If any of these two parameters are set 
to `False`, the corresponding pipeline will not be executed. Once these parameters are set, we can 
run the tool calling `python3 main.py`. 

#### 2.2.1. Model Config
<a id="markdown-model-config" name="model-config"></a>

To run the `gpt3/5b` config, we need to set up the `conf/search_config/gpt3/5b.yaml` file correctly.
The config is split in two sections: `train_settings` and `inference_settings`. 

```yaml
train_settings:
  model_size_in_b: 5 # unit in billion parameters
  num_nodes: 20
  gpus_per_node: 8
  max_training_days: 5 # unit in days
  limit_search_runs: 100 # Max number of runs to be launched in parallel for grid search.
  output_top_n: 10  # The result will print the top N fastest training configs.
  max_minutes_per_run: 40 # minutes per run for the grid search.
  tflops_per_gpu: 140  # Estimated tflops per GPU.
  num_tokens_in_b: 300  # Unit in billions, typically 300B for GPT3 models.
  vocab_size: 51200
  logs: ${base_results_dir}/${search_config_value}  # Example base_results_dir/gpt3/126m
  tensor_parallel_sizes: null  # null to use our recommendation, or a list, such as [1, 2, 4, 8]
  pipeline_parallel_sizes: null  # null to use our recommendation, or a list, such as [1, 2, 4, 8, 10]
  micro_batch_sizes: null  # null to use our recommendation, or a list, such as [1, 2, 4, 8, 16]
  act_ckpt_layers: null  # null to use our recommendation, or a list, such as [0, 1, 2, 3]
 
inference_settings:
  vocab_size: 51200
  start_id: 50256
  end_id: 50256
  input_seq_len: 60
  output_seq_len: 20
  top_n: 10
  logs: ${base_results_dir}/${search_config_value}  # Example base_results_dir/gpt3/126m
  tensor_parallel_sizes: [1, 2, 4, 8]
  pipeline_parallel_sizes: [1, 2, 3, 4]
  max_batch_sizes: [1, 2, 8, 16, 32, 64, 256]
```

#### 2.2.2. Base Config Generation
<a id="markdown-base-config-generation" name="base-config-generation"></a>

Every time we call `python3 main.py`, a base configuration will be generated for the given model, 
and it will be saved to the `logs` directory indicated in your config files. The base configuration 
consists of a YAML file that can be run using the NeMo-Megatron training container. However, this 
base configuration has not yet been optimized to achieve the highest possible throughput, this will 
be achieved in step 2.1.2.


#### 2.2.3. Training HP Search
<a id="markdown-training-hp-search" name="training-hp-search"></a>

To run the training HP search pipeline, the parameter `run_training_hp_search` must be set to `True` 
in the `conf/config.yaml` file. The model used to search the best training HPs must be selected 
using the `search_config` parameter in `conf/config.yaml`. For example, by default, this parameter 
will be set to `gpt3/5b`, so our tool will search the optimal training HPs for a 5B parameter GPT-3 
model. The configuration for this model can be found in the `conf/search_config/gpt3/5b.yaml` file. 
To configure the behavior of the HP search, the following parameters can be modified in the 
correspoinding YAML file: 

```yaml
train_settings:
  model_size_in_b: 5 # unit in billion parameters
  num_nodes: 20
  gpus_per_node: 8
  max_training_days: 5 # unit in days
  limit_search_runs: 100 # Max number of runs to be launched in parallel for grid search.
  output_top_n: 10  # The result will print the top N fastest training configs.
  max_minutes_per_run: 40 # minutes per run for the grid search.
  tflops_per_gpu: 140  # Estimated tflops per GPU.
  num_tokens_in_b: 300  # Unit in billions, typically 300B for GPT3 models.
  vocab_size: 51200
  logs: ${base_results_dir}/${search_config_value}  # Example base_results_dir/gpt3/126m
  tensor_parallel_sizes: null  # null to use our recommendation, or a list, such as [1, 2, 4, 8]
  pipeline_parallel_sizes: null  # null to use our recommendation, or a list, such as [1, 2, 4, 8, 10]
  micro_batch_sizes: null  # null to use our recommendation, or a list, such as [1, 2, 4, 8, 16]
  act_ckpt_layers: null  # null to use our recommendation, or a list, such as [0, 1, 2, 3]
```

The `model_size_in_b` parameter indicates how many billion parameters the model should contain, and 
the tool will provide a config and HPs for a model of that size. The `num_nodes` parameter indicates 
how many nodes will be used to train this model to full convergence, after the HP search is finished. 
Therefore, it will be ignored by the HP search tool, and it will only be used when generating the 
final config YAML files. The `gpus_per_node` parameter indicates how many GPUs are available in each 
node. The `max_training_days` parameter shows how many days this model will be trained for, when 
training to full convergence. It will be written to the final config YAML files. This parameter can 
also be used when `model_size_in_b` is set to `null`, as you will see in section 2.2. The 
`limit_search_runs` parameter can be used to limit the number of configs that will be searched 
during the HP search stage. We recommend selecting a value between 30 and 100 for this parameter. 
The tool will probably need to search at least 30 different configs to find the optimal one. However, 
if the compute is available in your cluster, we recommend increasing this parameter to a value close 
to 100. The `output_top_n` parameter can be used to configure how much detail the output summary file 
will include. By default, when set to 10, it will output the top 10 configurations. The 
`max_minutes_per_run` parameter indicates how long to run each configuration for, in minutes. We 
recommend using at least 20 minutes per run for the smaller models, and increasing it to over 60 
minutes for the larger models. The `tflops_per_gpu` parameter provides an estimate of the TFLOPs 
each GPU can achieve when training LLMs with NeMo-Megatron. This value is only used to provide an 
estimate of how long the model will take to train for full convergence, so you can know the time to 
train before you even begin training your model. The `num_tokens_in_b` parameter indicates how many 
billions of tokens you will train your model for, when training to full convergence. It will be used 
when estimating how long it will take to train the model, to the desired number of tokens. The 
`vocab_size` parameter must show the vocabulary size that will be used during training. The `logs` 
parameter can be used to configure where the result logs will be saved. By default, this directory 
will be created inside the `base_results_dir` indicated in the `conf/config.yaml` file. Finally, 
the `tensor_parallel_sizes`, `pipeline_parallel_sizes`, `micro_batch_sizes`, and `act_ckpt_layers` 
parameters can be used to override the heuristics that choose the grid search space for these 
four parameters. If these are left as `null`, our tool will select appropriate values. However, 
if you wish to override them, you can use these parameters. For example, if you only wish to search 
for configurations with Tensor Parallelism (TP) values of 1 and 2, you can set 
`tensor_parallel_sizes: [1, 2]` and leave the other parameters as `null`.  

#### 2.2.4. Inference HP Search
<a id="markdown-inference-hp-search" name="inference-hp-search"></a>

To run the inferencei HP search pipeline, the parameter `run_inference_hp_search` must be set to `True`
in the `conf/config.yaml` file. The model used to search the best inference HPs must be selected
using the `search_config` parameter in `conf/config.yaml`. For example, by default, this parameter
will be set to `gpt3/5b`, so our tool will search the optimal inference HPs for a 5B parameter GPT-3
model. The configuration for this model can be found in the `conf/search_config/gpt3/5b.yaml` file.
To configure the behavior of the HP search, the following parameters can be modified in the
correspoinding YAML file:

```yaml
inference_settings:
  vocab_size: 51200  # Number of decoder layers
  start_id: 50256  # id of start token in vocabulary
  end_id: 50256  # id of end token in vocabulary
  input_seq_len: 60  # Length of the input sequence
  output_seq_len: 20  # Length of the output sequence
  max_latency_ms: 1200  # Maximum allowed latency
  top_n: 10  # Top N models to be output in the summary.
  logs: ${base_results_dir}/${search_config_value}  # Log directory
  tensor_parallel_sizes: [1, 2, 4, 8]
  pipeline_parallel_sizes: [1]
  max_batch_sizes: [1, 2, 8, 16, 32, 64, 256]
```

The `vocab_size` parameter indicates the vocabulary size. The `start_id` and `end_id` parameters 
indicate the ids of the start and end tokens in the vocabulary, respectively. The `input_seq_len` 
and `output_seq_len` parameters show the length of the input and output sequences for the model, 
respectively. The `max_latency_ms` value can be used to only output models that meet that latency 
requirement, in miliseconds. The `top_n` parameter can be used to modify how many configs will be 
written to the output summary file. The `logs` parameter indicates where the logs will be stored.
Finally, the `tensor_parallel_sizes`, `pipeline_parallel_sizes`, and `max_batch_sizes` must be a 
list of values to generate the desired HP search. In this case, these values cannot be null, they 
must be provided by the user.


### 2.3. Running Custom Model Size Configs
<a id="markdown-running-custom-model-size-configs" name="running-custom-model-size-configs"></a>

The HP Tool is capable of recommending a model size, based on your hardware and training time 
constraints. For instance, if you want to train a GPT-3 model, but don't know what model size is 
appropriate, you can input the number of nodes (and GPUs per node) available in your cluster, 
the amount of time you want to spend training the model, and the tool will recommend a model size 
that can be trained in that time with your hardware. To see an example of this, you can look at 
the `conf/search_config/gpt3/unknown_size.yaml` file. In this file, the `model_size_in_b` 
parameter is set to null. This is how you can tell the tool to recommend a model size to you. 
For the recommendation to work correctly, the `num_nodes`, `gpus_per_node`, and `max_training_days` 
parameters must indicate the number of nodes and GPUs per node available, and how long you wish to 
train the model for. Also, the tool needs to know the vocabulary size, number of tokens you will 
train the model for, and the estimated TFLOPS per GPU your hardware can achieve. These can be 
modified using the `vocab_size`, `num_tokens_in_b`, and `tflops_per_gpu` parameters, respectively. 
Once all these parameters are set correctly, and after selecting the `gpt3/unknown_size` as the 
config to run in the `search_config` parameter in the `conf/config.yaml` file, the training 
pipeline can be executed calling `python3 main.py`. This will produce a base configuration for 
the suggested model size. If `run_training_hp_search` or `run_inference_hp_search` are set to 
`True`, the tool will also search for the HPs for training or inference, using the rest of the 
configuration yaml file as input. The tool will behave the same way as when using a pre-defined 
config.

### 2.4. Interpreting The Results
<a id="markdown-interpreting-the-results" name="interpreting-the-results"></a>

When the tool generates the base configuration for a model, it will be saved inside the directory 
specified in the `logs` parameter in your config files. By default, this will be 
`.../bignlp-hp-tool/results/<model_name>/<model_size/`. As the default `search_config` value is 
set to `gpt3/5b`, the results can be found in the `.../bignlp-hp-tool/results/gpt3/5b/` directory. 
The base config will be available inside that directory, with the name `base_cfg_<model_size>.yaml`. 

If the training HP search pipeline is run, the results will be in three different directories inside 
your `logs` directory. The `candidate_configs` directory contains all the YAML files with all the 
configurations generated by the HP search. The `training_logs` directory contains all the logs of 
training each of the individual configs the tool generated. If `limit_search_runs` was set to 30, 
then there should be 30 different directories with the logs for each model. 

Finally, after all 
the training runs have finished and the final run has analyzed the throughput of each configuration, 
the final model recommendation will be stored in the `final_results` directory. This directory will 
contain a log file which lists the `output_top_n` fastest configs, sorted from fastest to slowest. 
It will also contain the recommendation of which model is the fastest. The directory will also 
contain a YAML file, which corresponds to the config with the lowest training time. This is the 
recommended model for training. 

Notes: 
 - Since the HP search tool uses the minimum number of nodes necessary to save compute and time, 
the result might vary slightly when increasing the node count for these models.
 - If one of the optimal configs is very close to 100% GPU memory utilization, it is possible that 
the full training job will crash due to a memory spike. We recommend using a config that keeps the 
memory usage under 98% to avoid this issue. To save some memory, the recommendation is to try 
increasing the activation checkpointing layers by one each time. The performance will suffer 
slightly, but the memory footprint will be reduced.

### 2.5. Logging Runs with Weights and Biases
<a id="markdown-logging-runs-with-weights-and-biases" name="logging-runs-with-weights-and-biases"></a>

Weights and Biases (W&B) can be used to log all the training search runs. To achieve this, the 
`wandb` parameters must be modified in the `conf/config.yaml` file. First, `enable` must be set to 
`True`. Then, the `api_key_file` must be set to point to the path where the file which contains 
the W&B API key. The API key must be in the first line of that file. Finally, the `project` parameter
must have the name of the W&B project where the metrics will be stored. The name of each run does not 
need to be provided. It will be automatically generated by the tool, using the model name, model size, 
and hyper-parameters used for each specific run.

```yaml
wandb:  # Weights and Biases (W&B) logging.
    enable: True 
    api_key_file: null
    project: bignlp-hp-tool
```




