# BigNLP-Scripts

Scripts and code to provide end-to-end data preparation and training for
Megatron-LM.

## Table of contents

  - [Installation](#installation)
  - [General Configuration](#general-configuration)
  - [Data Preparation](#data-preparation)
  - [GPT-3 Training](#gpt-3-training)
  - [Model Evaluation](#model-evaluation)
  - [Deploying the BigNLP model](#deploying-the-bignlp-model)
    - [Model inference deployment process](#model-inference-deployment-process)
    - [1. Prepare environment](#1-prepare-environment)
    - [2. Provide model and inference configuration](#2-provide-model-and-inference-configuration)
      - [2.1 Predefined configuration for selected models](#21-predefined-configuration-for-selected-models)
      - [2.2. Optimal configuration search](#22-optimal-configuration-search)
        - [2.2.1 Random weights checkpoint benchmark](#221-random-weights-checkpoint-benchmark)
        - [2.2.2. Trained checkpoint benchmark](#222-trained-checkpoint-benchmark)
      - [2.3. Review deployment search results](#23-review-deployment-search-results)
    - [3. Prepare NVIDIA Triton Model Repository and run accuracy / performance tests](#3-prepare-nvidia-triton-model-repository-and-run-accuracy-performance-tests)
    - [4. Run NVIDIA Triton Server with selected Model Repository](#4-run-nvidia-triton-server-with-selected-model-repository)
  - [Performance](#performance)
    - [Benchmarking](#benchmarking)
    - [Results](#results)
      - [Training Accuracy Results](#training-accuracy-results)
      - [Training Performance Results](#training-performance-results)
      - [Inference performance](#inference-performance)
        - [5B model](#5b-model)
          - [5B: Chatbot question for answering](#5b-chatbot-for-question-answering)
          - [5B: Translation and style transfer](#5b-translation-and-style-transfer)
          - [Summary for 5B results](#summary-for-5b-results)
        - [20B model](#20b-model)
          - [20B: Chatbot question for answering](#20b-chatbot-for-question-answering)
          - [20B: Translation and style transfer](#20b-translation-and-style-transfer)
          - [Summary for 20B results](#summary-for-20b-results)
        - [Model size and performance](#model-size-and-performance)
          - [Online scenario](#online-scenario)
          - [Offline scenario](#offline-scenario)



## Installation:
To be able to call the necessary scripts from the login node on a cluster, some
packages must be installed using the requirements.txt file:
        cd bignlp-scripts
        pip3 install --user -r requirements.txt


## General Configuration
The first parameter that must be set is the bignlp_path parameter inside the
conf/config.yaml file, which must point to the absolute path where the
bignlp-scripts repository is stored in the file system.

Every other path or directory in all the config files can either be an absolute
or a relative path. Every path starting with the “/” symbol will be considered
an absolute path, and everything else will be treated as a relative path, and
the path indicated in the bignlp_path parameter of the conf/config.yaml file
will be appended to the beginning of each relative path.

The end to end pipeline can be executed running: python3 main.py
This will read the entire configuration and execute the desired pipelines.


## Data Preparation
We provide utilities to download and prepare the publicly available The Pile
dataset, which is formed by 22 smaller datasets.The dataset is already blended
by using the mix described in their paper. It is recommended to store this
repository and the datasets in a file system shared by all the nodes (gpfs).

The configuration used for data preparation must be specified in the 
bignlp-scripts/conf/config.yaml file, and run_data_preparation must be set to
True to run it.  The data_preparation parameter specifies which file to use for
data preparation configuration purposes. The default value is set to
download_pile, which can be found in 
bignlp-scripts/config/data_preparation/download_pile.yaml. The parameters can
be modified to perform the different tasks and to decide where to store the
datasets, vocab and merges files.


## GPT-3 Training
We provide an easy-to-use yet powerful pipeline to perform distributed training
of GPT-3 models across multiple nodes and GPUs. We also provide
well-established recipes for different sizes of GPT-3 models, where the
throughput has been maximized and the converge properties of the model have
been tested and confirmed.

The configuration used for the training pipeline must be specified in the
bignlp-scripts/conf/config.yaml file, specifying the training parameter, which
specifies which file to use for training purposes. The run_training parameter must be set to True to run the training pipeline. The default value is set to 5b, which can be found in bignlp-scripts/config/training/5b.yaml. The parameters can be modified to adjust the hyperparameters of the training runs.


## Model Evaluation
We also provide a simple tool to help evaluate the trained checkpoints. You can
 evaluate the capabilities of the GPT-3 model on the following downstream
evaluation tasks: lambada, boolq, race, piqa, hellaswag, winogrande, wikitext2,
and wikitext103.

The configuration used for the evaluation needs to be specified in the
bignlp-scripts/conf/config.yaml file, specifying the evaluation parameter,
which specifies which file to use for evaluation purposes. The run_evaluation
parameter must be set to True to run the evaluation pipeline. The default value
is set to evaluate_lambada, which can be found in
bignlp-scripts/config/evaluation/evaluate_lambada.yaml. The parameters can be
modified to adapt different evaluation tasks and checkpoints in evaluation runs.


## Deploying the BigNLP model


This section describes the deployment of the BigNLP model on the NVIDIA Triton
Inference Server with FasterTransformer Backend on both single and multiple
node environments.  NVIDIA Triton Inference Server supports many inference
scenarios, of which two most important are:
* Offline inference  scenario - with a goal to maximize throughput regardless
  of the latency, usually achieved with increasing batch size and using server
  static batching feature.
* Online inference scenario - with a goal to maximize throughput within a given
  latency budget, usually achieved with small batch sizes and increasing
  concurrency requests to the server, using dynamic batching feature.

[NVIDIA Triton Model Navigator](https://github.com/triton-inference-server/model_navigator)
helps with conversion and setting up a deployment environment to do inference
for models from BigNLP training scripts. Use scripts to convert models to a new
format, then use NVIDIA Triton Inference Server to process inference requests.

The inference scripts execute at a Slurm cluster in several steps:
* Megatron/NeMo checkpoint conversion to FasterTransformer format.
* Preparation of model repository for NVIDIA Triton Inference Server.
* Profiling and selecting the best inference model and NVIDIA
  Triton Inference Server configuration.
* Accuracy verification.
* Profiling of deployed models.

The inference container is pulled from a Docker registry. You must ensure that
your Slurm configuration allows access to your registry. NVIDIA provides the
container with all components necessary for inference at the
[NGC Docker registry](https://ngc.nvidia.com/catalog/containers).
Inference scripts use the [pyxis slurm plug-in](https://github.com/NVIDIA/pyxis)
to pull and run the container in a node.


The navigator script converts a checkpoint from a training format to the
[FasterTransformer](https://github.com/triton-inference-server/fastertransformer_backend)
format. The NVIDIA Triton Model Navigator looks for a trained
checkpoint in the workspace passed as an argument and creates a navigator
workspace with all output files, which can be used for production inference
deployment.

The NVIDIA Triton Model Navigator script generates many NVIDIA Triton model
repositories and manages them to conduct optimization of configuration
parameters. This optimizes GPU memory and makes inference a lot faster. NVIDIA
Triton Inference Server’s optimization tool Model Analyzer helps to find the
best configuration, taking into account constraints defined in the navigator’s
configuration. It is possible to set constraints for latency, number of GPUs
and [NVIDIA DGX A100](https://www.nvidia.com/en-us/data-center/dgx-a100/)
machines. All generated models are profiled to report latency and throughput.
Once the model is optimized, you can deploy it to your inference infrastructure
and use it in production.


### Model inference deployment process

![Model inference deployment process diagram](img/inference_deployment_flow.png)


### 1. Prepare environment

The whole solution uses a set of Docker containers executed at Slurm cluster
using the pyxis plug-in. The training container also includes conversion
scripts and NVIDIA Triton Model Navigator. The inference container is just the
NVIDIA Triton Inference Server with the FasterTransformer backend installed.
Install the BigNLP scripts dependencies on the head node of your cluster:

```
pip install -r requirements.txt
```

You can use `virtualenv` to prevent polluting your
head node environment for other Python projects. If your Slurm configuration
lacks pip, then you can use [get\_pip.py](https://github.com/pypa/get-pip)
with just `python3`.

You must set your configuration for a slurm cluster in YAML file:

```yaml
slurm:                  # example config for enterprise cluster
  sbatch_parameters:    # this overwrites sbatch parameters generated by submitit
    account: null       # slurm account
    partition: "batch"  # slurm partition
    exclude: null       # slurm nodes, which should be excluded from jobs
  srun_args: ["--mpi", "pmix"] # additional slurm arguments list
  enable_gpus_allocation: true
  job_name_prefix: "bignlp-"
env:
  pyxis_container_workdir: /bignlp_workdir
  pyxis_training_container_image: nvcr.io/ea-bignlp/bignlp-training:21.10-py3-base
  pyxis_inference_container_image: nvcr.io/ea-bignlp/bignlp-inference:21.10-py3-base
```

The `sbatch_parameters` section configures Slurm job parameters. The `srun_args`
should contain [MPI](https://slurm.schedmd.com/mpi_guide.html) configuration
valid for your cluster.

The `env` section sets pyxis development environment:
 * `pyxis_container_workdir`: Work directory used in Docker container.
 * `pyxis_training_container_image`: NGC training container for BigNLP.
 * `pyxis_inference_container_image`: NGC inference container for BigNLP.

### 2. Provide model and inference configuration

#### 2.1 Predefined configuration for selected models

The repository contains the conf/inference folder with predefined NVIDIA Triton
Model Navigator configurations saved in YAML files. Those configurations are
prepared for 5B, 20B, 175B and 530B GPT3 models for two input/output
configurations 200/200 and 60/20. The configurations cover inference with
several GPUs at one node.  The files are present in the 
`conf/inference/optimal_configurations` folder.

The configuration changes for different input sequence lengths and output
sequence lengths used in inference tasks. An application like chatbot can work
with an input of 60 tokens and an output of 20 tokens. Scenarios like text
translation require much longer lengths closer to 200 for input tokens and 200
for output tokens. The RAM usage for a bigger batch size with longer sequence
lengths increases significantly, so optimal configurations set different
maximum batch size values for different sequence lengths. The predefined
configuration files can be used with the `prepare_model_repository.py` script
described later. The files are marked with a number of parameters in model
architecture like 5B, which means 5 billion parameters.

Input sequence lengths 60 and output 20:
* **5B GPT3**: `5b_io_60_20.yaml`
* **20B GPT3**: `20b_io_60_20.yaml`
* **175B GPT3**: `175b_io_60_20.yaml`
* **530B GPT3**: `530b_io_60_20.yaml`

Input sequence lengths 200 and output 200:
* **5B GPT3**: `5b_io_200_200.yaml`
* **20B GPT3**: `20b_io_200_200.yaml`
* **175B GPT3**: `175b_io_200_200.yaml`
* **530B GPT3**: `530b_io_200_200.yaml`

The configuration folder also contains configuration for random
FasterTransformer checkpoints. It is possible to start FasterTransformer
inference without weight files because the engine just initializes them to
random values. This model can’t deliver any valid accuracy, but it is possible
to benchmark inference constraints like latency before the expensive training
of a large model is finished. The folder `conf/inference/model_specs` contains a
folder with predefined random model configuration, which cover range of example
GPT3 configurations, where each folder is marked with a number of model
parameters:
* **5B**: `5b.ft`
* **20B**: `20b.ft`
* **89B**: `89b.ft`
* **175B**: `175b.ft`
* **310B**: `310b.ft`
* **530B**: `530b.ft`


#### 2.2. Optimal configuration search

##### 2.2.1 Random weights checkpoint benchmark

NVIDIA Triton Model Navigator can benchmark inference before training is
finished and verify inference constraints ahead of time; for example maximum
latency budget or number of GPUs, thus cost of inference. For performance
reasons, if you already know model size and parameters, you can use the
FasterTransformer NVIDIA Triton backend to generate a checkpoint with random
weights inside the NVIDIA Triton Inference Server.

The first step in the benchmark script generates a random checkpoint based on
your configuration. The second step configures model repositories. The third
step starts a set of NVIDIA Triton Inference Servers and executes the
performance measurements for each.

The inputs:
* Random model configuration - For example, `conf/inference/model_specs/5b.ft`
* Docker image with training and profiling scripts.
* Docker image with NVIDIA Triton and FasterTransformer backend.
* Performance profile configuration YAML file.

The outputs:
* Performance report.
* Performance results.
* Optimal configurations.
* NVIDIA Triton model stores with a placeholder for the trained model checkpoint.

You can benchmark a model using
`infer_scripts/profile_model_with_random_weights.py` script:

```
python3 ./infer_scripts/profile_model_with_random_weights.py \
    --cluster-config-path <Your cluster config>.yaml \
    --navigator-config-path ./conf/inference/profile_offline.yaml \
    --model-path conf/inference/model_specs/5b.ft \
    --model-name ft_5B \
    --tensor-parallel-sizes 1 \
    --pipeline-parallel-sizes 1 \
    --input-output-lengths 60,20 \
    --max-batch-sizes 1 \
    --max-latency-ms 100000 \
    -v
```

The parameters:
* `cluster-config-path`: Cluster configuration YAML file.
* `navigator-config-path`: Navigator configuration YAML;
   for example,`./conf/inference/profile_offline.yaml`
* `model-path`: This model path contains a YAML file with
   random checkpoint configuration.
* `model-name`: Your model name for NVIDIA Triton repository.
* `tensor-parallel-sizes`: Tensor parallel factor; for example, `1 2 4 8`
* `pipeline-parallel-sizes`: Pipeline parallel factor; for example, `1 2 3 4`
* `input-output-lengths`: Analyzed input and output lengths in format of 
   `<input_len>,<output_len>[ <input_len>,<output_len> …]`;
   for example, `60,20 200,200`
* `max-batch-sizes`: Maximum batch sizes used for optimization;
   for example, `1 2 4 8 16 256`
* `max-latency-ms`: Maximum p99 latency valid for your scenario.
* `top-n-configs`: Number of optimal configurations to save.

The parameters `tensor-parallel-sizes`, `pipeline-parallel-sizes`,
`input-output-lengths`, and `max-batch-sizes` are used to generate combinations of
possible configurations for FasterTransformer and performance measurement
scripts. The profile script compares throughput normalized to 1 GPU of all
generated configurations and prints N-best configurations taking into account a
maximum latency constraint. If you request very small maximum latency, then the
script won’t be able to find any valid configurations.

The repository contains two profile configurations for Model Navigator:
* `conf/inference/profile_offline.yaml` - Configuration for offline scenario
   focusing on changing batch sizes but not user request concurrency.
* `conf/inference/profile_online.yaml` - Configuration for online scenario
   focusing on changing user request concurrency.


The random model configuration for the model-path parameter is in YAML file:
```yaml
decoder_layers: 105  # Number of decoder layers
head_num: 128        # Number of heads in layer
size_per_head: 160   # Size per head
inter_size: 81920    # It can be: inter_size = size_per_head * head_num * 4
tensor_para_size: 8  # Default tensor parallel configuration (ignored)
vocab_size: 51200    # Vocabulary size based on vocabulary file
start_id: 50256      # id of start token in vocabulary
end_id: 50256        # id of end token in vocabulary
```

The output files are saved in the `current_folder/infer_workspace_<YYYYmmdd_HHMMSS>`.
The N best configurations are printed to the terminal.
The `infer_workspace_<YYYYmmdd_HHMMSS>` folder contains CSV file with all
measurements combined:

```
navigator_workspace/analyzer/results/metrics-model-inference.csv
```

The best configuration is selected based on the throughput normalized for one
GPU. It is possible to deploy the same model at a number of GPUs, so the cost
of model deployment is not constant for all configurations. The script
normalizes this cost by dividing throughput of a model instance by the number
of GPUs used for computation.

##### 2.2.2. Trained checkpoint benchmark

Alternatively, to generate checkpoints randomly, you can use a trained
checkpoint to look for optimal configuration; however, for larger models that
might take a significant amount of time and might not be feasible.

The inputs:
* Megatron/NeMo trained checkpoint.
* Docker image with training and profiling scripts.
* Docker image with NVIDIA Triton and FasterTransformer backend.
* Performance profile configuration YAML file.

The outputs:
* Performance report.
* Performance results.
* Optimal configurations.
* NVIDIA Triton model stores with trained FasterTransformer model checkpoint.

Model repository preparation for the NVIDIA Triton Inference Server:

```
python3 ./infer_scripts/profile_model.py \
    --cluster-config-path <Your cluster config>.yaml \
    --navigator-config-path ./conf/inference/profile_offline.yaml \
    --model-path <Your path to training checkpoint> \
    --model-name model_name -v \
    --tensor-parallel-sizes 1 \
    --pipeline-parallel-sizes 1 \
    --input-output-lengths 60,20 \
    --max-batch-sizes 1 \
    --max-latency-ms 100000 \
    -v
```

The parameters:
* `cluster-config-path`: Cluster configuration YAML file.
* `navigator-config-path`: Navigator configuration YAML;
   for example,`./conf/inference/profile_offline.yaml`
* `model-path`: This model path contains a trained Megatron/NeMo checkpoint.
   A NeMo checkpoint must be passed as a file with .nemo extension,
   but a Megatron checkpoint must be passed as a folder.
* `model-name`: Your model name for NVIDIA Triton repository.
* `tensor-parallel-sizes`: Tensor parallel factor; for example, `1 2 4 8`
* `pipeline-parallel-sizes`: Pipeline parallel factor; for example, `1 2 3 4`
* `input-output-lengths`: Analyzed input and output lengths in format of 
   `<input_len>,<output_len>[ <input_len>,<output_len> …]`;
   for example, `60,20 200,200`
* `max-batch-sizes`: Maximum batch sizes used for optimization;
   for example, `1 2 4 8 16 256`
* `max-latency-ms`: Maximum p99 latency valid for your scenario.
* `top-n-configs`: Number of optimal configurations to save.

The parameters `tensor-parallel-sizes`, `pipeline-parallel-sizes`,
`input-output-lengths`, and `max-batch-sizes` are used to generate combinations of
possible configurations for FasterTransformer and performance measurement
scripts. The profile script compares throughput normalized to 1 GPU of all
generated configurations and prints N-best configurations taking into account a
maximum latency constraint. If you request very small maximum latency, then the
script won’t be able to find any valid configurations.

#### 2.3. Review deployment search results

The `profile_model_with_random_weights.py` and
`profile_model.py` scripts create a folder
`infer_workspace_<YYYYmmdd_HHMMSS>` with a timestamp at the end.

It contains the following folders:
* `model_name-ft_gpu_counts_8-converted.ft`: Folders with converted
   FasterTransformer checkpoints.
* `logs`: Logs.
* `model_repo_model_name-io_60_20-half_1-pp_1-tp_8-mbs_256`:
   NVIDIA Triton model repository for input sequence length 60
   and output length 20 for pipeline parallel 2 and tensor parallel 8
   and maximum batch size 256.
*  `model_repo_model_name-io_60_20-half_1-pp_1-tp_8-mbs_256`:
   NVIDIA Triton model repository for input sequence length 60
   and output length 20 for pipeline parallel 1 and tensor parallel 8 and
   maximum batch size 256.
* `navigator_workspace`: Folder to NVIDIA Triton Model Navigator configurations.
* `slurm_workspace`: Folder with Slurm logs and sbatch scripts.

Both profile scripts print a list of the best models with the name
of the NVIDIA Triton model repository with the best results and performance metrics.

Results from `profile_model.py` and `profile_model_with_random_weights.py`
scripts are saved for review under:
`./infer_workspace-<YYYYmmdd_HHMMSS>/navigator_workspace/analyzer/results/metrics-model-inference.csv`

The CSV file contains several columns:
* `Model` - NVIDIA Triton model name.
* `Batch` - Batch size.
* `Concurrency` - User request concurrency.
* `Model Config Path` - Path to model configuration.
* `Backend Parameters` - Measurement and backend parameters (PP - pipeline
  parallel, TP - tensor parallel, and half - FP16 used for some computations),
  `max_input` - maximum sequence input length, `max_sec` - maximum sequence input
  length plus maximum sequence output length.
* `Preferred Batch Sizes` - List of preferred batch sizes used in NVIDIA Triton configuration.
* `Satisfies Constraints` - “Yes” if a model satisfies the p99 latency constraint, set as the max-latency-ms parameter.
* `Throughput (inder/sec)` - Throughput not normalized for number of GPUs but just measured for one model instance.
* `p95 Latency(ms)`.
* `p99 Latency(ms)`.

Best configurations are mentioned from the top,
To review configurations, check the directory with all generated configs:
`infer_workspace-<YYYYmmdd_HHMMSS>/navigator_workspace/top_configs`

NVIDIA Triton model repositories contain symbolic links to folders with weights.
You should copy final folder with model to expand links into files.

```
  cp -rL <NVIDIA Triton store from script> <destination>
```


### 3. Prepare NVIDIA Triton Model Repository and run accuracy / performance tests
Having the best config and trained checkpoint. A trained model checkpoint is
required as this is final model deployment and verification. For large models,
loading a checkpoint from storage can take a significant amount of time.  

The inputs:
* Trained model checkpoint.
* Docker image with NVIDIA Triton and FasterTransformer backend.
* Lambada dataset.
* Model vocabulary.
* Model merges file.

The English data for accuracy experiments can be downloaded from open resources.

The Lambada dataset can be downloaded from GITHUB:

```
wget https://raw.githubusercontent.com/cybertronai/bflm/master/lambada_test.jsonl
```

The vocabulary and merge files can be downloaded from the Huggingface project:

```
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
```

It’s recommended that you put all files in one folder used for accuracy
verification of your model.

The outputs:
* NVIDIA Triton Model Repository with a converted model in FasterTransformer format.
* Accuracy measurement report.
* Performance measurement report.

The accuracy report is stored in the current directory in the file `lambada_metrics.csv`.
You can verify your model running in NVIDIA Triton by using the Lambada dataset:

```
python3 ./infer_scripts/prepare_model_repository.py \
    --cluster-config-path <Your cluster config>.yaml \
    --navigator-config-path ./conf/inference/small_mbs_256-pp_1-tp_1-io_60_20.yaml \
    --model-path <Your path to training checkpoint> \
    --model-name model_name -v \
    --dataset-dir <Your lambada folder> \
    --model-repository-path <Your output path for NVIDIA Triton model repository> \
    --accuracy-tests \
    --performance-tests
```

Parameters:
* `cluster-config-path`: Cluster configuration YAML file.
* `navigator-config-path`: Navigator configuration to set up NVIDIA Triton.
* `model-path`: This model path contains a trained Megatron/NeMo checkpoint.
   A NeMo checkpoint must be passed as a file with .nemo extension,
   but a Megatron checkpoint must be passed as a folder.
* `model-name`: Model name.
* `dataset-dir`: Folder with downloaded lambada dataset, merges and vocabulary files.
* `model-repository-path`: Path to result NVIDIA Triton Model Repository.
* `accuracy-tests`: Run accuracy tests.
* `performance-tests`: Run performance offline and online tests.

The parameter `navigator-config-path` contains the Navigator configuration to
convert a model, set up a NVIDIA Triton, and parameters to perform performance
tests. You must set some basic parameters to have a working model to verify
accuracy. You can use a predefined configuration for this task, which sets
basic values for a tiny model:

```
./conf/inference/small_mbs_256-pp_1-tp_1-io_60_20.yaml
```

You must check your model size and look for optimal configuration to run
accuracy for your model. The larger models must be run with many GPUs and nodes
to work. The predefined configurations for some GPT3 architectures and
inference tasks are described in the _Predefined configurations_ section above.

### 4. Run NVIDIA Triton Server with selected Model Repository

The inputs:
* NVIDIA Triton model repository with FasterTransformer checkpoint
   ready for inference at production.
* Docker image with NVIDIA Triton and FasterTransformer backend.

The outputs:
* Running NVIDIA Triton model instance serving model in Slurm cluster.

To run the NVIDIA Triton Model Navigator, do the following:
```
python3 ./infer_scripts/run_tritonserver.py \
    --cluster-config-path <Your cluster config>.yaml \
    --model-repository-path <Your output path for NVIDIA Triton model repository> \
    -v
```

The parameters:
* `cluster-config-path`: Cluster configuration YAML file.
* `model-repository-path`: NVIDIA Triton model repository path from folder
   generated by `prepare_model_repository.py` script.

The NVIDIA Triton model repository created in scripts above contains symbolic
links. You need to expand links for `run_tritonserver.py` to
be able to access files, when they are mounted in slurm containers.

The script saves NVIDIA Triton logs so you can verify what happens when
FasterTransformer loads a checkpoint.

## Performance
### Benchmarking

### Results
#### Training Accuracy Results
Training accuracy: NVIDIA SuperPOD (20 x 8 x A100 80GB)
Try to mimic results reporting style from Deep Learning Examples

Evaluation of the different models on all the available tasks:


#### Training Performance Results
Training performance: NVIDIA SuperPOD (20 x 8 x A100 80GB)
Try to mimic results reporting style from Deep Learning Examples

#### Inference performance

The most important factor for NLP model performance is the size of a model. You
can use a smaller model to get faster inference but it will likely degrade your
accuracy. 

If you know your model size, then there are two parameters you can vary to find
the best throughput and keep inside a latency budget:
* Number of GPUs used for one instance of your model.
* Batch size used during processing requests.

The same model can be executed with different amounts of GPUs and nodes so the
basic throughput values don't reflect cost of inference like for one GPU model.
A throughput normalized to one GPU is used as a proxy for cost of inference in
graphs and tables below.

##### 5B model

The 5B model can fit into a single A100 80GB GPU. Still FasterTranformer can
run 5B model using tensor parallel splitting of model between multiple GPUs and
pipeline parallel, when different transformer layers are distributed across
many nodes it gives the possibility to utilize different tradeoffs (e.g.
latency vs throughput). You can also consider using several DGX nodes in
SuperPOD as one instance of the FasterTransformer model. You should also
consider an inference task for your application. Some inference tasks require
longer token sequence lengths  for input and for output.

##### 5B Chatbot for question answering

Let’s consider a scenario with a chatbot for question answering. It can be
implemented with FasterTransformer, when sequence length for input tokens is 60
and output length is 20. Two graphs below show how latency and throughput vary,
when a certain number of GPUs is used for inference for batch size=1 and for
batch size=256.


![5B GPT-3 | batch\_size: 1 | input\_len:60 | output\_len:20](img/5B_GPT_3_batch_size_1_input_len_60_output_len_20.svg)

![5B GPT-3 | batch\_size: 256 | input\_len:60 | output\_len:20](img/5B_GPT_3_batch_size_256_input_len_60_output_len_20.svg)


If latency achievable at 1-GPU configuration fits within latency budget, then
the best performance can be derived from the graph below, which shows how
latency and throughput change for different batch sizes used for computations.


![5B GPT-3 | # of GPU: 1 | input\_len:60 | output\_len:20](img/5B_GPT_3_of_GPU_1_input_len_60_output_len_20.svg)

A chatbot with a latency budget within 380 ms can work for batch size=64 and 1
GPU used for computation.


##### 5B: Translation and style transfer

A translation or style transfer inference task requires input length 200 and
output length 200.

![5B GPT-3 | batch\_size: 1 | input\_len:200 | output\_len:200](img/5B_GPT_3_batch_size_1_input_len_200_output_len_200.svg)

![5B GPT-3 | batch\_size: 256 | input\_len:200 | output\_len:200](img/5B_GPT_3_batch_size_256_input_len_200_output_len_200.svg)

The graph for 1 GPU with many batch sizes shows what batch size can fit into a
certain latency budget.


![5B GPT-3 | # of GPU: 1 | input\_len:200 | output\_len:200](img/5B_GPT_3_of_GPU_1_input_len_200_output_len_200.svg)

The graph clearly shows that the translation or style transfer inference task
with latency budget 2000 milliseconds can be deployed using 1 GPU and batch
size = 16.

##### Summary for 5B results

The table below contains performance measurements from all graphs for the 5B
model running in FasterTransformer at NVIDIA DGX A100 80GB.


<details>

<summary>
5B model: Latency and throughput for different number of GPUs and batch sizes.
</summary>

| GPUs | Latency p99                | Normalized throughput to 1 GPU | Latency p99 | Normalized throughput to 1 GPU | Latency p99                  | Normalized throughput to 1 GPU | Latency p99 | Normalized throughput to 1 GPU |
| ---- | -------------------------- | ------------------------------ | ----------- | ------------------------------ | ---------------------------- | ------------------------------ | ----------- | ------------------------------ |
|      | Input len 60 output len 60 |                                |             |                                | Input len 200 output len 200 |                                |             |                                |
|      | BS=256                     |                                | BS=1        |                                | BS=256                       |                                | BS=1        |                                |
| 1    | 1143                       | 224                            | 172         | 5.81                           | 9048                         | 28.3                           | 1623        | 0.616                          |
| 2    | 799                        | 160                            | 126         | 3.95                           | 6018                         | 21.3                           | 1219        | 0.410                          |
| 4    | 529                        | 121                            | 94          | 2.66                           | 3939                         | 16.2                           | 923         | 0.271                          |
| 8    | 436                        | 73                             | 115         | 1.08                           | 3154                         | 10.1                           | 998         | 0.125                          |
| 16   | 327                        | 49                             | 101         | 0.62                           | 2776                         | 5.8                            | 977         | 0.064                          |
| 24   | 273                        | 39                             | 100         | 0.42                           | 2484                         | 4.3                            | 950         | 0.044                          |
| 32   | 284                        | 28                             | 95          | 0.33                           | 2517                         | 3.2                            | 897         | 0.035                          |

</details>

##### 20B model

To improve accuracy a larger model can be used.

##### 20B: Chatbot for question answering

![20B GPT-3 | batch\_size: 1 | input\_len:60 | output\_len:20](img/20B_GPT_3_batch_size_1_input_len_60_output_len_20.svg)

![20B GPT-3 | batch\_size: 256 | input\_len:60 | output\_len:20](img/20B_GPT_3_batch_size_256_input_len_60_output_len_20.svg)

![20B GPT-3 | # of GPU: 1 | input\_len:60 | output\_len:20](img/20B_GPT_3_of_GPU_1_input_len_60_output_len_20.svg)



##### 20B: Translation and style transfer


![20B GPT-3 | batch\_size: 1 | input\_len:200 | output\_len:200](img/20B_GPT_3_batch_size_1_input_len_200_output_len_200.svg)

![20B GPT-3 | batch\_size: 256 | input\_len:200 | output\_len:200](img/20B_GPT_3_batch_size_256_input_len_200_output_len_200.svg)

![20B GPT-3 | # of GPU: 1 | input\_len:200 | output\_len:200](img/20B_GPT_3_of_GPU_4_input_len_200_output_len_200.svg)


##### Summary for 20B results

The table below contains performance measurements from all graphs for the 20B
model running in FasterTransformer at NVIDIA DGX A100 80GB.

<details>

<summary>
20B model: Latency and throughput for different number of GPUs and batch sizes.
</summary>

| GPUs | Latency p99                | Normalized throughput to 1 GPU | Latency p99 | Normalized throughput to 1 GPU | Latency p99                  | Normalized throughput to 1 GPU | Latency p99 | Normalized throughput to 1 GPU |
| ---- | -------------------------- | ------------------------------ | ----------- | ------------------------------ | ---------------------------- | ------------------------------ | ----------- | ------------------------------ |
|      | Input len 60 output len 60 |                                |             |                                | Input len 200 output len 200 |                                |             |                                |
|      | BS=256                     |                                | BS=1        |                                | BS=64,128,256                |                                | BS=1        |                                |
| 1    | 4146                       | 62                             | 560         | 1.78                           | 10772                        | 5.9                            | 5650        | 0.177                          |
| 2    | 2429                       | 53                             | 359         | 1.39                           | 10544                        | 6.1                            | 3548        | 0.141                          |
| 4    | 1592                       | 40                             | 251         | 1.00                           | 10453                        | 6.1                            | 2486        | 0.101                          |
| 8    | 1169                       | 27                             | 230         | 0.54                           | 7909                         | 4.0                            | 2147        | 0.058                          |
| 16   | 923                        | 17                             | 218         | 0.29                           | 7380                         | 2.2                            | 2131        | 0.029                          |
| 24   | 758                        | 14                             | 218         | 0.19                           | 6511                         | 1.6                            | 2123        | 0.020                          |
| 32   | 742                        | 11                             | 224         | 0.14                           | 6200                         | 1.3                            | 2124        | 0.015                          |

</details>

##### Model size and performance
###### Online scenario

An online scenario focuses on the minimization of latency. Large checkpoints
were generated with randomly initialized weights.


![Chatbot Q&A | batch\_size: 1 | input\_len:60 | output\_len:20](img/Chatbot_Q_A_batch_size_1_input_len_60_output_len_20.svg)


![Translation or style transfer | batch\_size: 1 | input\_len:200 | output\_len:200](img/Translation_or_style_transfer_batch_size_1_input_len_200_output_len_200.svg)

The performance measurements were obtained at NVIDIA DGX A100 80GB nodes.



<details>

<summary>
Performance for different model sizes in online scenario
</summary>

|                         | Len input 60 output 20 |                   |            |                             |                                |                | Len input 200 output 200 |                   |            |                             |                                |                |
| ----------------------- | ---------------------- | ----------------- | ---------- | --------------------------- | ------------------------------ | -------------- | ------------------------ | ----------------- | ---------- | --------------------------- | ------------------------------ | -------------- |
| Parameters number \[B\] | Latency\[ms\]          | Infer/sec per GPU | Batch size | Tensor parallel (GPUs used) | Pipeline parallel (nodes used) | Number of GPUs | Latency\[ms\]            | Infer/sec per GPU | Batch size | Tensor parallel (GPUs used) | Pipeline parallel (nodes used) | Number of GPUs |
| 5B                      | 93                     | 2.68              | 1          | 4                           | 1                              | 4              | 923                      | 0.271             | 1          | 4                           | 1                              | 4              |
| 13B                     | 189                    | 1.32              | 1          | 4                           | 1                              | 4              | 1893                     | 0.132             | 1          | 4                           | 1                              | 4              |
| 20B                     | 251                    | 0.50              | 1          | 8                           | 1                              | 8              | 2230                     | 0.056             | 1          | 8                           | 1                              | 8              |
| 89B                     | 464                    | 0.27              | 1          | 8                           | 1                              | 8              | 4585                     | 0.027             | 1          | 8                           | 1                              | 8              |
| 175B                    | 923                    | 0.14              | 1          | 8                           | 1                              | 8              | 8873                     | 0.014             | 1          | 8                           | 1                              | 8              |
| 310B                    | 1354                   | 0.09              | 1          | 8                           | 1                              | 8              | 13391                    | 0.005             | 1          | 8                           | 2                              | 16             |
| 530B                    | 2118                   | 0.03              | 1          | 8                           | 2                              | 16             | 20936                    | 0.003             | 1          | 8                           | 2                              | 16             |

</details>

###### Offline scenario

The offline scenario focuses on maximum throughput. The two graphs below show
latency and throughput for two tasks. The first one is chatbot questions
answering and a second one is translation or style transfer.


![Chatbot Q&A | batch\_size: 256 | input\_len:60 | output\_len:20](img/Chatbot_Q_A_batch_size_256_input_len_60_output_len_20.svg)


![Translation or style transfer | batch\_size: max | input\_len:200 | output\_len:200](img/Translation_or_Style_Transfer_batch_size_max_input_len_200_output_len_200.svg)

The chatbot scenario can be executed with batch size equal to 256 for all model
sizes so it is possible to utilize computing resources in GPUs.


<details>

<summary>
Performance for different model sizes in offline scenario
</summary>

|                         | Len input 60 output 20 |                   |            |                 |                                |                | Len input 200 output 200 |                   |            |                             |                                |                |
| ----------------------- | ---------------------- | ----------------- | ---------- | --------------- | ------------------------------ | -------------- | ------------------------ | ----------------- | ---------- | --------------------------- | ------------------------------ | -------------- |
| Parameters number \[B\] | Latency\[ms\]          | Infer/sec per GPU | Batch size | Tensor parallel | Pipeline parallel (nodes used) | Number of GPUs | Latency\[ms\]            | Infer/sec per GPU | Batch size | Tensor parallel (GPUs used) | Pipeline parallel (nodes used) | Number of GPUs |
| 5B                      | 1143                   | 224.0             | 256        | 1               | 1                              | 1              | 9047                     | 28.297            | 256        | 1                           | 1                              | 1              |
| 13B                     | 2756                   | 92.9              | 256        | 1               | 1                              | 1              | 13390                    | 9.559             | 256        | 2                           | 1                              | 2              |
| 20B                     | 4145                   | 61.8              | 256        | 1               | 1                              | 1              | 10453                    | 6.123             | 256        | 4                           | 1                              | 4              |
| 89B                     | 2889                   | 22.2              | 256        | 4               | 1                              | 4              | 17815                    | 1.796             | 256        | 8                           | 1                              | 8              |
| 175B                    | 2033                   | 15.7              | 256        | 8               | 1                              | 8              | 16181                    | 0.494             | 64         | 8                           | 1                              | 8              |
| 310B                    | 6768                   | 2.4               | 256        | 8               | 2                              | 16             | 13686                    | 0.018             | 2          | 8                           | 1                              | 8              |
| 530B                    | 8660                   | 1.8               | 256        | 8               | 2                              | 16             | 20936                    | 0.003             | 1          | 8                           | 2                              | 16             |

</details>

