# BigNLP-Scripts

Scripts and code to provide end-to-end data preparation and training for
Megatron-LM.


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
env:
  pyxis_container_workdir: /bignlp_workdir
  pyxis_training_container_image: nvcr.io/nvidian/swdl/bignlp:21.10.04-py3-base
  pyxis_inference_container_image: nvcr.io/nvidian/swdl/bignlp:infer-main-py3-base
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


### Convert and optimize model for inference

Triton Inference Server loads the model from the model repository. It contains
a configuration and binary files with model weights. You must convert training
checkpoint into inference Triton model repository for FasterTransformer.

The `profile_model.py` script can generate Triton model repositories and
find the most optimal. You can configure script to look for online configuration
with latency constrain.

The python script uses _submitit_ Python library to schedule and control slurm jobs.
It can setup one job for conversion using training Docker container and another
jobs for inference verification using inference Docker container.
The FasterTransformer backend can run model and multiple GPUs and multiple nodes.
A format of files changes for each configuration so the converted files can
work only with predefined set of GPUs and machines setup with MPI communication.
The `profile_model.py` can convert the same training checkpoint to many hardware
configurations and setup slurm cluster to run a job matching necessary configuration.

The number of GPUs used for model instance determines vale for tensor parallel
(TP) model processing. The FasterTransformer uses tensor parallel processing
in one node between GPUs here and pipeline parallel (PP) processing across the nodes.

The same checkpoint can be used for many input sequence lengths and
many input lengths. The chatbot requires much shorter sentences than translation
so you must decide what sequences match your scenario. The model is the same for all sequence lengths
but performance requirements for different lengths are not similar.
The longer sequences increase RAM consumption and computing very much so you must profile
model for your sequence lengths.

The inputs:
 * PyTorch trained checkpoint.
 * Docker images with Triton and FasterTransformer backend.
 * Model configuration for Navigator

The outputs:
 * The collection of Triton model repositories with FasterTransformer checkpoint ready for inference at production.
 * The optimization results with list of performance metrics.


Model repository preparation for Triton Inference Server:

```
python3 ./infer_scripts/profile_model.py \
    --cluster-config-path ./conf/inference/your_cluster_config.yaml \
    --navigator-config-path ./conf/inference/profile_offline.yaml \
    --model-path /your/path/to/training/checkpoint/ \
    --model-name model_name -v \
    --tensor-parallel-sizes 8 \
    --pipeline-parallel-sizes 1 2 \
    --input-output-lengths 60,20 \
    --max-batch-sizes 256 \
```

The parameters:
 * `cluster-config-path`: cluster configuration YAML file
 * `navigator-config-path`: Navigator configuration YAML e.g. `./conf/inference/profile_offline.yaml`
 * `model-path`: Your path to training checkpoint
 * `model-name`: Your model name for Triton repository
 * `tensor-parallel-sizes`: Tensor parallel factor e.g.: `1 2 4 8` 
 * `pipeline-parallel-sizes`: Pipeline parallel factor e.g.: `1 8`
 * `input-output-lengths`: Analyzed input and output lengths e.g. `20,8 60,20`
 * `max-batch-sizes`: Maximum batch sizes used for optimization e.g. `1 2 4 8 16 256`

The parameters for optimization (like `max-batch_sizes`) are used to 
generate many configuration using Cartesian product. You can set many values
and get from `profile_model.py` script the most optimal configuration.


The `profile_model.py` script creates a folder `infer_workspace_xxx` with
time stamp at the end. It contains certain folders:
 * `model_name-ft_gpu_counts_8-converted.ft`: folders with converted FasterTransformer checkpoints
 * `logs`: logs
 * `model_repo_model_name-mbs_256-pp_1-tp_8-half_1-io_60_20`: Triton model repository for pipeline parallel 1 and tensor parallel 8 for input sequence length 60 and output length 20
 * `model_repo_model_name-mbs_256-pp_2-tp_8-half_1-io_60_20`: Triton model repository for pipeline parallel 2 and tensor parallel 8 for input sequence length 60 and output length 20
 * `navigator_workspace`: Folder to Triton Model Navigator configurations
 * `slurm_workspace`: Folder with slurm logs and sbatch scripts

The `profile_model.py` script prints list of the best models with the name of
Triton model repository with the best results and performance metrics.


### Start server and load model with Triton


The inputs:
 * Triton model repository with FasterTransformer checkpoint ready for inference at production.
 * Docker image with Triton and FasterTransformer backend.

The outputs:
 * Running Triton model instance serving model in slurm cluster.

To run the Triton Model Navigator, do the following:


```
python3 ./infer_scripts/run_tritonserver.py \
    --cluster-config-path ./conf/inference/your_cluster_config.yaml \
    --model-repository-path infer_workspace-xxx/model_repo_model_name-mbs_256-pp_1-tp_8-half_1-io_60_20 \
    -v
```

The parameters:
 * `cluster-config-path`: cluster configuration YAML file
 * `model-repository-path`: Triton model repository path from folder generated by `profile_model.py`


The script saves Triton logs so you can see what happens,
when FasterTransformer loads a checkpoint.


### Verify model accuracy

The inputs:
 * Triton model repository with FasterTransformer checkpoint ready for inference at production.
 * Docker image with Triton and FasterTransformer backend.
 * Lambada dataset.
 * Model vocabulary.
 * Model merges file.

The Triton model repository is generated by the `prepare_model_repository.sh` script.

The English data for accuracy experiments can be downloaded from open resources.

The Lambada dataset you can download from GITHUB:

```
wget https://raw.githubusercontent.com/cybertronai/bflm/master/lambada_test.jsonl
```

The vocabulary and merge files can be downloaded from the Huggingface GPT2 project:

```
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
```

You should put all those files in one folder used for accuracy verification
of your model.


The outputs:
 * Accuracy measurement report.

The accuracy report is stored in the current directory in the file lambada_metrics.csv.

You can verify your model running in Triton by using the Lambada dataset:

```
python3 ./infer_scripts/test_model.py \
    --cluster-config-path ./conf/inference/cluster_bcm.yaml \
    --navigator-config-path ./conf/inference/small_mbs_256-pp_1-tp_1-io_60_20.yaml \
    --model-path /your/path/to/training/checkpoint/ \
    --model-name megatron_345m -v
    --dataset-dir /your/lambada/folder/ 
```


Parameters:
 * `cluster-config-path`: cluster configuration YAML file
 * `navigator-config-path` Navigator configuration to setup Triton. 
 * `model-path`: Your path to training checkpoint
 * `model-name`: model name
 * `dataset-dir`: Folder with downloaded lambada dataset, merges and vocabulary files.


The parameter `navigator-config-path` contains Navigator configuration to setup Triton.
You must set some basic parameters to have working model to verify accuracy.
You can used predefined configuration for this task, which sets basic values
for tiny model:
```
./conf/inference/small_mbs_256-pp_1-tp_1-io_60_20.yaml
```
You must check your model size and look for optimal configuration to run 
accuracy for your model. The larger models must be run with many GPUs and nodes
to work.


### Benchmark inference with random weights

Triton Model Navigator can benchmark inference before training is finished.
If you already know how large model you need to get good accuracy, then you
can use this configuration to create random weights inside Triton Inference 
Server. This random model can be benchmarked to see what is, the best inference
configuration and verify constrains like maximum latency.

The first step in benchmark script generates random checkpoint based
on your configuration. The second configures model repositories and 
starts the Triton Inference Server. The third step executes the performance
measurements inside several cluster nodes.

The FasterTransformer can be just configured with Triton model repository,
but with missing weight files. It will just print warnings during start
and initialize weights to random values. It starts much faster,
when there is no need to load weights, so many configurations can be quite fast
verified with little effort. 

The inputs:
 * Random model configuration 
 * Docker image with training scripts.
 * Docker image with Triton and FasterTransformer backend.
 * Performance profile configuration YAML file.

The outputs:
 * Performance report
 * Triton model stores with random weights.


You can benchmark model using `infer_scripts/run_benchmark_test.sh` script:

```
python3 ./infer_scripts/profile_model_with_random_weights.py \
    --cluster-config-path ./conf/inference/your_cluster_config.yaml \
    --navigator-config-path ./conf/inference/profile_offline.yaml \
    --model-path conf/inference/model_specs/89b.ft \
    --model-name ft_89B \
    --tensor-parallel-sizes 4 8 \
    --pipeline-parallel-sizes 1 2 \
    --input-output-lengths 200,200 \
    --max-batch-sizes 1 2 8 16 32 64 256 \
    --max-latency-ms 1000 \
    -v
```

The parameters:
 * `cluster-config-path`: cluster configuration YAML file
 * `navigator-config-path`: Navigator configuration YAML e.g. `./conf/inference/profile_offline.yaml`
 * `model-path`: This model path contains just YAML file with random checkpoint configuration.
 * `model-name`: Your model name for Triton repository
 * `tensor-parallel-sizes`: Tensor parallel factor e.g.: `1 2 4 8` 
 * `pipeline-parallel-sizes`: Pipeline parallel factor e.g.: `1 8`
 * `input-output-lengths`: Analyzed input and output lengths e.g. `20,8 60,20`
 * `max-batch-sizes`: Maximum batch sizes used for optimization e.g. `1 2 4 8 16 256`
 * `max-latency-ms`: Maximum latency valid for your scenario.

The random model configuration for `model-path` parameter is in YAML file:

```yaml
decoder_layers: 105  # Number of decoder layers
head_num: 128        # Number of heads in layer
size_per_head: 160   # Size per head
inter_size: 81920    # inter_size = size_per_head * head_num * 4
tensor_para_size: 8  # Default tensor parallel configuration (ignored)
vocab_size: 51200    # Vocabulary size based on vocabulary file
start_id: 50256      # ????
end_id: 50256        # ????

```

The output files are saved in the current folder `infer_workspace_xxx` with
time stamp at the end. The all configurations with report about ten best
configurations is printed to terminal.





