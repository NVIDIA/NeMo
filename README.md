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


## Benchmark multinode inference
(Only tested on selene)

```
cd bignlp-scripts/infer_scripts
bash multi_node_infer.sh
```
