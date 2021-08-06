# BigNLP-Scripts

Scripts and code to provide end-to-end data preparation and training for
Megatron-LM.

The training dataset is "[The Pile](https://pile.eleuther.ai/)" and it consists
of 22 smaller dataset. The dataset is already blended by using the mix
described in their [paper](https://arxiv.org/pdf/2101.00027.pdf).

It is best to store this repository in a file system shared by all the nodes,
since the downloaded datasets will be stored inside this repository.


## Dependencies:
Download or clone the Megatron-LM repository and place it inside the
bignlp-scripts directory. Both the preprocessing and training need to run code
in the Megatron-LM repository.


## Download and Preprocess Training Dataset
Main directory: prepare_dataset


#### Usage:
```
cd bignlp-scripts/prepare_dataset
bash prepare_training_data.sh
```

#### Tricks and Tips:
##### Using a Subset of the Training Data
If you want to download only a subset of the training data, you can download
any of the 30 files that comprise it. You can do this by changing the array
information on the following scripts:
```
prepare_dataset/download/download_all_pile_files.sh
prepare_dataset/extract/extract_all_pile_files.sh
prepare_dataset/preprocess/preprocess_all_pile_files.sh
```
Set the #SBATCH --array=0-29 to the files you want to download, using any
numbers from 0 to 29, either separated by a comma or a dash.
If you do this, you will also need to modify the gpt_blend.sh file to match the
datasets to use for training.

For example, to download only the 0th file:
```
#SBATCH --array=0
```


##### Limiting the number of nodes in the cluster
The easiest way to limit the number of nodes to add is to add a %N after the
array directive. For example, to download all 30 files (0-29) using only N=4
nodes, you can use:
```
#SBATCH --array=0-29%4
```


#### Result Dataset:
The result will be the dataset in binary format (.bin) stored in a directory
named `the_pile`.



## Training GPT-3
Main directory: train_scripts

#### Train 126M Parameter Model:
To train the 126M model do:
```
cd bignlp-scripts/train_scripts
sbatch run_gpt3_126m.sh
```


## Download the Test Data

#### LAMBADA Dataset:
To download the LAMBADA test dataset do:
```
cd bignlp-scripts/prepare_dataset
bash prepare_lambada_test_data.sh
```


## Evaluating GPT-3
Main directory: eval_scripts

#### Evaluate 126M Parameter Model on LAMBADA:
To evaluate the 126M model do:
```
cd bignlp-scripts/eval_scripts
sbatch eval
```

