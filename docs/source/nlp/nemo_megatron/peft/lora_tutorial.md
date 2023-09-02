# Introduction

This notebook demonstrates how to use NeMo's implementation of LoRA (Low Rank Adaptation) for fine-tuning large 
language models. This implementation is based on the paper, [LORA:LOW-RANK ADAPTATION OFLARGE LANGUAGE MODELS](https://openreview.net/pdf?id=nZeVKeeFYf9) by Hu et al.

This example demonstrates how to:
    
    1. Train a LoRA model on a simple Extractive QA task.
    2. Inspect the trained LoRA model showing the parameters it contains.
    3. Run inference with the based model with the LoRA parameters.

This tutorial focuses on LoRA, but the training and evaluation methods described here are applicable 
for other Parameter-efficient finetuning (PEFT) methods in NeMo. 
It also uses GPT as a base model to demonstrate the PEFT API, but the methods apply to T5 as well.

# Data and Task preparation
Use LoRA to teach the GPT model to do Extractive Question Answering.

Using the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) reading comprehension dataset, consisting of 
questions posed by crowd workers on a set of Wikipedia articles, where the answer to every question is a segment of 
text. More information on [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) can be found on their website or in 
their paper by Rajpurkar et. al 
"[Know What You Don’t Know: Unanswerable Questions for SQuAD](https://arxiv.org/pdf/1806.03822.pdf)".

LoRA (and all PEFT tuning) models expect at least two fields in the jsonl files. The `input` field should contain all 
the tokens necessary for the model to generate the `output`. For example for extractive QA, the `input` should contain 
the context text as well as the question.

```
[
    {"input": "User: Context: [CONTEXT_1] Question: [QUESTION_1]\n\nAssistant:", "output": [ANSWER_1]},
    {"input": "User: Context: [CONTEXT_2] Question: [QUESTION_2]\n\nAssistant:", "output": [ANSWER_2]},
    {"input": "User: Context: [CONTEXT_3] Question: [QUESTION_3]\n\nAssistant:", "output": [ANSWER_3]},
]
```
Note the use of keywords in the input such as, `Context:`, `Question:` to separate the text representing the context and 
question. The keyword `User:` is used and the end each input ends with `\n\nAssistant:` tokens. These are recommended 
because NeMo's instruction-tuned models are trained with a prefix of `User:` and suffix `\n\nAssistant:`.

Download and preprocess the dataset. 
For each dataset there are  preprocessing scripts pre-written in NeMo's example directory located in `examples/nlp`. 
Download those scripts now.

```python
import os

# You can replace DATA_DIR and NEMO_DIR with your own locations
NEMO_DIR = ".."
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
SQUAD_DIR = os.path.join(DATA_DIR, "SQuAD")
os.makedirs(SQUAD_DIR, exist_ok=True)
BRANCH = 'main'
```
```bash
# download the preprocessing scripts from github for the purpose of this tutorial
!wget -nc https://raw.githubusercontent.com/NVIDIA/NeMo/${BRANCH}/scripts/dataset_processing/nlp/squad/prompt_learning_squad_preprocessing.py
# Download the SQuAD dataset
!wget -nc https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
!wget -nc https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
!mv train-v1.1.json ${SQUAD_DIR}
!mv dev-v1.1.json ${SQUAD_DIR}

# Preprocess squad data
!python prompt_learning_squad_preprocessing.py --sft-format --data-dir ${SQUAD_DIR}

# For this tutorial, use a subset of the dataset for demonstration purposes
!head -200 ${SQUAD_DIR}/squad_train.jsonl > ${SQUAD_DIR}/squad_short_train.jsonl
!head -20 ${SQUAD_DIR}/squad_val.jsonl > ${SQUAD_DIR}/squad_short_val.jsonl
```

# Model Setup
## Model Config
To begin setting up the config file needed for PEFT tuning, use a single config for all supported PEFT 
methods (LoRA, Adapter, IA3 and P-Tuning, as well as combinations of these).  All PEFT methods use the GPT fine-tuning 
class `MegatronGPTSFTModel` as the frozen base and use the `add_adapter()` method to add weights for PEFT.

Create a config object for LoRA training.
```python
from omegaconf import OmegaConf
import os
import wget

CONFIG_DIR = os.path.join(NEMO_DIR, "conf")
os.makedirs(CONFIG_DIR, exist_ok=True)

# Download the example config file
wget.download(f'https://raw.githubusercontent.com/NVIDIA/NeMo/{BRANCH}/examples/nlp/language_modeling/tuning/conf/megatron_gpt_peft_tuning_config.yaml', CONFIG_DIR)

# Load the example config file so we can start editing it
CONFIG_PATH = os.path.join(CONFIG_DIR, "megatron_gpt_peft_tuning_config.yaml")
config = OmegaConf.load(CONFIG_PATH)
```

The `config` contains several attributes required by the `MegatronGPTSFTModel`. First set the training data 
path and the validation data path in the config.

The `config` allows you to set a list of `jsonl` files as training files and sample examples from each file with 
different probabilities. For simplicity, you can use just one training file and thus the sampling probability 
is set to `1.0`

You can also monitor validation loss from multiple validation files during training. For simplicity, you can use 
just one validation file.

```python
config.model.data.train_ds.file_names = [f"{SQUAD_DIR}/squad_short_train.jsonl"]
config.model.data.train_ds.concat_sampling_probabilities=[1.0]
config.model.data.validation_ds.file_names = [f"{SQUAD_DIR}/squad_short_val.jsonl"]
config.model.data.validation_ds.names=["squad_val"]
```

## PEFT Config
The attribute [config.model.peft](https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/tuning/conf/megatron_gpt_peft_tuning_config.yaml#L79) 
contains settings that control the PEFT training method and its related hyperpameters. 
NeMO currently supports `lora`, `adapter`, `ptuning` and `ia3`. You can instruct the training script to use one of 
these methods by setting the config.model.peft.peft_scheme attribute.

The other hyperparams associated with LoRA tuning are present in the 
[config.model.peft.lora_tuning](https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/tuning/conf/megatron_gpt_peft_tuning_config.yaml#L96) 
attribute.

```python
config.model.peft.peft_scheme="lora"  # You also set this to adapter or ptuning or ia3
print(OmegaConf.to_yaml(config.model.peft.lora_tuning))
```

print(OmegaConf.to_yaml(config.model.peft.lora_tuning))
#%% md
**Note:** In the original LoRA paper each attention projection (`K`, `Q`, `V` and `O`) can have their own Low-Rank 
projections. However, NeMo's attention implementation fuses `KQV` into a single projection and thus our LoRA 
implementation learns a single Low-Rank projection for `KQV` in a combined fashion. NeMO does not support LoRA for the 
`O` matrix at this point.

## Prompt Formatting
The `config.model.data.train_ds.prompt_template` attribute allows you to further tweak the format of the input and output 
if needed. This example shows  "encoding" the format inside the `jsonl` file directly. This keeps the 
`prompt_template` in the config simple.(See previous section on Data Preparation).

```python
config.model.data.train_ds.prompt_template ="{input} {output}"
```

## Pretrained GPT Model Setup
Next set the "base language model" to perform LoRA tuning. Obviously, larger base models will 
have better performance on downstream tasks but for the purposes of this tutorial a small 345M parameter 
GPT model is used.

```python
# Check what GPT .nemo models are available on NGC
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
megatron_gpt_345m_nemo_url = MegatronGPTModel.list_available_models()[0].location
print(megatron_gpt_345m_nemo_url) # should point to the 345m megatron gpt model '.nemo' file
gpt_file_name = "megatron_gpt_345m.nemo"
```
If you wanted to use the GPT model class directly,  instantiate a trainer then download the model by calling 
running `gpt_model = MegatronGPTModel.from_pretrained(model_name="megatron_gpt_345m", trainer=trainer).cuda()`. 
You need the `.nemo` file in our working NeMo directory in this tutorial, so download it using `wget`. 
```bash
!wget  -nc --content-disposition {megatron_gpt_345m_nemo_url} -O {NEMO_DIR}/{gpt_file_name}
```

Next,  set where you want to save all the intermediate training logs and checkpoints. As well as other training 
settings such as: number of training steps, batch size and validation check interval, and num_workers for data 
processing.

```python
config.model.restore_from_path = gpt_file_name
config.exp_manager.exp_dir=f"{NEMO_DIR}/peft_lora"
config.exp_manager.explicit_log_dir="training_info"
config.trainer.max_steps=100
config.model.micro_batch_size=1
config.model.global_batch_size=4
config.trainer.val_check_interval=50
config.model.data.train_ds.num_workers=0  # 0 is recommended which uses the main thread to process training examples
config.model.data.validation_ds.num_workers=0 # 0 is recommended which uses the main thread to process the validation examples
print(OmegaConf.to_yaml(config.model))
```

# Training Setup
## Building the PyTorch Lightning Trainer
NeMo models are primarily PyTorch Lightning modules and therefore are entirely compatible with the PyTorch Lightning 
ecosystem.
Modify the trainer config and then instantiate a Trainer object with `MegatronTrainerBuilder`

```python
import torch
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder

# let's modify some trainer configs
# check if we have GPU available and uses it
accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
config.trainer.accelerator = accelerator
config.trainer.devices = 1
config.trainer.max_epochs = 4
config.trainer.val_check_interval = 1.0

# for PyTorch Native AMP set precision=16
config.trainer.precision = 16 if torch.cuda.is_available() else 32

print("Trainer config - \n")
print(OmegaConf.to_yaml(config.trainer))

trainer = MegatronTrainerBuilder(config).create_trainer()
```

## Setting up a NeMo Experiment

NeMo has an experiment manager that handles logging and checkpointing for you.
```python
from nemo.utils.exp_manager import exp_manager

# Set name of the experiment 
config.name = 'lora_example_tuning'
config.exp_manager.resume_if_exists = False
print(OmegaConf.to_yaml(config.exp_manager))

# Init the experiment manager and view the exp_dir
exp_dir = exp_manager(trainer, config.get("exp_manager", None))
print(exp_dir)
```

# Training
To set up the process for training a LoRA model, first require a config that contains details about the base 
language model upon which we will train our LoRA model. First extract the `model_cfg` from the checkpoint and 
update it with any new settings we employ in our current (LoRA) `config`. These are merged in the `merge_cfg_with` 
function.

```python
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel

model_cfg = MegatronGPTSFTModel.merge_cfg_with(config.model.restore_from_path, config)
```

Next, instantiate the GPT model class and add the LoRA adapter.
When you call `add_adapter`, the model prints out the parameter count before and after the operation. 
You can see the number of trainable parameters increase after adding the adapter.
To print the parameter count manually, call `model.summarize()`.

```python
from nemo.collections.nlp.parts.peft_config import LoraPEFTConfig

model = MegatronGPTSFTModel.restore_from(config.model.restore_from_path, model_cfg, trainer=trainer)
model.add_adapter(LoraPEFTConfig(model_cfg))

print("Parameter count manually:\n", model.summarize())
```
Simply substitute with the `MegatronT5SFTModel` class to use T5 instead of GPT.

To use a different PEFT method, you can use a different config class in place of `LoraPEFTConfig`, such as `AttentionAdapterPEFTConfig`, `IA3PEFTConfig`, `PtuningPEFTConfig`. You can also use a combination of the methods by passing in a list:
`model.add_adapter([LoraPEFTConfig(model_cfg), PtuningPEFTConfig(model_cfg)])`

You can now start training.
```python
trainer.fit(model)
```

Once training is completed you should see a saved '.nemo' file in this folder 
`{config.exp_manager.explicit_log_dir}/checkpoints`. 
This checkpoint will only contain the trained adapter weights, and not the frozen base model weights.

# Inference
The model object from `trainer.fit(model)` is also capable of doing inference. For the tutorial, re-load the 
saved `.nemo` lora model along with a `.nemo` base language model to simulate a more realistic scenario 
(where training does not happen right before inference).

First, load and modify a config file that will be used for inference.
```python
# Download the example config file
wget.download(f'https://raw.githubusercontent.com/NVIDIA/NeMo/{BRANCH}/examples/nlp/language_modeling/tuning/conf/megatron_gpt_peft_eval_config.yaml', CONFIG_DIR)

# Load the example config file to start editing it
CONFIG_EVAL_PATH = os.path.join(CONFIG_DIR, "megatron_gpt_peft_eval_config.yaml")
config_eval = OmegaConf.load(CONFIG_EVAL_PATH)
```
Modify the `config_eval` object that you created above. Set the base language model as the `345m`
model you downloaded earlier.

Additionally, set the `model.peft.restore_from_path` with the LoRA model you just trained. 
For the tutorial, use the validation data for inference as well.

```python
config_eval.model.restore_from_path="megatron_gpt_345m.nemo"
config_eval.model.peft.restore_from_path="./training_info/checkpoints/lora_example_tuning.nemo"
config_eval.model.data.test_ds.file_names=[f"{SQUAD_DIR}/squad_short_val.jsonl"]
config_eval.model.data.test_ds.names=["test_set"]
config_eval.model.data.test_ds.global_batch_size=1
config_eval.model.data.test_ds.micro_batch_size=1
config_eval.model.data.test_ds.tokens_to_generate=30
config_eval.inference.greedy=True

trainer_eval = MegatronTrainerBuilder(config_eval).create_trainer()

```

The `config_eval` object is the hydra config at inference/test time. This means it should contain information relevant 
for inference/test time. You still need to know some properties that were set at training time 
for example, was the training done with `BOS` enabled or not, and other model specific attributes.

Extract the `peft_model_cfg` from the '.nemo' file of the LoRA model you trained. Then load the base language model as well as the lora model you trained.

```python
model_cfg = MegatronGPTSFTModel.merge_cfg_with(config_eval.model.restore_from_path, config_eval)
model_eval = MegatronGPTSFTModel.restore_from(config_eval.model.restore_from_path, model_cfg, trainer=trainer_eval)
model_eval.load_adapters("./training_info/checkpoints/lora_example_tuning.nemo")  # TODO load_adapters need some change
model_eval.freeze()
```

Next, prepare the dataset and the dataloader objects that the model will perform inference on.
```python
_test_ds = model_eval._build_dataset(peft_model_cfg.data.test_ds, is_train=False)
from torch.utils.data import DataLoader
request_dl = DataLoader(
    dataset=_test_ds[0],
    batch_size=peft_model_cfg.data.test_ds.global_batch_size,
    collate_fn=_test_ds[0].collate_fn,
)
config_inference = OmegaConf.to_container(config_eval.inference, resolve=True)
model_eval.set_inference_config(config_inference)
```

Finally, call `trainer.predict` which triggers the inference process. The `response` object contains the outputs of the model.
```python
response = trainer_eval.predict(model_eval, request_dl)
for batch in response:
    for s in batch['sentences']:
        print(f"{s}\n\n")
```
