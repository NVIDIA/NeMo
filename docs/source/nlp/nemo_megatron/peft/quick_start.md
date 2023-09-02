# Quick Start Guide

The quick start guide provides an overview to a PEFT workflow in NeMo. 

#### Terminology: PEFT vs Adapter
In the tutorial we will use "PEFT" to describe the overall parameter efficient finetuning method, and "adapter"
to describe the additional module injected to a frozen base model. Each PEFT model may use one or more types of adapters.

Unfortunately, one of the PEFT methods is sometimes confusingly called "adapter", 
since it was one of the first proposed usage of adapter modules. 
We will call this PEFT method "Attention Adapter" or the "Canonical Adapter" to distinguish the two. 


## How PEFT work in NeMo models 
Each PEFT method has one or more types of adapters that need to be injected into the base model. 
In NeMo models, the adapter logic and adapter weights are already built into the submodules, 
but they are disabled by default for ordinary training and finetuning. 

When doing PEFT, the adapter logic path can be enabled when `model.add_adapter(peft_cfg)` is called.
Inside of this function, the model scans through each adapter applicable to the current 
PEFT method with each of its submodules in order to find adapter logic paths that can be enabled. 
Then, the base models weights are frozen, while newly added adapter weights are unfrozen and allowed to be updated 
during finetuning, hence achieving efficiency in the number of parameters finetuned.


## PEFT config classes 
Each PEFT method is specified by a `PEFTConfig` class which stores the types of adapters applicable to the PEFT method,
as well as hyperparameters required to initialize these adapter modules. There are four PEFT methods currently supported:
1. Attention (Canonical) Adapter: `AttentionAdapterPEFTConfig`
2. LoRA: `LoraPEFTConfig`
3. IA3: `IA3PEFTConfig`
4. P-Tuning: `PtuningPEFTConfig`

These config classes make experimenting with different adapters as easy as changing the config class.

Moreover, it is possible to use a combination of the PEFT methods in NeMo since they are orthogonal to each other. 
This can be easily done by passing in a list of 
`PEFTConfig` objects to `add_adapter` instead of a single one. 
For example, a common workflow is to combine P-Tuning and Adapter, and this can be achieved with 
`model.add_adapter([PtuningPEFTConfig(model_cfg), AttentionAdapterPEFTConfig(model_cfg)])`


## Base model classes
PEFT in NeMo is built with a mix-in class that does not belong to any model in particular. This means that the same 
interface is available to different NeMo models. At the moment, we support GPT (`MegatronGPTSFTModel`) and 
T5 (`MegatronT5SFTModel`). Support for more models will be added in the near future!



## Full finetuning vs PEFT
It is easy to switch between full finetuning and PEFT by simply removing calls to `add_adapter` and `load_adapter`. 

The code snippet below illustrate the core API of full finetunin and PEFT.

```diff
trainer = MegatronTrainerBuilder(config).create_trainer()
model_cfg = MegatronGPTSFTModel.merge_cfg_with(config.model.restore_from_path, config)

model = MegatronGPTSFTModel.restore_from(restore_path, model_cfg, trainer) # restore from pretrained ckpt
peft_cfg = LoRAPEFTConfig(model_cfg)
+ model.add_adapter(peft_cfg) 
trainer.fit(model)  # saves adapter weights only

# Restore from base then load adapter API 
model = MegatronGPTSFTModel.restore_from(restore_path, trainer, model_cfg)
+ model.load_adapters(adapter_save_path, peft_cfg)
model.freeze()
trainer.predict(model)
```