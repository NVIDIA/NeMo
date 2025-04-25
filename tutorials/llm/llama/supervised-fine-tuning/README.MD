# ChipNeMo - Supervised Fine Tuning on Llama 2 7b with NeMo Framework

[](https://github.com/NVIDIA/NeMo/tree/main/tutorials/llm/llama/domain-adaptive-pretraining#chipnemo---custom-tokenization--domain-adaptive-pre-training-on-llama-2-7b-with-nemo-framework)

[ChipNeMo](https://arxiv.org/pdf/2311.00176) is a chip design domain-adapted Large Language Model (LLM). Instead of directly deploying off-the-shelf commercial or open-source LLMs, the paper adopts the following domain adaptation techniques: domain-adaptive tokenization, domain-adaptive continued pre-training, model alignment with domain-specific instructions, and domain-adapted retrieval models. Specifically, Llama 2 foundation models are continually pre-trained with more than 20 billion tokens on domain-specific chip design data, including code and documents. They are then fine-tuned with instruction datasets from design data as well as external sources. Evaluations on the resultant domain-adapted ChipNeMo model demonstrate that domain-adaptive pre-training of language models can lead to superior performance in domain-related downstream tasks compared to their base Llama 2 counterparts, without degradations in generic capabilities.

Here, we share a tutorial with best practices on SFT (Supervised Fine Tuning) for a ChipNeMo-like code generation use case.

## Requirements

[](https://github.com/NVIDIA/NeMo/tree/main/tutorials/llm/llama/domain-adaptive-pretraining#requirements)

### Software Requirements

[](https://github.com/NVIDIA/NeMo/tree/main/tutorials/llm/llama/domain-adaptive-pretraining#software-requirements)

* Access to latest NeMo Framework NGC Containers
* This playbook has been tested on: nvcr.io/nvidia/nemo:25.02. It is expected to work similarly on other environments.

### Hardware Requirements

[](https://github.com/NVIDIA/NeMo/tree/main/tutorials/llm/llama/domain-adaptive-pretraining#hardware-requirements)

* This playbook can run on CPUs or GPUs. For GPUs, this playbook has been tested on minimum 2xA100 80G

### Data Curation

[](https://github.com/NVIDIA/NeMo/tree/main/tutorials/llm/llama/domain-adaptive-pretraining#data-curation)

* In this tutorial, we will leverage open source verilog dataset from huggingface contain natural language code description and actual verilog pairs.
* We implement custom data download, preprocessing and split logics. This is needed if you plan to use customized data instead of popular out of the box benchmarks which NeMo2.0 framework provides.

## Pretraining for DAPT

This step is covered in the *domain-adaptive-pretraining* folder


## Supervised Fine Tuning

After DAPT, the model is exposed to domain specific data including verilog code. We further customize the model with curated high quality natural language-verilog code SFT data. This would help the model achieve instruction following ability and act as a coding assistant given instruction to generate Verilog codes. Detailed step-by-step walk through can be found in the `supervised_fine_tuning_nemo2.0.ipynb` notebook.


## Deployment of .nemo via NIMs

[](https://github.com/NVIDIA/NeMo/tree/main/tutorials/llm/llama/domain-adaptive-pretraining#deployment-of-nemo-via-nims)

Once the SFT model is completed a .nemo checkpoint will be saved. Follow the document 'Deploying .nemo as a NIM' to deploy the .nemo checkpoint as NIMs and send inference request.
