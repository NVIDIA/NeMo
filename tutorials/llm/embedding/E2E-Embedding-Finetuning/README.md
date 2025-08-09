# Fine-Tuning Embedding Models with NeMo

This repository offers comprehensive scripts and instructions for fine-tuning embedding models using NVIDIA NeMo. It covers dataset preparation, model import from Hugging Face, fine-tuning on multi-GPU setups, and experiment monitoring.

## 📂 Repository Structure

```
.
├── README.md                # This file
├── download_dataset.py      # Download & prepare dataset
├── import_e5_large.py       # Import E5-Large-V2 HF model into NeMo
├── import_llama1b.py        # Import LLaMA-3.2-1B HF model into NeMo
├── finetune_e5.py           # Fine-tune E5 on dataset
└── finetune_llama1b.py      # Fine-tune LLaMA on dataset
```

## 🎯 Workflow Overview

### 1. Dataset Preparation
Download and prepare the AllNLI triplet dataset:
```bash
python3 download_allnli_triplet.py
```
This generates `allnli_triplet.json` with entries like:
```
[
  {"query": "example query", "pos_doc": "positive document", "neg_doc": "negative document"},
  ...
]
```

### 2. Model Import
Import pre-trained models from Hugging Face to NeMo format:
- For E5-Large-V2:
  ```bash
  python3 import_e5_large.py
  ```
  Output: `e5-large-v2.nemo`
- For LLaMA-3.2-1B:
  ```bash
  python3 import_llama1b.py
  ```
  Output: `Llama-3.2-1B.nemo`

### 3. Fine-Tuning
Fine-tune the models using the dataset:
- E5-Large-V2:
  ```bash
  python3 finetune_e5.py 
  ```
- LLaMA-3.2-1B:
  ```bash
  python3 finetune_llama1b.py
  ```
Edit hyperparameters directly in the scripts for customization.

### 4. Monitoring and Experimentation
Use Jupyter Lab within an Enroot container to monitor training progress and access GPUs. Refer to the Quickstart section for setup.

## 🛠️ Prerequisites

- **Hardware**: Single node with at least 1 GPUs (e.g., 8×A100 or 8×H100).
- **Software**:
  - Enroot and NVIDIA Container Toolkit.
  - Optional: Docker
  - NVIDIA NeMo Container (e.g., `nvcr.io/nvidia/nemo:25.04`).
  - Hugging Face token set as `$HUGGINGFACE_TOKEN`.

## 🚀 Quickstart: Launch Jupyter Lab in Container

1. Pull and create the Enroot container:
   ```bash
   enroot import "docker://$HUGGINGFACE_TOKEN@nvcr.io#nvidia/nemo:25.04"
   enroot create -n nemo-25.04 "$PWD/nvidia+nemo+25.04.sqsh"
   ```

2. Mount and start the container:
   ```bash
   mkdir -p .jupyter_runtime .jupyter_config
   enroot start --root \
     --mount "$PWD:/workspace" \
     --env NVIDIA_VISIBLE_DEVICES=all \
     --env NVIDIA_DRIVER_CAPABILITIES=all \
     --env JUPYTER_RUNTIME_DIR=/workspace/.jupyter_runtime \
     --env JUPYTER_CONFIG_DIR=/workspace/.jupyter_config \
     nemo-25.04 \
     jupyter-lab --ip=0.0.0.0 --allow-root --NotebookApp.token="embedding" --port=1234 --notebook-dir=/workspace
   ```

3. Inside Jupyter Lab:
   - Check GPU availability: `!nvidia-smi`
   - Log in to Hugging Face: `!huggingface-cli login`
   - Execute scripts or convert them to notebooks for interactive runs.

## ⚙️ Optional Docker Setup
For docker container environments:
```bash
docker run \
  --gpus all \
  --shm-size=2g \
  --net=host \
  --ulimit memlock=-1 \
  --rm -it \
  -v ${PWD}:/workspace \
  -w /workspace \
  nvcr.io/nvidia/nemo:25.04 bash

jupyter-lab --ip=0.0.0.0 --allow-root --NotebookApp.token="embedding" --port=1234 --notebook-dir=/workspace

```

