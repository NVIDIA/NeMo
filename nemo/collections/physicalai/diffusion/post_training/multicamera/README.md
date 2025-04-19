# Cosmos Multicamera Diffusion Post-Training: User Guide

## Prerequisites

### 1. Review General Requirements

- System Configuration
  - **NVIDIA GPU and driver**: Ensure you have access to the minimum compute required to run the model(s), as listed in the model support matrix.
  - **Containerization Platform**: We recommend using Docker with NVIDIA Container Runtime (alternatively, you may use NVIDIA enroot).
- Get your [Hugging Face User Access Token](https://huggingface.co/docs/hub/en/security-tokens), which is required to obtain the Cosmos models for training and inference.
- Get your [Weights and Biases API Key](https://docs.wandb.ai/support/find_api_key/) for logging and tracking.

### 2. Allocate Resources and Run the Container

Run the following command to allocate a node and launch the container. you can build container using `nemo/collections/physicalai/Dockerfile`
```bash
docker build -f nemo/collections/physicalai/Dockerfile .
```


## Multi-Node

### 3. Modify the Environment Variables Inside of the Training Script

```bash
# nemo/collections/physicalai/diffusion/post_training/multicamera
export HF_TOKEN="your huggingface access token"
export WANDB_API_KEY="your wandb API key"
export WANDB_PROJECT="name of your wandb project"

# use these default paths unless you have your own model/data for post-training
export HF_HOME=/path/to/.cache/huggingface
export XDG_CACHE_HOME=/path/to/.cache
```

### 4. Launch the Training Script using SBATCH

```bash
# multinode scheduled
sbatch -A coreai_dlalgo_llm -N16 --tasks-per-node=8 --gpus-per-node=8 -p polar,polar3,polar4 --dependency=singleton --signal=TERM@600 -t 4:00:0 nemo/collections/physicalai/diffusion/post_training/multicamera/train.sh
```

### OR

### 4. Manually Allocate the Node and Launch the Training Script

```bash
# Can replace the 1 in -N1 with the number of desired nodes
salloc -A coreai_dlalgo_llm -N1 --tasks-per-node=8 --gpus-per-node=8 -p interactive -t 4:00:0
bash nemo/collections/physicalai/diffusion/post_training/multicamera/train.sh
```


# Single Node

### 3. Allocate Resources and Run the Container

Run the following command to allocate a node and launch the container

```bash
salloc -A coreai_dlalgo_llm -N1 --tasks-per-node=8 --gpus-per-node=8 -p interactive -t 4:00:0
srun -N1 --tasks-per-node 1 --container-image path/to/container   --container-mounts "/lustre:/lustre/,/home:/home" --no-container-mount-home     --pty bash
```

### 4. Configure Environment Variables Inside of the Container

navigate into NeMo repo

```bash
# Install packages
pip install lru-dict pyquaternion git+https://github.com/NVIDIA/NeMo-Run.git@f07877f80dbfc91c4d37683864cf2552b96d62f1
pip install -U moviepy==1.0.3

# Export environment variables
export PYTHONPATH=$PYTHONPATH:nemo/collections/physicalai/datasets/dataverse
export HF_TOKEN="your huggingface access token"
export WANDB_API_KEY="your wandb API key"
export WANDB_PROJECT="name of your wandb project"
export WANDB_RESUME=allow
export NVTE_FUSED_ATTN=0 
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# use these default paths unless you have your own model/data for post-training
export HF_HOME=/path/to/.cache/huggingface
export XDG_CACHE_HOME=/path/to/.cache
```

### 5. Post-Train the model
```bash
torchrun --nproc_per_node=8 nemo/collections/physicalai/diffusion/post_training/multicamera/multicamera.py 
```

