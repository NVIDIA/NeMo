# Cosmos Diffusion-based World Foundation Models: NeMo Framework User Guide

Learn how to [post-train](#post-train) Cosmos Diffusion-based World Foundation Models (WFMs) using the [NVIDIA NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html) for your custom Physical AI tasks by following this guide.

## Model Support Matrix

The NeMo Framework supports the following Cosmos Diffusion models. Review the available models and their compute requirements for post-tuning and inference to determine the best model for your use case.

| Model Name                               | Model Status | Compute Requirements for Post-Training |
|----------------------------------------------|------------------|------------------------------------------|
| Cosmos-1.0-Diffusion-7B-Text2World           | **Supported**    | 8 NVIDIA GPUs*                           |
| Cosmos-1.0-Diffusion-14B-Text2World          | **Supported**    | 8 NVIDIA GPUs*                           |
| Cosmos-1.0-Diffusion-7B-Video2World          | **Supported**    | 8 NVIDIA GPUs*                           |
| Cosmos-1.0-Diffusion-14B-Video2WorldB        | **Supported**    | 8 NVIDIA GPUs*                           |


**\*** `H100-80GB` or `A100-80GB` GPUs are recommended.

## Post-Training Support Matrix

Cosmos Diffusion-based WFMs can be post-trained for a variety of Physical AI tasks. Review the following table for a list of available Physical AI post-training tasks:

| Post-training Task  | Post-Training Support Status |
|-------------------------|--------------------|
| General post-training     | **Supported**      |
| Instruction control     | **Supported**    |
| Action control          | **Supported**    |
| Camera control          | **Supported**    |
| Multi-view generation   | **Supported**    |
| Multi-view generation with vehicle trajectory control | **Supported** |

## Prerequisites

### 1. Review General Requirements

- System Configuration
  - **NVIDIA GPU and driver**: Ensure you have access to the minimum compute required to run the model(s), as listed in the model support matrix.
  - **Containerization Platform**: We recommend using Docker with NVIDIA Container Runtime (alternatively, you may use NVIDIA enroot).
- Get your [Hugging Face User Access Token](https://huggingface.co/docs/hub/en/security-tokens), which is required to obtain the Cosmos models for training and inference.
- Get your [Weights and Biases API Key](https://docs.wandb.ai/support/find_api_key/) for logging and tracking.

### 2. Clone the Cosmos Repository

```bash
git clone git@github.com:NVIDIA/Cosmos.git
```

### 3. Start the Container

The [NeMo Framework container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo) supports post-training and inference for Cosmos Diffusion models.

Run the following command to download and start the container:
```bash
docker run --ipc=host -it --gpus=all \
  -v $PATH_TO_COSMOS_REPO:/workspace/Cosmos \
  nvcr.io/nvidian/nemo:cosmos.1.0.2 bash
```

### 4. Download Checkpoints

To help you get started, we've provided a [download script](../download_diffusion_nemo.py) to get the Cosmos Diffusion Text2World and Video2World checkpoints from Hugging Face. These checkpoints are in the NeMo distributed checkpoint format required to run post-training and inference with NeMo Framework.

1. Set the following environment variables:
   ```bash
   # You must set HF_HOME before running this script.
   export HF_TOKEN="<your/HF/access/token>"
   export HF_HOME="<path/to/store/checkpoints>"
   ```
2. Run the following command to download the models:
   ```bash
   cd /workspace/Cosmos
   python cosmos1/models/diffusion/nemo/download_diffusion_nemo.py
   ```

## Post-train

Post-training a Cosmos Diffusion-based WFM enables you to train the model to generate videos that are more specific to your Physical AI use case.

For example, if you want to generate action sequences for a specific robot, you can post-train the model to generate videos that are more aligned with typical actions/outcomes for that robot.

There are 3 steps to post-training: preparing a dataset, preprocessing the data, and post-training the model.

### 1. Prepare a Dataset

The first step is to prepare a dataset. Post-training a Cosmos-1.0-Diffusion-Text2World/Cosmos-1.0-Diffusion-Video2World model enables you to generate videos of a specific subject in new environments using a collection of input videos of that same subject as reference material.

You must provide a folder containing a collection of videos in **MP4 format**, preferably 720p. These videos should focus on the subject throughout the entire video so that each video chunk contains the subject.

Run the following command to download the sample videos used for post-training:

```bash
huggingface-cli download nvidia/Cosmos-NeMo-Assets --repo-type dataset --local-dir cosmos1/models/diffusion/assets/ --include "*.mp4*"
```

### 2. Preprocess Data for Single Subject Post-training

The second step is to preprocess the input videos. This generates the post-training samples and the metadata required for the post-training process by:

1. Selecting `N` chunks of 121 frames from each video, generating `N` post-training samples per video.
2. Encoding the 121 frames by first independently compressing the first frame and then applying an 8x temporal compression for the rest of the frames.
3. Generating `total_samples = # of videos x # of chunks` post-training samples.

Before proceeding, ensure all videos are in **RGB format**. Complete the following steps to generate the post-training samples and metadata for the robot dataset. Remember to follow the given prompt format by including the subject's name in the prompt. For example, if the subject is "robot," the prompt should read `"A video of sks robot."`.

1. Set the following environment variables:
   ```bash
   export HF_TOKEN="<your/HF/access/token>"
   export HF_HOME="<path/to/store/checkpoints>"

   # Path to Raw mp4 videos.
   export RAW_DATA="cosmos1/models/diffusion/assets/nemo_diffusion_example_data"

   # Path to Processed Dataset.
   export CACHED_DATA="./cached_data" && mkdir -p $CACHED_DATA
   ```

2. Run the following command to preprocess the data:

   ```bash
   python cosmos1/models/diffusion/nemo/post_training/prepare_dataset.py \
   --dataset_path $RAW_DATA \
   --output_path $CACHED_DATA \
   --prompt "A video of sks teal robot." \
   --num_chunks 500
   ```

Executing the [data preprocessing script](./prepare_dataset.py) generates the following files for each video (using `[i]` as the `index` of the video) at `$CACHED_DATA` path:

- **`[i].info.json`**: Metadata for the video sample.
- **`[i].t5_text_embeddings.pth`**: T5-generated text embedding for the video clip.
- **`[i].t5_text_mask.pth`**: Mask for T5 text embedding, set to all ones by default to use the entire text embedding.
- **`[i].video_latent.pth`**: 3D spatiotemporal video tokens generated from the video tokenizer.
- **`[i].conditioning_latent.pth`**: 3D spatiotemporal video tokens generated from the video tokenizer on the first nine frames of the input video. These conditioning latents are only used during Video2World training.

### 3. Preprocess Data for Robot Instruction (or other Custom Prompt) Post-training

Robot instruction post-training uses instructions as input prompts. Instructions are imperative prompts and correspond to the physical actions performed by the robot in a video. The instruction dataset processing workflow generalizes to any custom input prompt per video.

1. Create instruction dataset

Create a dataset folder containing videos and per video instructions in the following format:

```
robot_dataset
   videos
      id1.mp4
      id2.mp4
      ...
   instructions
      id1.json
      id2.json
```

- **`robot_dataset/videos/id1.mp4`**: video clip
- **`robot_dataset/instructions/id1.json`**: json file with key `language_instruction_0` mapping to a text instruction

2. Run the following command to preprocess the data:
   ```bash
   python cosmos1/models/diffusion/nemo/post_training/prepare_instruction_dataset.py \
   --dataset_path robot_dataset \
   --output_path robot_dataset/processed \
   --num_chunks 500
   ```
The output dataset is saved in `robot_dataset/processed/` in the same format described in the previous section.

### 3. Post-train the Model

The third step is to post-train the model. This step uses NeMo Framework's data and model parallelism capabilities to train the model on the post-training samples. This is accomplished by using utilizing Fully Sharded Data Parallel (FSDP) and Tensor Parallelism.

- **FSDP**: Distributes model parameters, optimizer states, and activations across all GPUs
- **Tensor Parallelism**: Spreads the parameter tensor of individual layers across GPUs.

> **NOTE**:
> For the 14B model, we also employ activation checkpointing to facilitate single-node training.

#### Run the Post-training Script


Complete the following steps to post-train the Cosmos-1.0-Diffusion-7B-Text2World or Cosmos-1.0-Diffusion-7B-Video2World models on the robot dataset using 8 GPUs.

##### Text2World

1. Set the following environment variables:
   ```bash
   export HF_TOKEN="<your/HF/access/token>"
   export HF_HOME="<path/to/store/checkpoints>"

   # Optionally, you can monitor training progress with Weights and Biases (wandb).
   export WANDB_API_KEY="</your/wandb/api/key>"
   export WANDB_PROJECT_NAME="cosmos-diffusion-nemo-post-training"
   export WANDB_RUN_ID="cosmos_diffusion_7b_text2world_finetune"
   ```
2. Run the following command for Cosmos-Diffusion-Text2World-7B general post-training:
   ```bash
   NVTE_FUSED_ATTN=0 \
   CUDA_DEVICE_MAX_CONNECTIONS=1 \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   torchrun --nproc_per_node=8 cosmos1/models/diffusion/nemo/post_training/general.py \
       --yes \
       --factory cosmos_diffusion_7b_text2world_finetune \
       data.path=$CACHED_DATA \
       trainer.max_steps=1000 \
       optim.config.lr=1e-6
   ```

###### Configuration Options

Before getting started, review the following parameters made available to the script. You can adjust these parameters to optimize performance based on your specific requirements.

| Parameter                      | Description                                                                     | Default |
|--------------------------------|---------------------------------------------------------------------------------|---------|
| `--factory`                   | recipe to use cosmos_diffusion_7b_text2world_finetune or cosmos_diffusion_14b_text2world_finetune for general post-training                                   | cosmos_diffusion_7b_text2world_finetune    |
| `data.path`                   | Path to processed post-training dataset (str).                                    | None    |
| `resume.restore_config.path`  | Path to pre-trained Cosmos Diffusion NeMo distributed checkpoint (str).         | None    |
| `optim.config.lr`             | Learning rate (float).                                                          | 1e-6    |

##### Video2World

1. Set the following environment variables:
   ```bash
   export HF_TOKEN="<your/HF/access/token>"
   export HF_HOME="<path/to/store/checkpoints>"

   # Optionally, you can monitor training progress with Weights and Biases (wandb).
   export WANDB_API_KEY="</your/wandb/api/key>"
   export WANDB_PROJECT_NAME="cosmos-diffusion-nemo-post-training"
   export WANDB_RUN_ID="cosmos_diffusion_7b_video2world_finetune"
   ```
2. Run the following command for Cosmos-Diffusion-Video2World-7B general post-training:
   ```bash
   NVTE_FUSED_ATTN=0 \
   CUDA_DEVICE_MAX_CONNECTIONS=1 \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   torchrun --nproc_per_node=8 cosmos1/models/diffusion/nemo/post_training/video2world.py \
       --yes \
       --factory cosmos_diffusion_7b_video2world_finetune \
       data.path=$CACHED_DATA \
       trainer.max_steps=1000 \
       optim.config.lr=1e-6
   ```

You can now run inference with your post-trained model using the instructions [here](../inference/README.md#run-the-inference-script-with-post-trained-model).

###### Configuration Options

Before getting started, review the following parameters made available to the script. You can adjust these parameters to optimize performance based on your specific requirements.

| Parameter                      | Description                                                                     | Default |
|--------------------------------|---------------------------------------------------------------------------------|---------|
| `--factory`                   | recipe to use cosmos_diffusion_7b_video2world_finetune or cosmos_diffusion_14b_video2world_finetune for video2world post-training                                   | cosmos_diffusion_7b_video2world_finetune    |
| `data.path`                   | Path to processed post-training dataset (str).                                    | None    |
| `resume.restore_config.path`  | Path to pre-trained Cosmos Diffusion NeMo distributed checkpoint (str).         | None    |
| `optim.config.lr`             | Learning rate (float).                                                          | 1e-6    |
| `trainer.max_steps`           | Max number of post-training steps (int).                                             | 1000    |
| `log.log_dir`                 | Path to folder to save post-training logs and checkpoints (str).                     | None    |


## DiT Video2World Action Control Post-Training

### Prerequisites

#### Data Pre-Processing and Tokenization

To preprocess and tokenize video data for diffusion-based action control, refer to the documentation in [action_control/README.md](./action_control/README.md). Diffusion datasets have a different continuous T5 video tokenizer than the auto-regressive model variant. The action control dataset should be downloaded to the default `HF_HOME=/root/.cache/huggingface` (or `HF_CACHE_DIR`), which we will mount to `HF_HOME=/root/.cache/huggingface` in our development containers.

#### Diffusion-Based V2W Action Control Source Code

The base container for DiT V2W action control can be downloaded with the following command:

```bash
docker pull nvcr.io/nvidian/nemo:cosmos.1.0.2
```

##### Cosmos

Clone and checkout the `main` branch of Cosmos.

```bash
git clone https://github.com/NVIDIA/Cosmos.git
cd Cosmos
COSMOS_CODE_DIR=$(pwd)
```

### Action Control Fine-Tuning

When you have preprocessed datasets and model checkpoints cached in `HF_CACHE_DIR` or the default `HF_HOME=/root/.cache/huggingface`, as well as the above repositories cloned and checked out, we have all the resources necessary for action control fine-tuning.

#### Setup

Fine-tuning can be run with both Docker or Slurm.

##### Docker Run

To setup and interact with a Docker container for action control fine-tuning, run the following command:

```bash
docker run --rm -it --name dit-action-ctrl-train-job --gpus all -v $COSMOS_CODE_DIR:/opt/Cosmos -v $HF_CACHE_DIR:$HF_HOME nvcr.io/nvidian/nemo:cosmos.1.0.2
```

##### Slurm `srun`

Requires an `enroot` container image, which you can build using the following command:

```bash
enroot import -o cosmos-action-control-nemo-25.02rc3.sqsh docker://nvcr.io/nvidian/nemo:cosmos.1.0.2
```

For instructions on how to install enroot on your system, refer to the Enroot documentation: https://github.com/NVIDIA/enroot.

Then run the following command for interactive single-node training:

```bash
srun --partition <partitions> --account <account> --nodes 1 --gpus-per-node <gpus_per_node> --job-name <jobname> --ntasks-per-node <gpus_per_node> --time <runtime> --container-mounts=$COSMOS_CODE_DIR:/opt/Cosmos,$HF_CACHE_DIR:$HF_HOME --container-image cosmos-action-control-nemo-25.02rc3.sqsh --exclusive --pty bash
```

#### Action Control Fine-Tuning

Inside your container or compute node environment, run the following command to start fine-tuning the (7B) DiT model:

```bash
export WANDB_API_KEY=<WANDB_KEY>       # Existence of this ENV variable activates WandB logging when this is exported to the ENV. If not set, no logs will be uploaded to Weights & Biases.
export WANDB_NAME=<JOB_NAME>
export WANDB_RUN_GROUP=<GROUP_NAME>
export WANDB_PROJECT=<PROJECT_NAME>
export EXP_DIR=<EXPERIMENT_DIR>        # Experiment directory that holds your logs and checkpoints for your run. Exposed here for easy access.
export HF_TOKEN=<HF_TOKEN>             # Used to download models or the dataset. Not necessary if you already prepared your HuggingFace cache.

# Change to our mounted Cosmos working directory.
cd /opt/Cosmos

# Launch fine-tuning (in the Cosmos/ directory) with the `cosmos_diffusion_7b_video2world_action_ctrl_finetune` recipe!
torchrun --nproc_per_node 8 cosmos1/models/diffusion/nemo/post_training/video2world.py --yes --factory cosmos_diffusion_7b_video2world_action_ctrl_finetune data.global_batch_size=2 trainer.strategy.tensor_model_parallel_size=8 optim.config.lr=1e-5
```

To review the complete set of parameters such as the learning rate, checkpointing strategy, model parameters, required video metadata and Tensor `dtype`, and more, the `cosmos_diffusion_{model_size}_video2world_action_ctrl_finetune` NeMo2 recipe and `DiTConfig` are located in these modules:

- NeMo2 Recipe: `nemo-vfm/nemo/collections/diffusion/train.py::finetune_{7b,14b}_action_control`
  - Training and checkpoint settings.
- `DiTConfig`: `nemo-vfm/nemo/collections/diffusion/models/model.py::DiT{7B,14B}Video2WorldActionConfig`
  - Model architecture configuration, including the action control vector dimension, e.g. `config.action_emb_dim=7` (i.e. of shape `(B, 7)`) for the Bridge dataset.
- Data Module: `nemo-vfm/nemo/collections/diffusion/datamodule.py::ActionControlDiffusionDataset`
  - Action control dataset and data loading.

For more information on NeMo2 recipes and NeMo Lightning settings, you can refer to: https://github.com/NVIDIA/NeMo.

#### Multi-Node Training

For multi-node training, an example `sbatch <SCRIPT_NAME>.sbatch` script to fine-tune on 8 nodes with 8 GPU's per node is provided below:

```bash
#!/bin/bash
#SBATCH -A coreai_dlalgo_llm
#SBATCH -p batch
#SBATCH -N 8
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-node 8
#SBATCH --gres=gpu:8
#SBATCH --time 04:00:00
#SBATCH --mail-type=FAIL
#SBATCH --exclusive                             # exclusive node access
#SBATCH --output=/path/to/slurm/job/logs/slurm_%x._%j.out
#SBATCH -J coreai_dlalgo_llm.cosmos-action-control-ar-post-training-20250225-8node

# Mount our code into the base image.
export IMAGE=</path/to/enroot/image/cosmos-action-control-nemo-25.02rc3.sqsh>
COSMOS_CODE_DIR=</path/to/Cosmos>
HF_CACHE_DIR=</path/to/hf/cache/dir>

export WANDB_API_KEY=<WANDB_KEY>
export HF_TOKEN=<HF_TOKEN>
export MOUNT=$COSMOS_CODE_DIR:/opt/Cosmos,$HF_CACHE_DIR:/root/.cache/huggingface
export RUN_NAME="cosmos_action_control_diffusion_8node_globalbatch64_node_qatest4"
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=$((SLURM_NNODES * $MICRO_BATCH_SIZE))
export NUM_DEVICES=8

export nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
export nodes_array=($nodes)
export head_node=${nodes_array[0]}
export head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

read -r -d '' CMD <<EOF
cd /opt/Cosmos
export HF_HOME=/root/.cache/huggingface
export HF_TOKEN=$HF_TOKEN
export PYTHONPATH=/workspace/Cosmos:]\$PYTHONPATH
export WANDB_API_KEY=$WANDB_API_KEY
export WANDB_PROJECT="cosmos-nemo-diffusion-action-control-gtc-j"
export WANDB_RUN_ID=$RUN_NAME
export WANDB_RUN_NAME=cosmos-7b-diffusion-action-run
export NUM_NODES=8
export NVTE_FUSED_ATTN=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GLOBAL_BATCH_SIZE=64
export TENSOR_MODEL_PARALLEL_SIZE=8
export NUM_LAYERS=28
export EXP_DIR=nemo_experiments/$RUN_NAME
python cosmos1/models/diffusion/nemo/post_training/video2world.py \
--yes \
--factory cosmos_diffusion_7b_video2world_action_ctrl_finetune
EOF

echo "$CMD"

srun \
  --mpi=pmix \
  --container-image=${IMAGE} \
  --container-mounts=${MOUNT} \
  bash -c "$CMD"
```


## Multicamera Post-Training

1. Install necessary dependencies
   ```bash
   pip install lru-dict pyquaternion git+https://github.com/NVIDIA/NeMo-Run.git@f07877f80dbfc91c4d37683864cf2552b96d62f1
   pip install -U moviepy==1.0.3
   ```

2. Modify Environment Variables
   ```bash
   export PYTHONPATH=$PYTHONPATH:/opt/NeMo/nemo/collections/physicalai/datasets/dataverse
   export NVTE_FUSED_ATTN=0
   export CUDA_DEVICE_MAX_CONNECTIONS=1
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   export HF_TOKEN={'Huggingface Token'}
   export HF_HOME={'Path to Huggingface Cache'}
   export PYTHONPATH=$PYTHONPATH:{'Path to Cosmos'}
   export WANDB_API_KEY={'api key'}
   export WANDB_PROJECT={'project name'}
   export XDG_CACHE_HOME=cosmos1/models/diffusion/nemo/post_training/multicamera
   ```

3. Run Multicamera Script (overfit dataset)
   ```bash
   torchrun --nproc_per_node=8 cosmos1/models/diffusion/nemo/post_training/multicamera.py  --factory cosmos_multicamera_diffusion_7b_text2world_overfit   --yes   trainer.max_steps=1000     optim.config.lr=1e-6
   ```

4. (Optional) Run Largescale Full Post-Training  
   You will need to prepare you own data similar with the overfit dataset in step 3. Change XDG_CACHE_HOME to point to the custom dataset
   ```bash
   torchrun --nproc_per_node=8 cosmos1/models/diffusion/nemo/post_training/multicamera.py --yes   trainer.max_steps=1000     optim.config.lr=1e-6
   ```

   ```
   # below are examples of different post-training recipes

   # single layer debug, add this to the command above
   # model.config.num_layers=1 resume.restore_config=None

   # with trajectory control
   # --factory cosmos_multicamera_diffusion_7b_text2world_finetune_w_traj

   # image2world
   # --factory cosmos_multicamera_diffusion_7b_image2world_finetune

   # image2world with trajectory control
   # --factory cosmos_multicamera_diffusion_7b_image2world_finetune_w_traj
   ```

For different post-training recipes make sure to change the autoresume directory so that checkpoints generated from recipe A are not used for recipe B. Modify 'recipe.trainer.callbacks.dirpath' and 'recipe.resume.resume_from_directory'.  

'recipe.trainer.val_check_interval' and 'recipe.trainer.limit_val_batches' can be set to 0 to disable validation during post-training.  

Under `HF_HOME`, there should be a `hub` folder, which contains the base checkpoints. Under `multicamera` folder, there should be view embeddings, `mkdir -p $HF_HOME/multicamera && symlink -s cosmos1/models/diffusion/nemo/post_training/multicamera/*.pt $HF_HOME/multicamera`
```
hub/
├── models--nvidia--Cosmos-1.0-Diffusion-7B-Text2World/
├── models--nvidia--Cosmos-1.0-Diffusion-7B-Video2World/
└── models--nvidia--Cosmos-1.0-Tokenizer-CV8x8x8/

multicamera/
├── video_camera_embeddings_v0_camera_front_tele_30fov.pt
├── video_camera_embeddings_v0_camera_rear_tele_30fov.pt
├── video_camera_embeddings_v0_camera_front_wide_120fov.pt
├── video_neg_prompt_embeddings_v0.pt
├── video_camera_embeddings_v0_camera_cross_left_120fov.pt
├── video_camera_embeddings_v0_camera_cross_right_120fov.pt
├── video_camera_embeddings_v0_camera_rear_left_70fov.pt
└── video_camera_embeddings_v0_camera_rear_right_70fov.pt
```

## DiT Video2World Camera-Control Post-Training


### Data-preprocessing

To preprocess and tokenize video data for diffusion-based camera control, pleae refer to the documentation in [camera_control/README.md](./camera_control/README.md). These instructions will guide users on how to download the DL3DV dataset and prepare samples for camera-control post-training.


### Running Camera-Control Post-Training

To run camera-control post-training, users simply need to point the `video2world.py` post-training script to their prepared dataset, and select the `cosmos_diffusion_7b_cameractrl_finetune` factory function. Here is an example run command:

```bash
export NVTE_FUSED_ATTN=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 7)

torchrun --nproc_per_node=8 \
	cosmos1/models/diffusion/nemo/post_training/video2world.py \
	--yes \
	--factory cosmos_diffusion_7b_cameractrl_finetune \
	data.path=$CACHED_DATA
```
where `$CACHED_DATA` points to the prepared camera-control dataset as instructed in the [camera control dataset preparation instructions](./camera_control/README.md).
