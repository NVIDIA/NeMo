# Cosmos Autoregressive-based World Foundation Models: NeMo Framework User Guide

Learn how to [post-train](#post-train) Cosmos Autoregressive-based World Foundation Models (WFMs) using the [NVIDIA NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html) for your custom Physical AI tasks by following this guide.

## Model Support Matrix

The NeMo Framework supports the following Cosmos Autoregressive (AR) models. Review the available models and their compute requirements for post-training and inference to determine the best model for your use case.

| Model Name        | Model Status           | Compute Requirements for Post-Training |
|-------------------------|----------------------------|-------------------------------------------|
| Cosmos-1.0-Autoregressive-4B                  | **Supported**       | 2 NVIDIA GPUs*             |
| Cosmos-1.0-Autoregressive-12B                 | **Supported**       | 8 NVIDIA GPUs*             |
| Cosmos-1.0-Autoregressive-5B-Video2World      |  **Supported**      | 2 NVIDIA GPUs*             |
| Cosmos-1.0-Autoregressive-13B-Video2World     |  **Supported**      | 8 NVIDIA GPUs*             |

**\*** `H100-80GB` or `A100-80GB` GPUs are recommended.

## Post-Training Support Matrix

Cosmos Autoregressive-based WFMs can be post-trained for a variety of Physical AI tasks. Review the following table for a list of available Physical AI post-training tasks:

| Post-training Task  | Support Status |
|-------------------------|--------------------|
| General post-training   | **Supported**      |
| Instruction control     | **Supported**    |
| Action control          | **Supported**      |
| Camera control          | **Coming Soon**    |
| Multi-view generation   | **Coming Soon**    |
| Multi-view generation with vehicle trajectory control | **Coming Soon** |
| Changing the Video Tokenizer | **Supported** |

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

The [NeMo Framework container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo) supports post-training and inference for Cosmos AR models.

Run the following command to download and start the container:

   ```bash
   docker run --ipc=host -it --gpus=all \
    -v $PATH_TO_COSMOS_REPO:/workspace/Cosmos \
    nvcr.io/nvidia/nemo:25.02.rc3 bash
   ```

### 4. Download Checkpoints

To help you get started, we've provided a [download script](../download_autoregressive_nemo.py) to get the Cosmos Autoregressive checkpoints from Hugging Face. These checkpoints are in the NeMo distributed checkpoint format required to run post-training and inference with NeMo Framework.

1. Set the following environment variables:

   ```bash
   # You must set HF_HOME before running this script.
   export HF_TOKEN="<your/HF/access/token>"
   export HF_HOME="<path/to/store/checkpoints>"
   ```

2. Run the following command to download the models:

   ```bash
   cd /workspace/Cosmos
   python cosmos1/models/autoregressive/nemo/download_autoregressive_nemo.py
   ```

## Post-train

Post-training a Cosmos Autoregressive-based WFM enables you to train the model to generate videos using frame predictions that are more specific to your Physical AI use case.

For example, if you want to generate action sequences for a specific robot, you can post-train the model to generate videos that are more aligned with typical actions/outcomes for that robot.

There are 3 steps to post-training: preparing a dataset, preprocessing the data, and post-training the model.

### 1. Prepare a Dataset

The first step is to prepare a dataset. Post-training a Cosmos-1.0-Autoregressive model enables you to get better video-frame predictions for your specific use case.

You must provide a folder containing a collection of videos in **MP4 format**, preferably 720p. In this guide, we'll use the sample videos located in the `cosmos1/models/autoregressive/assets/v1p0/batch_inputs` directory.

### 2. Preprocess Data

#### 4B and 12B Models
The second step is to preprocess the data to create an [indexed dataset](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/datasets).

The `IndexedDataset` class is the lowest-level data interface in Megatron Core and creates a `.bin` and `.idx` file.

Before proceeding, ensure all videos are in **RGB format**. Complete the following steps to preprocess the data.

1. Set the following environment variables:

   ```bash
   pip install --no-cache-dir imageio[ffmpeg] pyav iopath

   export HF_TOKEN="<your/HF/access/token>"
   export HF_HOME="<path/to/store/checkpoints>"

   # Path to Raw mp4 videos.
   export RAW_DATA="cosmos1/models/autoregressive/assets/v1p0/batch_inputs"

   # Path to Processed Dataset.
   export OUTPUT_PREFIX="./indexed_videos"
   ```

2. Run the following command to preprocess the data:

   ```bash
   cd /workspace/Cosmos
   git lfs pull --include=$RAW_DATA

   python cosmos1/models/autoregressive/nemo/post_training/prepare_dataset.py \
   --input_videos_dir $RAW_DATA \
   --output_prefix $OUTPUT_PREFIX
   ```

Executing the [data preprocessing script](./prepare_dataset.py) for the base model generates the following files for each video:

- **`[i].idx` File**: This file contains metadata at the dataset level:
  - **Index Header**: Ensures backward compatibility.
  - **Index Version**: Maintains backward compatibility.
  - **Data Type Code**: Numeric code indicating the data type used in the data file.
  - **Sequence Count**: Total number of sequences in the dataset.
  - **Document Count**: Total number of documents in the dataset.

- **`[i].bin` File**: This file includes metadata at the document and sequence levels:
  - **Elements per Sequence**: Number of elements in each sequence.
  - **Byte Offset per Sequence**: Pointer indicating the start of each sequence.
  - **Sequence Index Range**: Consecutive index range `[...)` for each document.

#### 5B and 13B Models
The second step is to preprocess the data to pre compute the text and video embeddings for finetuning..

Before proceeding, ensure all videos are in **RGB format**. Complete the following steps to preprocess the data.

1. Set the following environment variables:

   ```bash
   pip install --no-cache-dir imageio[ffmpeg] pyav iopath

   export HF_TOKEN="<your/HF/access/token>"
   export HF_HOME="<path/to/store/checkpoints>"

   # Path to Raw mp4 videos.
   export RAW_DATA="cosmos1/models/autoregressive/assets/v1p0/batch_inputs"

   # Path to Processed Dataset.
   export OUTPUT_PREFIX="./indexed_videos"
   ```

2. Run the following command to preprocess the data:

   ```bash
   cd /workspace/Cosmos
   git lfs pull --include=$RAW_DATA

   python3 cosmos1/models/autoregressive/nemo/post_training/video2world_prepare_dataset.py \
   --input_jsonl $RAW_DATA/video2world.jsonl \
   --output_dir $OUTPUT_PREFIX
   ```

Executing the [data preprocessing script](./video2world_prepare_dataset.py) for the base model generates

Executing the [data preprocessing script](./prepare_dataset.py) for the base model generates the following files for each video:

- **`[i].pt` File**: This file contains video tokens or prompt embeddings:
  - It has a format `<train/test/val>_<prompt/video>_<idx>.pt`

- **`[i]metadata.json` File**: This file includes metadata:
  - It tells you the number of train test and validation samples

### 3. Post-train the Model

The third step is to post-train the model. This step uses NeMo Framework's data and model parallelism capabilities to train the model on the post-training samples. This is accomplished by utilizing Tensor Parallelism.

- **Tensor Parallelism**: Spreads the parameter tensor of individual layers across GPUs.

#### Run the Post-training Script

##### 4B AND 12B Models

Complete the following steps to post-train the Cosmos-1.0-Autoregressive-4B model.

1. Set the following environment variables:

   ```bash
   export HF_TOKEN="<your/HF/access/token>"
   export HF_HOME="<path/to/store/checkpoints>"

   # Number of GPU devices available for post-training. At least 2 for 4B and 8 for 12B.
   export NUM_DEVICES=2

   # Optionally, you can monitor training progress with Weights and Biases (wandb).
   export WANDB_API_KEY="</your/wandb/api/key>"
   export WANDB_PROJECT_NAME="cosmos-autoregressive-nemo-finetuning"
   export WANDB_RUN_ID="cosmos_autoregressive_4b_finetune"
   ```

2. Run the following command for Cosmos-1.0-Autoregressive-4B post-training:

   ```bash
   torchrun --nproc-per-node $NUM_DEVICES cosmos1/models/autoregressive/nemo/post_training/general.py \
   --data_path $OUTPUT_PREFIX \
   --split_string 4,1,1 \
   --log_dir ./logs \
   --max_steps 10 --save_every_n_steps 5 \
   --tensor_model_parallel_size $NUM_DEVICES \
   --model_path nvidia/Cosmos-1.0-Autoregressive-4B
   ```

3. You can now run inference with your post-trained model using the instructions [here](../inference/README.md#run-the-inference-script-with-post-trained-model).

##### 5B and 13B Models

Complete the following steps to post-train the Cosmos-1.0-Autoregressive-5B model.

1. Set the following environment variables:

   ```bash
   export HF_TOKEN="<your/HF/access/token>"
   export HF_HOME="<path/to/store/checkpoints>"

   # Number of GPU devices available for post-training. At least 4 for 5B and 8 for 13B.
   export NUM_DEVICES=4

   # Optionally, you can monitor training progress with Weights and Biases (wandb).
   export WANDB_API_KEY="</your/wandb/api/key>"
   export WANDB_PROJECT_NAME="cosmos-autoregressive-nemo-finetuning"
   export WANDB_RUN_ID="cosmos_autoregressive_5b_finetune"
   ```

2. Run the following command for Cosmos-1.0-Autoregressive-5B-Video2World post-training:

   ```bash
   torchrun --nproc-per-node $NUM_DEVICES \
      cosmos1/models/autoregressive/nemo/post_training/video2world_finetuning.py \
      --data_path $OUTPUT_PREFIX \
      --log_dir ./logs \
      --max_steps 10 --save_every_n_steps 5 \
      --tensor_model_parallel_size $NUM_DEVICES \
      --model_path nvidia/Cosmos-1.0-Autoregressive-5B-Video2World
   ```

3. You can now run inference with your post-trained model using the instructions [here](../inference/README.md#run-the-inference-script-with-post-trained-model).

#### Configuration Options

Before getting started, review the following parameters that made available to the script. You can adjust these parameters to optimize performance based on your specific requirements.

| Parameter | Description | Default |
|---|---|---|
| `--data_path` | Specifies the location of your preprocessed dataset. Ensure this path points to the directory containing your `.bin` and `.idx` files. | `/path/to/data` |
| `--model_path` | Specifies the directory to the cosmos model to run post-training on. | `nvidia/Cosmos-1.0-Autoregressive-4B` |
| `--index_mapping_dir` | Specifies the directory to store the indexed dataset. | `./index_mapping` |
| `--log_dir` | Specifies the directory to store the logs and checkpoints. | `./log_dir` |
| `--split_string` | Specifies the data split ratios for training, validation, and testing. (Only valid for Base Model (4B and 12B)) | `4,1,1` |
| `--tensor_model_parallel_size` | Controls the number of GPUs used for model parallelism. Increase this number to scale up, ensuring your hardware can support the additional load. | `2` |
| `--max_steps` | Defines the total number of training steps. Adjust based on training duration and storage capacity. | `100` |
| `--save_every_n_steps` | Defines how often checkpoints are saved. Adjust based on training duration and storage capacity. | `10` |
| `--global_batch_size` | Tweaks to optimize memory usage and training speed. Larger batch sizes may improve convergence but require more memory. | `2` |
| `--micro_batch_size` | Tweaks to optimize memory usage and training speed. Larger batch sizes may improve convergence but require more memory. | `1` |
| `--lr` | Sets the learning rate. A common starting point is `5e-5`, but this can be adjusted based on model performance and convergence behavior. | `5e-5` |
| `--max_epochs` | The maximum number of epochs to run during post-training | `10` |
