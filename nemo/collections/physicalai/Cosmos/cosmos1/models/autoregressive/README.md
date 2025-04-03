# Cosmos Autoregressive-based World Foundation Models

## Table of Contents
- [Getting Started](#getting-started)
  - [Set Up Docker Environment](#set-up-docker-environment)
  - [Download Checkpoints](#download-checkpoints)
- [Usage](#usage)
  - [Model Types](#model-types)
  - [Single and Batch Generation](#single-and-batch-generation)
  - [Sample Commands](#sample-commands)
    - [Base Models (4B/12B)](#base-basepy-4b-and-12b)
    - [Video2World Models (5B/13B)](#video2world-video2worldpy-5b-and-13b)
  - [Arguments](#arguments)
    - [Common Parameters](#common-parameters)
    - [Base Specific Parameters](#base-specific-parameters)
    - [Video2World Specific Parameters](#video2world-specific-parameters)
  - [Safety Features](#safety-features)

This page details the steps for using the Cosmos autoregressive-based world foundation models.

## Getting Started

### Set Up Docker Environment

Follow our [Installation Guide](../../../INSTALL.md) to set up the Docker environment. All commands on this page should be run inside Docker.

### Download Checkpoints

1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token. Set the access token to 'Read' permission (default is 'Fine-grained').

2. Log in to Hugging Face with the access token:

```bash
huggingface-cli login
```

3. Download the Cosmos model weights from [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-6751e884dc10e013a0a0d8e6):

```bash
PYTHONPATH=$(pwd) python cosmos1/scripts/download_autoregressive.py --model_sizes 4B 5B 12B 13B
```

4. The downloaded files should be in the following structure:

```
checkpoints/
├── Cosmos-1.0-Autoregressive-4B
│   ├── model.pt
│   └── config.json
├── Cosmos-1.0-Autoregressive-5B-Video2World
│   ├── model.pt
│   └── config.json
├── Cosmos-1.0-Autoregressive-12B
│   ├── model.pt
│   └── config.json
├── Cosmos-1.0-Autoregressive-13B-Video2World
│   ├── model.pt
│   └── config.json
├── Cosmos-1.0-Tokenizer-CV8x8x8
│   ├── decoder.jit
│   ├── encoder.jit
│   └── mean_std.pt
├── Cosmos-1.0-Tokenizer-DV8x16x16
│   ├── decoder.jit
│   └── encoder.jit
├── Cosmos-1.0-Diffusion-7B-Decoder-DV8x16x16ToCV8x8x8
│   ├── aux_vars.pt
│   └── model.pt
└── Cosmos-1.0-Guardrail
    ├── aegis/
    ├── blocklist/
    ├── face_blur_filter/
    └── video_content_safety_filter/
```

## Usage


### Model Types

There are two model types available for autoregressive world generation:

1. **Base**: Supports world generation from image/video input

* Models: `Cosmos-1.0-Autoregressive-4B` and `Cosmos-1.0-Autoregressive-12B`
* Inference script: [base.py](/cosmos1/models/autoregressive/inference/base.py)

2. **Video2World**: Supports world generation from image/video input and text input

* Models: `Cosmos-1.0-Autoregressive-5B-Video2World` and `Cosmos-1.0-Autoregressive-13B-Video2World`
* Inference script: [video2world.py](/cosmos1/models/autoregressive/inference/video2world.py)

Our models now support video extension up to 33 frames. Starting from either a single image or a 9-frame video input, they can generate the remaining frames to reach the 33-frame length (generating 32 or 24 frames, respectively).

We have evaluated all eight possible configurations (4 models × 2 vision input types: image or video) using 100 test videos on physical AI topics. Below are the failure rates for each configuration:

| Model                                      | Image input | Video input (9 frames) |
|:------------------------------------------|:--------------:|:-------------------------:|
| Cosmos-1.0-Autoregressive-4B              | 15%           | 1%                       |
| Cosmos-1.0-Autoregressive-5B-Video2World  | 7%            | 2%                       |
| Cosmos-1.0-Autoregressive-12B             | 2%            | 1%                       |
| Cosmos-1.0-Autoregressive-13B-Video2World | 3%            | 0%                       |

We define failure cases as videos with severe distortions, such as:

* Sudden appearance of large unexpected objects
* Video degrading to a single solid color

Note that the following are not considered failures in our analysis:

* Static video frames
* Minor object distortions or artifacts

### Single and Batch Generation

We support both single and batch video generation.

For generating a single video, `base` mode requires the input argument `--input_image_or_video_path` (image/video input), while `video2world` mode requires both `--input_image_or_video_path` (image/video input) and `--prompt` (text input).

Note that our model only works with 1024x640 resolution videos. If the input image/video is not in this resolution, it will be resized and cropped.

For generating a batch of videos, both `base` and `video2world` require `--batch_input_path` (path to a JSONL file). For `base`, the JSONL file should contain one visual input per line in the following format, where each line must contain a "visual_input" field:

```json
{"visual_input": "path/to/video1.mp4"}
{"visual_input": "path/to/video2.mp4"}
```

For `video2world`, each line in the JSONL file must contain both "prompt" and "visual_input" fields:

```json
{"prompt": "prompt1", "visual_input": "path/to/video1.mp4"}
{"prompt": "prompt2", "visual_input": "path/to/video2.mp4"}
```

### Sample Commands

There are two main demo scripts for autoregressive world generation: `base.py` and `video2world.py`. Below you will find sample commands for single and batch generation, as well as commands for running with low-memory GPUs using model offloading. We also provide a memory usage table comparing different offloading strategies to help with configuration.

#### Base (base.py): 4B and 12B

Generates world from image/video input.

The `input_type` argument can be either `video` or `image`. We have tuned the sampling parameters `top_p` and `temperature` to achieve the best performance. Please use the provided values in the command examples.

Note that the command examples below all use video input. If you want to use image input, please change the `input_type` to `image`.

##### Single Generation

```bash
# Example using 4B model
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python cosmos1/models/autoregressive/inference/base.py \
    --input_type=video \
    --input_image_or_video_path=cosmos1/models/autoregressive/assets/v1p0/input.mp4 \
    --video_save_name=Cosmos-1.0-Autoregressive-4B \
    --ar_model_dir=Cosmos-1.0-Autoregressive-4B \
    --top_p=0.8 \
    --temperature=1.0

# Example for low-memory GPUs using 4B model with model offloading
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python cosmos1/models/autoregressive/inference/base.py \
    --input_type=video \
    --input_image_or_video_path=cosmos1/models/autoregressive/assets/v1p0/input.mp4 \
    --video_save_name=Cosmos-1.0-Autoregressive-4B \
    --ar_model_dir=Cosmos-1.0-Autoregressive-4B \
    --top_p=0.8 \
    --temperature=1.0 \
    --offload_guardrail_models \
    --offload_diffusion_decoder \
    --offload_ar_model \
    --offload_tokenizer

# Example using 12B model
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python cosmos1/models/autoregressive/inference/base.py \
    --input_type=video \
    --input_image_or_video_path=cosmos1/models/autoregressive/assets/v1p0/input.mp4 \
    --video_save_name=Cosmos-1.0-Autoregressive-12B \
    --ar_model_dir=Cosmos-1.0-Autoregressive-12B \
    --top_p=0.9 \
    --temperature=1.0

# Example for low-memory GPUs using 12B model with model offloading
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python cosmos1/models/autoregressive/inference/base.py \
    --input_type=video \
    --input_image_or_video_path=cosmos1/models/autoregressive/assets/v1p0/input.mp4 \
    --video_save_name=Cosmos-1.0-Autoregressive-12B \
    --ar_model_dir=Cosmos-1.0-Autoregressive-12B \
    --top_p=0.9 \
    --temperature=1.0 \
    --offload_guardrail_models \
    --offload_diffusion_decoder \
    --offload_ar_model \
    --offload_tokenizer
```

##### Batch Generation

```bash
# Example using 4B model
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python cosmos1/models/autoregressive/inference/base.py \
    --input_type=video \
    --batch_input_path=cosmos1/models/autoregressive/assets/v1p0/batch_inputs/base.jsonl \
    --video_save_folder=outputs/Cosmos-1.0-Autoregressive-4B \
    --ar_model_dir=Cosmos-1.0-Autoregressive-4B \
    --top_p=0.8 \
    --temperature=1.0

# Example using 12B model
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python cosmos1/models/autoregressive/inference/base.py \
    --input_type=video \
    --batch_input_path=cosmos1/models/autoregressive/assets/v1p0/batch_inputs/base.jsonl \
    --video_save_folder=outputs/Cosmos-1.0-Autoregressive-12B \
    --ar_model_dir=Cosmos-1.0-Autoregressive-12B \
    --top_p=0.9 \
    --temperature=1.0
```

##### Example Output

Here is an example output video generated using base.py with image input, using `Cosmos-1.0-Autoregressive-12B`:

<video src="https://github.com/user-attachments/assets/634403a5-1873-42d7-8dd0-eb7fb4ac8cf4">
  Your browser does not support the video tag.
</video>

The input image used to generate this video can be found in `cosmos1/models/autoregressive/assets/v1p0/input.jpg`. The image is from [BDD dataset](http://bdd-data.berkeley.edu/).

Here is an example output video generated using base.py with 9-frame video input, using `Cosmos-1.0-Autoregressive-12B`:

<video src="https://github.com/user-attachments/assets/1a3ff099-87d7-41e8-b149-a25cfcd4f40b">
  Your browser does not support the video tag.
</video>

The input video used to generate this video can be found in `cosmos1/models/autoregressive/assets/v1p0/input.mp4`.

##### Inference Time and GPU Memory Usage

These numbers may vary based on system specifications and are provided for reference only.

| Offloading Strategy | Cosmos-1.0-Autoregressive-4B | Cosmos-1.0-Autoregressive-12B |
|-------------|---------|---------|
| No offloading | 31.3 GB | 47.5 GB |
| Guardrails | 28.9 GB | 45.2 GB |
| Guardrails & Diffusion decoder | 28.5 GB | 43.1 GB |
| Guardrails & Diffusion decoder & Tokenizer | 27.3 GB | 42.9 GB |
| Guardrails & Diffusion decoder & Tokenizer & AR model | 18.7 GB | 27.4 GB |

End-to-end inference runtime on one H100 without offloading and after model initialization:

| Cosmos-1.0-Autoregressive-4B | Cosmos-1.0-Autoregressive-12B |
|---------|---------|
| ~62 seconds | ~119 seconds |

#### Video2World (video2world.py): 5B and 13B

Generates world from image/video and text input.

The `input_type` argument can be either `text_and_video` or `text_and_image`. We have tuned the sampling parameters `top_p` and `temperature` to achieve the best performance. Please use the provided values in the command examples.

Note that the command examples below all use video input. If you want to use image input, please change the `input_type` to `text_and_image`.

##### Single Generation

```bash
# Example using 5B model
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python cosmos1/models/autoregressive/inference/video2world.py \
    --input_type=text_and_video \
    --input_image_or_video_path=cosmos1/models/autoregressive/assets/v1p0/input.mp4 \
    --prompt="A video recorded from a moving vehicle's perspective, capturing roads, buildings, landscapes, and changing weather and lighting conditions." \
    --video_save_name=Cosmos-1.0-Autoregressive-5B-Video2World \
    --ar_model_dir=Cosmos-1.0-Autoregressive-5B-Video2World \
    --top_p=0.7 \
    --temperature=1.0

# Example for low-memory GPUs using 5B model with model offloading
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python cosmos1/models/autoregressive/inference/video2world.py \
    --input_type=text_and_video \
    --input_image_or_video_path=cosmos1/models/autoregressive/assets/v1p0/input.mp4 \
    --prompt="A video recorded from a moving vehicle's perspective, capturing roads, buildings, landscapes, and changing weather and lighting conditions." \
    --video_save_name=Cosmos-1.0-Autoregressive-5B-Video2World \
    --ar_model_dir=Cosmos-1.0-Autoregressive-5B-Video2World \
    --top_p=0.7 \
    --temperature=1.0 \
    --offload_guardrail_models \
    --offload_diffusion_decoder \
    --offload_ar_model \
    --offload_tokenizer \
    --offload_text_encoder_model

# Example using 13B model
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python cosmos1/models/autoregressive/inference/video2world.py \
    --input_type=text_and_video \
    --input_image_or_video_path=cosmos1/models/autoregressive/assets/v1p0/input.mp4 \
    --prompt="A video recorded from a moving vehicle's perspective, capturing roads, buildings, landscapes, and changing weather and lighting conditions." \
    --video_save_name=Cosmos-1.0-Autoregressive-13B-Video2World \
    --ar_model_dir=Cosmos-1.0-Autoregressive-13B-Video2World \
    --top_p=0.8 \
    --temperature=1.0 \
    --offload_guardrail_models

# Example for low-memory GPUs using 13B model with model offloading
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python cosmos1/models/autoregressive/inference/video2world.py \
    --input_type=text_and_video \
    --input_image_or_video_path=cosmos1/models/autoregressive/assets/v1p0/input.mp4 \
    --prompt="A video recorded from a moving vehicle's perspective, capturing roads, buildings, landscapes, and changing weather and lighting conditions." \
    --video_save_name=Cosmos-1.0-Autoregressive-13B-Video2World \
    --ar_model_dir=Cosmos-1.0-Autoregressive-13B-Video2World \
    --top_p=0.8 \
    --temperature=1.0 \
    --offload_guardrail_models \
    --offload_diffusion_decoder \
    --offload_ar_model \
    --offload_tokenizer \
    --offload_text_encoder_model
```

##### Batch Generation

```bash
# Example using 5B model
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python cosmos1/models/autoregressive/inference/video2world.py \
    --input_type=text_and_video \
    --batch_input_path=cosmos1/models/autoregressive/assets/v1p0/batch_inputs/video2world.jsonl \
    --video_save_folder=outputs/Cosmos-1.0-Autoregressive-5B-Video2World \
    --ar_model_dir=Cosmos-1.0-Autoregressive-5B-Video2World \
    --top_p=0.7 \
    --temperature=1.0

# Example using 13B model
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python cosmos1/models/autoregressive/inference/video2world.py \
    --input_type=text_and_video \
    --batch_input_path=cosmos1/models/autoregressive/assets/v1p0/batch_inputs/video2world.jsonl \
    --video_save_folder=outputs/Cosmos-1.0-Autoregressive-13B-Video2World \
    --ar_model_dir=Cosmos-1.0-Autoregressive-13B-Video2World \
    --top_p=0.8 \
    --temperature=1.0 \
    --offload_guardrail_models
```

##### Example Output

Here is an example output video generated using video2world.py with image input, using `Cosmos-1.0-Autoregressive-13B-Video2World`:

<video src="https://github.com/user-attachments/assets/869f3b81-fabd-462e-a545-c04cdd9c1d22">
  Your browser does not support the video tag.
</video>

The input image used to generate this video can be found in `cosmos1/models/autoregressive/assets/v1p0/input.jpg`. The prompt for generating the video is:

```
A driving video captures a serene urban street scene on a sunny day. The camera is mounted on the dashboard of a moving vehicle, providing a first-person perspective as it travels down a two-lane road. The street is lined with parked cars on both sides, predominantly black and silver sedans and SUVs. The road is flanked by a mix of residential and commercial buildings, with a prominent red-brick building on the left side, featuring multiple windows and a flat roof. The sky is clear with a few scattered clouds, casting soft shadows on the street. Trees with lush green foliage line the right side of the road, providing a natural contrast to the urban environment. The camera remains steady, maintaining a consistent forward motion, suggesting a leisurely drive. Traffic is light, with a few vehicles moving in the opposite direction, including a black sedan and a yellow taxi. Street signs are visible, including a no-parking sign on the right. The overall atmosphere is calm and peaceful, with no pedestrians visible, emphasizing the focus on the drive and the surrounding urban landscape.
```

Here is an example output video generated using video2world.py with 9-frame video input, using `Cosmos-1.0-Autoregressive-13B-Video2World`:

<video src="https://github.com/user-attachments/assets/81840e1c-624b-4b01-9240-ab7db3722e58">
  Your browser does not support the video tag.
</video>

The input video used to generate this video can be found in `cosmos1/models/autoregressive/assets/v1p0/input.mp4`. The prompt for generating the video is:

```
A video recorded from a moving vehicle's perspective, capturing roads, buildings, landscapes, and changing weather and lighting conditions.
```

##### Inference Time and GPU Memory Usage

These numbers may vary based on system specifications and are provided for reference only.

| Offloading Strategy | Cosmos-1.0-Autoregressive-5B-Video2World | Cosmos-1.0-Autoregressive-13B-Video2World |
|-------------|---------|---------|
| No offloading | 66.2 GB | > 80 GB |
| Guardrails | 58.7 GB | 76.6 GB |
| Guardrails & T5 encoder | 41.3 GB | 58.0 GB |
| Guardrails & T5 encoder & Diffusion decoder | 29.0 GB | 46.9 GB |
| Guardrails & T5 encoder & Diffusion decoder & Tokenizer | 28.8 GB | 46.7 GB |
| Guardrails & T5 encoder & Diffusion decoder & Tokenizer & AR model | 21.1 GB | 30.9 GB |

End-to-end inference runtime on one H100 with no offloading for 5B model and guardrail offloading for 13B, after model initialization:

| Cosmos-1.0-Autoregressive-5B-Video2World | Cosmos-1.0-Autoregressive-13B-Video2World |
|---------|---------|
| ~73 seconds | ~150 seconds |

### Arguments

#### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--checkpoint_dir` | Directory containing model weights | "checkpoints" |
| `--video_save_name` | Output video filename for single video generation | "output" |
| `--video_save_folder` | Folder where all output videos are stored | "outputs/" |
| `--input_image_or_video_path` | Input image or video path. Required for single video generation | None |
| `--batch_input_path` | Folder containing input images or videos. Required for batch video generation | None |
| `--num_input_frames` | Number of input frames to use for Video2World prediction | 9 |
| `--temperature` | Temperature used while sampling | 1.0 (recommend using values in sample commands provided) |
| `--top_p` | Top-p value for top-p sampling | 0.8 (recommend using values in sample commands provided) |
| `--seed` | Random seed | 0 |
| `--disable_diffusion_decoder` | When set to True, use discrete tokenizer to decode discrete tokens to video. Otherwise, use diffusion decoder to decode video | False |
| `--offload_guardrail_models` | Offload guardrail models after inference, used for low-memory GPUs | False |
| `--offload_diffusion_decoder` | Offload diffusion decoder after inference, used for low-memory GPUs | False |
| `--offload_ar_model` | Offload AR model after inference, used for low-memory GPUs | False |
| `--offload_prompt_upsampler` | Offload prompt upsampler after inference, used for low-memory GPUs | False |

#### Base Specific Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--ar_model_dir` | Directory containing AR model weight | "Cosmos-1.0-Autoregressive-4B" |
| `--input_type` | Input type, either `video` or `image` | "video" |

#### Video2World Specific Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--ar_model_dir` | Directory containing AR model weight | "Cosmos-1.0-Autoregressive-4B" |
| `--input_type` | Input type, either `text_and_video` or `text_and_image` | "text_and_video" |
| `--prompt` | Text prompt for single video generation. Required for single video generation | None |
| `--input_prompts_path` | Path to JSONL file for batch video generation. Required for batch video generation | None |
| `--offload_text_encoder_model` | Offload text encoder after inference, used for low-memory GPUs | False |

### Safety Features

The model uses a built-in safety guardrail system that cannot be disabled. Generating human faces is not allowed and will be blurred by the guardrail.

For more information, check out the [Cosmos Guardrail Documentation](../guardrail/README.md).
