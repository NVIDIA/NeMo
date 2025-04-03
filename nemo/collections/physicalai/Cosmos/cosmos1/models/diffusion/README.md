# Cosmos Diffusion-based World Foundation Models

## Table of Contents
- [Getting Started](#getting-started)
  - [Set Up Docker Environment](#set-up-docker-environment)
  - [Download Checkpoints](#download-checkpoints)
- [Usage](#usage)
  - [Model Types](#model-types)
  - [Single and Batch Generation](#single-and-batch-generation)
  - [Sample Commands](#sample-commands)
    - [Text2World](#text2world-text2worldpy-7b-and-14b)
    - [Video2World](#video2world-video2worldpy-7b-and-14b)
  - [Arguments](#arguments)
    - [Common Parameters](#common-parameters)
    - [Text2World Specific Parameters](#text2world-specific-parameters)
    - [Video2World Specific Parameters](#video2world-specific-parameters)
  - [Safety Features](#safety-features)
  - [Prompting Instructions](#prompting-instructions)

This page details the steps for using the Cosmos diffusion-based world foundation models.

## Getting Started

### Set Up Docker Environment

Follow our [Installation Guide](../../../INSTALL.md) to set up the Docker environment. All commands on this page should be run inside Docker.

### Download Checkpoints

1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token. Set the access token to 'Read' permission (default is 'Fine-grained').

2. Log in to Hugging Face with the access token:

```bash
huggingface-cli login
```

3. Request access to Mistral AI's Pixtral-12B model by clicking on `Agree and access repository` on [Pixtral's Hugging Face model page](https://huggingface.co/mistralai/Pixtral-12B-2409). This step is required to use Pixtral 12B for the Video2World prompt upsampling task.

4. Download the Cosmos model weights from [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-6751e884dc10e013a0a0d8e6):

```bash
PYTHONPATH=$(pwd) python cosmos1/scripts/download_diffusion.py --model_sizes 7B 14B --model_types Text2World Video2World
```

5. The downloaded files should be in the following structure:

```
checkpoints/
├── Cosmos-1.0-Diffusion-7B-Text2World
│   ├── model.pt
│   └── config.json
├── Cosmos-1.0-Diffusion-14B-Text2World
│   ├── model.pt
│   └── config.json
├── Cosmos-1.0-Diffusion-7B-Video2World
│   ├── model.pt
│   └── config.json
├── Cosmos-1.0-Diffusion-14B-Video2World
│   ├── model.pt
│   └── config.json
├── Cosmos-1.0-Tokenizer-CV8x8x8
│   ├── decoder.jit
│   ├── encoder.jit
│   └── mean_std.pt
├── Cosmos-1.0-Prompt-Upsampler-12B-Text2World
│   ├── model.pt
│   └── config.json
├── Pixtral-12B
│   ├── model.pt
│   ├── config.json
└── Cosmos-1.0-Guardrail
    ├── aegis/
    ├── blocklist/
    ├── face_blur_filter/
    └── video_content_safety_filter/
```

## Usage

### Model Types

There are two model types available for diffusion world generation:

1. **Text2World**: Supports world generation from text input

* Models: `Cosmos-1.0-Diffusion-7B-Text2World` and `Cosmos-1.0-Diffusion-14B-Text2World`
* Inference script: [text2world.py](/cosmos1/models/diffusion/inference/text2world.py)

2. **Video2World**: Supports world generation from text and image/video input

* Models: `Cosmos-1.0-Diffusion-7B-Video2World` and `Cosmos-1.0-Diffusion-14B-Video2World`
* Inference script: [video2world.py](/cosmos1/models/diffusion/inference/video2world.py)

### Single and Batch Generation

We support both single and batch video generation.

For generating a single video, `Text2World` mode requires the input argument `--prompt` (text input). `Video2World` mode requires `--input_image_or_video_path` (image/video input). Additionally for Video2World, if the prompt upsampler is disabled, a text prompt must also be provided using the `--prompt` argument.

For generating a batch of videos, both `Text2World` and `Video2World` require `--batch_input_path` (path to a JSONL file). For `Text2World`, the JSONL file should contain one prompt per line in the following format, where each line must contain a "prompt" field:

```json
{"prompt": "prompt1"}
{"prompt": "prompt2"}
```

For `Video2World`, each line in the JSONL file must contain a "visual_input" field:

```json
{"visual_input": "path/to/video1.mp4"}
{"visual_input": "path/to/video2.mp4"}
```

If you disable the prompt upsampler by setting the `--disable_prompt_upsampler` flag, each line in the JSONL file will need to include both "prompt" and "visual_input" fields.

```json
{"prompt": "prompt1", "visual_input": "path/to/video1.mp4"}
{"prompt": "prompt2", "visual_input": "path/to/video2.mp4"}
```

### Sample Commands

There are two main demo scripts for diffusion world generation: `text2world.py` and `video2world.py`. Below you will find sample commands for single and batch generation, as well as commands for running with low-memory GPUs using model offloading. We also provide a memory usage table comparing different offloading strategies to help with configuration.

#### Text2World (text2world.py): 7B and 14B

Generates world from text input.

##### Single Generation

```bash
PROMPT="A sleek, humanoid robot stands in a vast warehouse filled with neatly stacked cardboard boxes on industrial shelves. \
The robot's metallic body gleams under the bright, even lighting, highlighting its futuristic design and intricate joints. \
A glowing blue light emanates from its chest, adding a touch of advanced technology. The background is dominated by rows of boxes, \
suggesting a highly organized storage system. The floor is lined with wooden pallets, enhancing the industrial setting. \
The camera remains static, capturing the robot's poised stance amidst the orderly environment, with a shallow depth of \
field that keeps the focus on the robot while subtly blurring the background for a cinematic effect."

# Example using 7B model
PYTHONPATH=$(pwd) python cosmos1/models/diffusion/inference/text2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-1.0-Diffusion-7B-Text2World \
    --prompt "$PROMPT" \
    --offload_prompt_upsampler \
    --video_save_name Cosmos-1.0-Diffusion-7B-Text2World

# Example using the 7B model on low-memory GPUs with model offloading. The speed is slower if using batch generation.
PYTHONPATH=$(pwd) python cosmos1/models/diffusion/inference/text2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-1.0-Diffusion-7B-Text2World \
    --prompt "$PROMPT" \
    --video_save_name Cosmos-1.0-Diffusion-7B-Text2World_memory_efficient \
    --offload_tokenizer \
    --offload_diffusion_transformer \
    --offload_text_encoder_model \
    --offload_prompt_upsampler \
    --offload_guardrail_models

# Example using 14B model with prompt upsampler offloading (required on H100)
PYTHONPATH=$(pwd) python cosmos1/models/diffusion/inference/text2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-1.0-Diffusion-14B-Text2World \
    --prompt "$PROMPT" \
    --video_save_name Cosmos-1.0-Diffusion-14B-Text2World \
    --offload_prompt_upsampler \
    --offload_guardrail_models
```

##### Batch Generation

```bash
# Example using 7B model
PYTHONPATH=$(pwd) python cosmos1/models/diffusion/inference/text2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-1.0-Diffusion-7B-Text2World \
    --batch_input_path cosmos1/models/diffusion/assets/v1p0/batch_inputs/text2world.jsonl \
    --video_save_folder outputs/Cosmos-1.0-Diffusion-7B-Text2World \
    --offload_prompt_upsampler
```

##### Example Output

Here is an example output video generated using text2world.py, using `Cosmos-1.0-Diffusion-7B-Text2World`:

<video src="https://github.com/user-attachments/assets/db7bebfe-5314-40a6-b045-4f6ce0a87f2a">
  Your browser does not support the video tag.
</video>

The upsampled prompt used to generate the video is:

```
In a sprawling, meticulously organized warehouse, a sleek humanoid robot stands sentinel amidst towering shelves brimming with neatly stacked cardboard boxes. The robot's metallic body, adorned with intricate joints and a glowing blue chest light, radiates an aura of advanced technology, its design a harmonious blend of functionality and futuristic elegance. The camera captures this striking figure in a static, wide shot, emphasizing its poised stance against the backdrop of industrial wooden pallets. The lighting is bright and even, casting a warm glow that accentuates the robot's form, while the shallow depth of field subtly blurs the rows of boxes, creating a cinematic depth that draws the viewer into this high-tech realm. The absence of human presence amplifies the robot's solitary vigil, inviting contemplation of its purpose within this vast, organized expanse.
```

If you disable the prompt upsampler by using the `--disable_prompt_upsampler` flag, the output video will be generated using the original prompt:

<video src="https://github.com/user-attachments/assets/b373c692-9900-4e73-80c2-4016caa47a82">
  Your browser does not support the video tag.
</video>

The original prompt is:
```
A sleek, humanoid robot stands in a vast warehouse filled with neatly stacked cardboard boxes on industrial shelves. The robot's metallic body gleams under the bright, even lighting, highlighting its futuristic design and intricate joints. A glowing blue light emanates from its chest, adding a touch of advanced technology. The background is dominated by rows of boxes, suggesting a highly organized storage system. The floor is lined with wooden pallets, enhancing the industrial setting. The camera remains static, capturing the robot's poised stance amidst the orderly environment, with a shallow depth of field that keeps the focus on the robot while subtly blurring the background for a cinematic effect.
```

Note that the robot face could be blurred sometimes by the guardrail in this example.

##### Inference Time and GPU Memory Usage

The numbers provided below may vary depending on system specs and are for reference only.

We report the maximum observed GPU memory usage during end-to-end inference. Additionally, we offer a series of model offloading strategies to help users manage GPU memory usage effectively.

For GPUs with limited memory (e.g., RTX 3090/4090 with 24 GB memory), we recommend fully offloading all models. For higher-end GPUs, users can select the most suitable offloading strategy considering the numbers provided below.

| Offloading Strategy | 7B Text2World | 14B Text2World |
|-------------|---------|---------|
| Offload prompt upsampler | 74.0 GB | > 80.0 GB |
| Offload prompt upsampler & guardrails | 57.1 GB | 70.5 GB |
| Offload prompt upsampler & guardrails & T5 encoder | 38.5 GB | 51.9 GB |
| Offload prompt upsampler & guardrails & T5 encoder & tokenizer | 38.3 GB | 51.7 GB |
| Offload prompt upsampler & guardrails & T5 encoder & tokenizer & diffusion model | 24.4 GB | 39.0 GB |

The table below presents the end-to-end inference runtime on a single H100 GPU, excluding model initialization time.

| 7B Text2World (offload prompt upsampler) | 14B Text2World (offload prompt upsampler, guardrails) |
|---------|---------|
| ~380 seconds | ~590 seconds |

#### Video2World (video2world.py): 7B and 14B

Generates world from text and image/video input.

##### Single Generation

Note that our prompt upsampler is enabled by default for Video2World, and it will generate the prompt from the input image/video. If the prompt upsampler is disabled, you can provide a prompt manually using the `--prompt` flag.

```bash
# Example using the 7B model
PYTHONPATH=$(pwd) python cosmos1/models/diffusion/inference/video2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-1.0-Diffusion-7B-Video2World \
    --input_image_or_video_path cosmos1/models/diffusion/assets/v1p0/video2world_input0.jpg \
    --num_input_frames 1 \
    --video_save_name Cosmos-1.0-Diffusion-7B-Video2World \
    --offload_prompt_upsampler

# Example using the 7B model on low-memory GPUs with model offloading. The speed is slower if using batch generation.
PYTHONPATH=$(pwd) python cosmos1/models/diffusion/inference/video2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-1.0-Diffusion-7B-Video2World \
    --input_image_or_video_path cosmos1/models/diffusion/assets/v1p0/video2world_input0.jpg \
    --num_input_frames 1 \
    --video_save_name Cosmos-1.0-Diffusion-7B-Video2World_memory_efficient \
    --offload_tokenizer \
    --offload_diffusion_transformer \
    --offload_text_encoder_model \
    --offload_prompt_upsampler \
    --offload_guardrail_models

# Example using 14B model with prompt upsampler offloading (required on H100)
PYTHONPATH=$(pwd) python cosmos1/models/diffusion/inference/video2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-1.0-Diffusion-14B-Video2World \
    --input_image_or_video_path cosmos1/models/diffusion/assets/v1p0/video2world_input0.jpg \
    --num_input_frames 1 \
    --video_save_name Cosmos-1.0-Diffusion-14B-Video2World \
    --offload_prompt_upsampler \
    --offload_guardrail_models
```

##### Batch Generation

```bash
# Example using 7B model with 9 input frames
PYTHONPATH=$(pwd) python cosmos1/models/diffusion/inference/video2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-1.0-Diffusion-7B-Video2World \
    --batch_input_path cosmos1/models/diffusion/assets/v1p0/batch_inputs/video2world_ps.jsonl \
    --video_save_folder outputs/Cosmos-1.0-Diffusion-7B-Video2World \
    --offload_prompt_upsampler \
    --num_input_frames 9

# Example using 7B model with 9 input frames without prompt upsampler, using 'prompt' field in the JSONL file
PYTHONPATH=$(pwd) python cosmos1/models/diffusion/inference/video2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-1.0-Diffusion-7B-Video2World \
    --batch_input_path cosmos1/models/diffusion/assets/v1p0/batch_inputs/video2world_wo_ps.jsonl \
    --video_save_folder outputs/Cosmos-1.0-Diffusion-7B-Video2World_wo_ps \
    --disable_prompt_upsampler \
    --num_input_frames 9
```

##### Example Output

Here is an example output video generated using video2world.py, using `Cosmos-1.0-Diffusion-14B-Video2World`:

<video src="https://github.com/user-attachments/assets/a840a338-5090-4f50-9790-42b7ede86ba6">
  Your browser does not support the video tag.
</video>

The upsampled prompt (generated by the prompt upsampler) used to generate the video is:

```
The video depicts a long, straight highway stretching into the distance, flanked by metal guardrails. The road is divided into multiple lanes, with a few vehicles visible in the far distance. The surrounding landscape features dry, grassy fields on one side and rolling hills on the other. The sky is mostly clear with a few scattered clouds, suggesting a bright, sunny day.
```

##### Inference Time and GPU Memory Usage

The numbers provided below may vary depending on system specs and are for reference only.

| Offloading Strategy                                                              | 7B Video2World | 14B Video2World |
|----------------------------------------------------------------------------------|---------|---------|
| Offload prompt upsampler                                                         | 76.5 GB | > 80.0 GB |
| Offload prompt upsampler & guardrails                                            | 59.9 GB | 73.3 GB |
| Offload prompt upsampler & guardrails & T5 encoder                               | 41.3 GB | 54.8 GB |
| Offload prompt upsampler & guardrails & T5 encoder & tokenizer                   | 41.1 GB | 54.5 GB |
| Offload prompt upsampler & guardrails & T5 encoder & tokenizer & diffusion model | 27.3 GB | 39.0 GB |

The following table shows the end-to-end inference runtime on a single H100 GPU, excluding model initialization time:

| 7B Video2World (offload prompt upsampler) | 14B Video2World (offload prompt upsampler, guardrails) |
|---------|---------|
| ~383 seconds | ~593 seconds |

### Arguments

#### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--checkpoint_dir` | Directory containing model weights | "checkpoints" |
| `--tokenizer_dir` | Directory containing tokenizer weights | "Cosmos-1.0-Tokenizer-CV8x8x8" |
| `--video_save_name` | Output video filename for single video generation | "output" |
| `--video_save_folder` | Output directory for batch video generation | "outputs/" |
| `--prompt` | Text prompt for single video generation. Required for single video generation. | None |
| `--batch_input_path` | Path to JSONL file for batch video generation. Required for batch video generation. | None |
| `--negative_prompt` | Negative prompt for improved quality | "The video captures a series of frames showing ugly scenes..." |
| `--num_steps` | Number of diffusion sampling steps | 35 |
| `--guidance` | CFG guidance scale | 7.0 |
| `--num_video_frames` | Number of frames to generate | 121 |
| `--height` | Output video height | 704 |
| `--width` | Output video width | 1280 |
| `--fps` | Frames per second | 24 |
| `--seed` | Random seed | 1 |
| `--disable_prompt_upsampler` | Disable automatic prompt enhancement | False |
| `--offload_diffusion_transformer` | Offload DiT model after inference, used for low-memory GPUs | False |
| `--offload_tokenizer` | Offload VAE model after inference, used for low-memory GPUs | False |
| `--offload_text_encoder_model` | Offload text encoder after inference, used for low-memory GPUs | False |
| `--offload_prompt_upsampler` | Offload prompt upsampler after inference, used for low-memory GPUs | False |
| `--offload_guardrail_models` | Offload guardrail models after inference, used for low-memory GPUs | False |

Note: we support various aspect ratios, including 1:1 (960x960 for height and width), 4:3 (960x704), 3:4 (704x960), 16:9 (1280x704), and 9:16 (704x1280). The frame rate is also adjustable within a range of 12 to 40 fps. The current version of the model only supports 121 frames.

#### Text2World Specific Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--diffusion_transformer_dir` | Directory containing DiT weights | "Cosmos-1.0-Diffusion-7B-Text2World" |
| `--prompt_upsampler_dir` | Directory containing prompt upsampler weights | "Cosmos-1.0-Prompt-Upsampler-12B-Text2World" |
| `--word_limit_to_skip_upsampler` | Skip prompt upsampler for better robustness if the number of words in the prompt is greater than this value | 250 |
#### Video2World Specific Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--diffusion_transformer_dir` | Directory containing DiT weights | "Cosmos-1.0-Diffusion-7B-Video2World" |
| `--prompt_upsampler_dir` | Directory containing prompt upsampler weights | "Pixtral-12B" |
| `--input_image_or_video_path` | Input video/image path for single video generation. Required for single video generation. | None |
| `--num_input_frames` | Number of video frames (1 or 9) | 1 |

### Safety Features

The model uses a built-in safety guardrail system that cannot be disabled. Generating human faces is not allowed and will be blurred by the guardrail.

For more information, check out the [Cosmos Guardrail Documentation](../guardrail/README.md).

### Prompting Instructions

The input prompt is the most important parameter under the user's control when interacting with the model. Providing rich and descriptive prompts can positively impact the output quality of the model, whereas short and poorly detailed prompts can lead to subpar video generation. Here are some recommendations to keep in mind when crafting text prompts for the model:

1. **Describe a single, captivating scene**: Focus on a single scene to prevent the model from generating videos with unnecessary shot changes.
2. **Limit camera control instructions**: The model doesn't handle prompts involving camera control well, as this feature is still under development.
3. **Prompt upsampler limitations**: The current version of the prompt upsampler may sometimes deviate from the original intent of your prompt, adding unwanted details. If this happens, you can disable the upsampler with the --disable_prompt_upsampler flag and edit your prompt manually. We recommend using prompts of around 120 words for optimal quality.

#### Cosmos-1.0-Prompt-Upsampler

The prompt upsampler automatically expands brief prompts into more detailed descriptions (Text2World) or generates detailed prompts based on input images (Video2World).

##### Text2World

When enabled (default), the upsampler will:

1. Take your input prompt
2. Process it through a finetuned Mistral model to generate a more detailed description
3. Use the expanded description for video generation

This can help generate better quality videos by providing more detailed context to the video generation model. To disable this feature, use the `--disable_prompt_upsampler` flag.

##### Video2World

When enabled (default), the upsampler will:

1. Take your input image or video
2. Process it through a Pixtral model to generate a detailed description
3. Use the generated description for video generation

Please note that the Video2World prompt upsampler does not consider any user-provided text prompt. To disable this feature, use the `--disable_prompt_upsampler` flag.
