# Cosmos Autoregressive-based World Foundation Models: Changing the Video Tokenizer

Learn how to post-train Cosmos Autoregressive-based World Foundation Models (WFMs) using the NVIDIA NeMo Framework when swapping out the original tokenizer for a new one. This recipe provides a conceptual overview of how to adapt your existing model to handle a different discrete video (DV) tokenizer, ensuring high-quality video generation under a new compression setting.

## Model Support Matrix

The NeMo Framework supports the following Cosmos Autoregressive (AR) models for post-training. Choose the model that best suits your compute and memory budget.

| Model Name                                | Model Status | Compute Requirements for Post-Training |
|-------------------------------------------|--------------|------------------------------------------|
| Cosmos-1.0-Autoregressive-4B              | Supported    | 2 NVIDIA GPUs*                           |
| Cosmos-1.0-Autoregressive-12B             | Supported    | 8 NVIDIA GPUs*                           |
| Cosmos-1.0-Autoregressive-5B-Video2World  | Supported    | 2 NVIDIA GPUs*                           |
| Cosmos-1.0-Autoregressive-13B-Video2World | Supported    | 8 NVIDIA GPUs*                           |

*Either H100-80GB or A100-80GB GPUs are recommended.

## Prerequisites

1. **Review General Requirements**
   - **GPU/Driver:** Ensure you have sufficient GPU memory (see table above).
   - **Containerization:** Use Docker with NVIDIA Container Runtime or a compatible environment.
   - **Hugging Face Access Token:** Required for downloading Cosmos checkpoints.
   - **(Optional) Weights & Biases:** For experiment tracking.

2. **Clone the Cosmos Repository**

   ```bash
   git clone https://github.com/NVIDIA/Cosmos.git
   ```

3. **Start the NeMo Framework Container**

   The official [NeMo Framework container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo) supports post-training and inference for Cosmos AR models. Ensure you have Docker installed along with the NVIDIA Container Runtime. Replace `$PATH_TO_COSMOS_REPO` with the path to your local clone.

   ```bash
   docker run --ipc=host -it --gpus=all \
     -v $PATH_TO_COSMOS_REPO:/workspace/Cosmos \
     nvcr.io/nvidia/nemo:25.02rc3 bash
   ```

4. **Install Python Dependencies**

   Inside the container (or your environment), install the following Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Why Change Tokenizers?

Cosmos Autoregressive models are typically trained on a specific tokenizer configuration (e.g., 8×16×16). If you’d like to reduce patch size or change compression (e.g., to 4×8×8), you can post-train the existing weights so that the model effectively aligns its internal representations with the new token embeddings—without re-training the tokenizer.

## Tutorial: Finetuning Cosmos-4B on 10k Videos with a New Tokenizer

In this tutorial, we will:
- Take a model originally trained on an 8×16×16 tokenizer.
- Post-train it on a 4×8×8 tokenizer.
- Demonstrate using a sample dataset of 10 (in production, we recommend using 10k videos from a distribution similar to that of pretraining).

### 1. Calculate Sequence Lengths

- **Original Sequence Length:** 12,800 tokens (33 frames * 640 px width * 1024 px height, when tokenized with an 8×16×16 tokenizer becomes ⌈33/8⌉ * (640/16) * (1024/16) = 5 * 64 * 80 = 12,800 tokens)
- **New Sequence Length:** 12,800 tokens (17 frames * 320 px width * 512 px height, when tokenized with a 4×8×8 tokenizer becomes ⌈17/8⌉ * (320/4) * (512/8) = 5 * 80 * 64 = 12,800 tokens)
- For other resolutions or frame counts, recalculate your maximum tokens to ensure you do not exceed the model’s capacity.

```bash
export WIDTH=512
export HEIGHT=320
export NUM_FRAMES=17
export TOKENIZER_COMPRESSION_FACTOR=4,8,8
export NUM_GPUS=8  # change this to your number of GPUs

export ENCODER_PATH="nvidia/Cosmos-0.1-Tokenizer-DV4x8x8"
export DECODER_PATH="nvidia/Cosmos-0.1-Tokenizer-DV4x8x8"
export INPUT_VIDEOS_DIR="cosmos1/models/autoregressive/assets/v1p0/batch_inputs"
export OUTPUT_TOKENS_DIR="./my_4x8x8_tokens"

mkdir -p ./experiments
```

### 2. Prepare the Dataset

Ensure that you have all the videos available in the sample dataset.

```
git lfs fetch --all
```

Run the following command to prepare the dataset.

```bash
torchrun --nproc-per-node=1 cosmos1/models/autoregressive/nemo/post_training/prepare_dataset.py \
  --width $WIDTH \
  --height $HEIGHT \
  --num_context_frames $NUM_FRAMES \
  --tokenizer_compression_factor $TOKENIZER_COMPRESSION_FACTOR \
  --encoder_path $ENCODER_PATH \
  --decoder_path $DECODER_PATH \
  --input_videos_dir $INPUT_VIDEOS_DIR \
  --output_prefix $OUTPUT_TOKENS_DIR
```

### 3. Training Recipe & Hyperparameters

This tutorial offers a baseline recipe to help you build intuition about post-training. You may tweak these parameters based on your domain, hardware, or convergence requirements. For production, consider:

- **Number of Videos:** 10,000
- **Global Batch Size:** 32
- **Learning Rate:** 1e-4
- **Recommended Steps:** ~300 steps (roughly 1 epoch for ~128M tokens)
- **Loss Convergence:** Target loss in the range of 6-7

A good rule of thumb:
- Monitor training/validation loss to ensure steady improvement.
- Evaluate sample outputs at various checkpoints to verify coherent generation.
- For more precision, consider running additional training steps.

In this tutorial, we will only post-train on 10 sample videos.

### 4. Run Post-training

Use the NeMo post-training or fine-tuning scripts (depending on your model variant) and point to your tokenized dataset. For example:

```bash
torchrun --nproc-per-node=$NUM_GPUS cosmos1/models/autoregressive/nemo/post_training/general.py \
  --data_path $OUTPUT_TOKENS_DIR \
  --split_string 10,1,1 \
  --log_dir ./experiments/example_log_dir \
  --global_batch_size 1 \
  --micro_batch_size 1 \
  --lr 1e-4 \
  --max_steps 100 --save_every_n_steps 25 \
  --max_epochs 1 \
  --tensor_model_parallel_size $NUM_GPUS \
  --model_path nvidia/Cosmos-1.0-Autoregressive-4B
```

- Adjust parameters such as `--max_steps`, `--global_batch_size`, and `--lr` to suit your needs.
- Ensure that `--model_path` matches the checkpoint originally trained on an 8×16×16 tokenizer.
- For an explanation on other configuration options, please see [the general post-training tutorial](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/autoregressive/nemo/post_training/README.md#configuration-options).

### 5. Monitor Quality

After training, monitor checkpoints and generate sample outputs. Please keep in mind that you will have to
disable the diffusion decoder, as this is not yet supported for non-standard tokenizers.

1. **Checkpoints:**  
   Checkpoints are saved under `./experiments/example_log_dir/default/checkpoints/`. For example:
   ```bash
   epoch=1-step=49
   epoch=1-step=99
   epoch=1-step=149
   ```
   Choose a checkpoint (e.g., `epoch=1-step=99`) for inference.

2. **Run Inference:**  
   Generate sample video outputs:
   ```bash
   # Set the checkpoint directory
   export CKPT_DIR="./experiments/example_log_dir/default/checkpoints/epoch=1-step=99"
   # Create an evaluation folder
   mkdir -p "$CKPT_DIR/evals"
   export INPUT_FILE="cosmos1/models/autoregressive/assets/v1p0/input.mp4"
   git lfs fetch --all
   torchrun --nproc-per-node=$NUM_GPUS \
     cosmos1/models/autoregressive/nemo/inference/general.py \
       --input_image_or_video_path $INPUT_FILE \
       --video_save_name "$CKPT_DIR/evals/generated.mp4" \
       --ar_model_dir $CKPT_DIR \
       --encoder_path $ENCODER_PATH \
       --decoder_path $DECODER_PATH \
       --disable_diffusion_decoder \
       --width $WIDTH \
       --height $HEIGHT \
       --num_context_frames $NUM_FRAMES \
       --tokenizer_compression_factor $TOKENIZER_COMPRESSION_FACTOR
   ```

3. **(Optional) Evaluate with TokenBench:**  
   To compute quantitative metrics (e.g., PSNR, SSIM), generate a large number of videos using the script above and place them into the `evals`
   folder of the checkpoint. Please ensure that the input filename in the original folder (e.g. `my-video.mp4` matches the output filename in the
   `evals` folder). This is because PSNR and SSIM compare video quality against a reference, and our script will match the output video with
   the same filename under the input folder.

   ```bash
   # Install extra dependencies for TokenBench
   pip install scikit-image imageio mediapy
   export EVAL_REFERENCE_VIDEOS_DIR="..." # <-- Replace with your eval videos input directory
   export EVAL_GENERATED_VIDEOS_DIR="$CKPT_DIR/evals"

   python cosmos1/models/autoregressive/nemo/post_training/tokenizer/token_bench.py \
     --gtpath $EVAL_REFERENCE_VIDEOS_DIR \
     --targetpath $EVAL_GENERATED_VIDEOS_DIR \
     --width $WIDTH \
     --num_frames $NUM_FRAMES \
     --recursive
   ```
   **Note:** Ensure that the filenames in the reference and generated folders match and that you use a sample size of around 1,000 videos for robust evaluation.

## Tutorial: Finetuning Cosmos-5B-Video2World with a New Tokenizer

To fine-tune a Video2World model, begin by setting up the container and environment as described above in the tutorial for the Cosmos-4B base model. Please note, for Video2World fine-tuning, the data format
is slightly different. The data format used in the fine-tuning script is a `.jsonl` file, where each line has the format

```
{"prompt": "[example prompt: a video of a robotic arm]", "visual_input": "/path/to/video.mp4"}
```

Please prepare such a file using the dataset you would like to fine-tune on. Change the environment variable `INPUT_JSONL` to the path to this file.

### 1. Environment Variables

Set common environment variables for consistency:

```bash
export WIDTH=512
export HEIGHT=320
export NUM_FRAMES=17
export TOKENIZER_COMPRESSION_FACTOR=4,8,8

export ENCODER_PATH=nvidia/Cosmos-0.1-Tokenizer-DV4x8x8
export DECODER_PATH=nvidia/Cosmos-0.1-Tokenizer-DV4x8x8

export NUM_GPUS=8 # Replace with the number of GPUs you have
export OUTPUT_TOKENS_DIR=./my_v2w_tokens # Replace with the directory to store your tokens
export INPUT_JSONL="cosmos1/models/autoregressive/assets/v1p0/batch_inputs/video2world.jsonl" # Replace with the filepath to your jsonl describing your dataset
```

### 2. Prepare the Video2World Dataset

Use the specialized script to resize videos, tokenize frames, and organize your dataset:

```bash
python cosmos1/models/autoregressive/nemo/post_training/video2world_prepare_dataset.py \
  --width $WIDTH \
  --height $HEIGHT \
  --num_context_frames $NUM_FRAMES \
  --tokenizer_compression_factor $TOKENIZER_COMPRESSION_FACTOR \
  --encoder_path $ENCODER_PATH \
  --decoder_path $DECODER_PATH \
  --output_dir $OUTPUT_TOKENS_DIR \
  --input_jsonl $INPUT_JSONL
```

### 3. Training/Finetuning for Video2World

Fine-tune a Video2World–specific checkpoint (or any compatible Cosmos AR model) on your new data. For example, using the 5B Video2World model:

```bash
torchrun --nproc-per-node=$NUM_GPUS cosmos1/models/autoregressive/nemo/post_training/video2world_finetuning.py \
  --model_path nvidia/Cosmos-1.0-Autoregressive-5B-Video2World \
  --data_path $OUTPUT_TOKENS_DIR \
  --save_every_n_steps 50 \
  --tensor_model_parallel_size $NUM_GPUS \
  --global_batch_size 1 \
  --micro_batch_size 1 \
  --lr 1e-4 \
  --max_steps 100 \
  --max_epochs 10 \
  --log_dir ./experiments/example_log_dir_v2w
```

*Tip:* Adjust `global_batch_size` and `micro_batch_size` based on your GPU capacity.

### 4. Video2World Inference

After training, run inference with the provided script to generate or reconstruct videos. Please keep in mind that you will have to
disable the diffusion decoder, as this is not yet supported for non-standard tokenizers.

```bash
export CKPT_DIR="./experiments/example_log_dir_v2w/default/checkpoints/epoch=0-step=49"
export INPUT_VIDEO="cosmos1/models/autoregressive/assets/v1p0/batch_inputs/0.mp4"
export INPUT_PROMPT="A video recorded from a moving vehicle's perspective, capturing roads, buildings, landscapes, and changing weather and lighting conditions."

git lfs pull $INPUT_VIDEO
mkdir -p "$CKPT_DIR/evals"

python cosmos1/models/autoregressive/nemo/inference/video2world.py \
  --input_image_or_video_path "$INPUT_VIDEO" \
  --prompt "$INPUT_PROMPT" \
  --width $WIDTH \
  --height $HEIGHT \
  --num_context_frames $NUM_FRAMES \
  --video_save_name "$CKPT_DIR/evals/0.mp4" \
  --tokenizer_compression_factor $TOKENIZER_COMPRESSION_FACTOR \
  --encoder_path $ENCODER_PATH \
  --decoder_path $DECODER_PATH \
  --ar_model_dir $CKPT_DIR \
  --disable_diffusion_decoder
```

*Key Parameters:*
- `--input_image_or_video_path`: Source video for context.
- `--prompt`: Textual guidance for video generation or reconstruction.
- `--encoder_path`/`--decoder_path`: Must match the tokenizer settings used during training.

### 5. (Optional) Evaluate with TokenBench

Follow the steps from the Cosmos-4B tutorial to evaluate with TokenBench. The command will be the same.
