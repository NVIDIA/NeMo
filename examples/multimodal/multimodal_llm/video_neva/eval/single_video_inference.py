"""
How to run this file:

python3 single_video_inference.py \
    --model-name <path of llava weights, for eg "LLaVA-7B-Lightening-v1-1"> \
    --projection_path <path of projection for \
    --video_path <video_path>
"""

import argparse
import os

import cv2
import numpy as np
import torch

# add new packages as below
from PIL import Image

from nemo.collections.multimodal.data.neva.conversation import Conversation, SeparatorStyle, conv_templates
from nemo.collections.multimodal.data.video_neva.video_neva_dataset import (
    TarOrFolderImageLoader,
    TarOrFolderVideoLoader,
)
from nemo.collections.multimodal.parts.utils import create_neva_model_and_processor

# Define constants
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"


def video_neva_infer(
    video_frames, question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len
):
    """
    Run inference using the Video-ChatGPT model.

    Parameters:
    sample : Initial sample
    video_frames (torch.Tensor): Video frames to process.
    question (str): The question string.
    conv_mode: Conversation mode.
    model: The pretrained Video-ChatGPT model.
    vision_tower: Vision model to extract video features.
    tokenizer: Tokenizer for the model.
    image_processor: Image processor to preprocess video frames.
    video_token_len (int): The length of video tokens.

    Returns:
    dict: Dictionary containing the model's output.
    """

    # Get inference config
    inference_config = model.get_inference_config()

    # Prepare question string for the model
    if model.get_model().vision_config.use_vid_start_end:
        qs = (
            question
            + '\n'
            + DEFAULT_VID_START_TOKEN
            + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len
            + DEFAULT_VID_END_TOKEN
        )
    else:
        qs = question + '\n' + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len

    # Prepare conversation prompt
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize the prompt
    inputs = tokenizer([prompt])

    # Preprocess video frames and get image tensor
    image_tensor = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']

    # Move inputs to GPU
    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    # Run model inference
    with torch.inference_mode():
        output_ids = model.generate(input_ids, **inference_config)

    # Check if output is the same as input
    n_diff_input_output = (input_ids != output_ids[:, : input_ids.shape[1]]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

    # Decode output tokens
    outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1] :], skip_special_tokens=True)[0]

    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--vision_tower_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--projection_path", type=str, required=False, default="")
    parser.add_argument("--video_path", type=str, required=True, default="")
    parser.add_argument("--conv_mode", type=str, required=False, default="")
    parser.add_argument("--num_frames", type=int, required=False, default=8)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    model, vision_tower, tokenizer, image_processor, video_token_len = create_neva_model_and_processor(
        args.model_name, args.projection_path
    )

    video_path = args.video_path

    if os.path.exists(video_path):
        video_data_loader = TarOrFolderVideoLoader(video_folder=video_path)
        video_object = video_data_loader.open_video(video_path)
        width = video_object.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video_object.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = video_object.get(cv2.CAP_PROP_FPS)
        frames = video_object.get(cv2.CAP_PROP_FRAME_COUNT)
        print(type(video_object), " Video Resolution: ", width, " x ", height, ", FPS: ", fps)

        video_frames = video_data_loader.flatten_frames(video_object, args.num_frames)
        print(type(frames), " Shape: ", frames.shape)

    question = input("Enter a question to check from the video:")
    conv_mode = args.conv_mode

    try:
        # Run inference on the video and add the output to the list
        output = video_neva_infer(
            video_frames, question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len
        )
        print("\n\n", output)

    except Exception as e:
        print(f"Error processing video file '{video_path}': {e}")
