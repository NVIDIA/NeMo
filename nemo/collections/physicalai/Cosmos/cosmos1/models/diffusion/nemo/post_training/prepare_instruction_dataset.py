# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0115,C0116,C0301

import argparse
import glob
import json
import os
import random

import torch
import torchvision
from cosmos1.utils import log
from einops import rearrange
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import T5EncoderModel, T5TokenizerFast

from nemo.collections.diffusion.models.model import DiT7BConfig


def get_parser():
    parser = argparse.ArgumentParser(description="Process some configurations.")
    parser.add_argument("--tokenizer_dir", type=str, default="", help="Path to the VAE model")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="video_dataset",
        help="Path to the dataset. A folder of with videos, instructions, and metas subfolders.",
    )
    parser.add_argument("--output_path", type=str, default="video_dataset_cached", help="Path to the output directory")
    parser.add_argument("--num_chunks", type=int, default=5, help="Number of random chunks to sample per video")
    parser.add_argument("--height", type=int, default=704, help="Height to resize video")
    parser.add_argument("--width", type=int, default=1280, help="Width to resize video")
    return parser


def init_t5():
    """Initialize and return the T5 tokenizer and text encoder."""
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-11b")
    text_encoder = T5EncoderModel.from_pretrained("google-t5/t5-11b")
    text_encoder.to("cuda")
    text_encoder.eval()
    return tokenizer, text_encoder


def init_video_tokenizer(tokenizer_dir: str):
    """Initialize and return the Cosmos Video tokenizer."""
    dit_config = DiT7BConfig(vae_path=tokenizer_dir)
    vae = dit_config.configure_vae()
    return vae


@torch.no_grad()
def encode_for_batch(tokenizer, encoder, prompts: list[str], max_length=512):
    """
    Encode a batch of text prompts to a batch of T5 embeddings.
    Parameters:
        tokenizer: T5 embedding tokenizer.
        encoder: T5 embedding text encoder.
        prompts: A batch of text prompts.
        max_length: Sequence length of text embedding (defaults to 512).
    """

    batch_encoding = tokenizer.batch_encode_plus(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_length=True,
        return_offsets_mapping=False,
    )

    # We expect all the processing is done on GPU.
    input_ids = batch_encoding.input_ids.cuda()
    attn_mask = batch_encoding.attention_mask.cuda()

    outputs = encoder(input_ids=input_ids, attention_mask=attn_mask)
    encoded_text = outputs.last_hidden_state

    lengths = attn_mask.sum(dim=1).cpu()
    for batch_id in range(encoded_text.shape[0]):
        encoded_text[batch_id][lengths[batch_id] :] = 0

    return encoded_text


def create_condition_latent_from_input_frames(tokenizer, input_frames, num_frames_condition=25):
    B, C, T, H, W = input_frames.shape
    num_frames_encode = tokenizer.pixel_chunk_duration
    assert (
        input_frames.shape[2] >= num_frames_condition
    ), f"input_frames not enough for condition, require at least {num_frames_condition}, get {input_frames.shape[2]}, {input_frames.shape}"
    assert (
        num_frames_encode >= num_frames_condition
    ), f"num_frames_encode should be larger than num_frames_condition, get {num_frames_encode}, {num_frames_condition}"

    # Put the conditioal frames to the begining of the video, and pad the end with zero
    condition_frames = input_frames[:, :, -num_frames_condition:]
    padding_frames = condition_frames.new_zeros(B, C, num_frames_encode - num_frames_condition, H, W)
    encode_input_frames = torch.cat([condition_frames, padding_frames], dim=2).to("cuda")
    vae = tokenizer.to(encode_input_frames.device)
    latent = vae.encode(encode_input_frames)
    return latent, encode_input_frames


def main(args):
    # Set up output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Initialize T5
    tokenizer, text_encoder = init_t5()

    # Initialize the VAE
    if args.tokenizer_dir == "":
        args.tokenizer_dir = snapshot_download("nvidia/Cosmos-1.0-Tokenizer-CV8x8x8")
    vae = init_video_tokenizer(args.tokenizer_dir)

    # Constants
    t5_embeding_max_length = 512
    chunk_duration = vae.video_vae.pixel_chunk_duration  # Frames per chunk
    cnt = 0  # File index

    video_folder = os.path.join(args.dataset_path, "videos")
    instruction_folder = os.path.join(args.dataset_path, "instructions")

    video_paths = glob.glob(os.path.join(video_folder, "*.mp4"))
    # Check if dataset_path is correct
    if not video_paths:
        raise ValueError(f"Dataset path {args.dataset_path} does not contain any .mp4 files.")

    # Process each video in the dataset folder
    with torch.no_grad():
        for video_path in tqdm(video_paths):
            instruction_path = os.path.join(instruction_folder, os.path.basename(video_path).replace(".mp4", ".json"))
            with open(instruction_path, "r") as f:
                instruction = json.load(f)["language_instruction_0"]
            # Read video (T x H x W x C)
            video, _, meta = torchvision.io.read_video(video_path)
            T, H, W, C = video.shape

            # Skip videos shorter than one chunk
            if T < chunk_duration:
                log.info(f"Video {video_path} is shorter than {chunk_duration} frames. Skipped.")
                continue

            # Sample random segments
            num_unique_chunks = T - chunk_duration + 1
            num_chunks = min(args.num_chunks, num_unique_chunks)
            for ix in range(num_chunks):
                if num_unique_chunks < args.num_chunks:
                    start_idx = ix
                else:
                    start_idx = random.randint(0, T - chunk_duration)
                chunk = video[start_idx : start_idx + chunk_duration]  # (chunk_duration, H, W, C)

                # Rearrange dimensions: (T, H, W, C) -> (T, C, H, W)
                chunk = rearrange(chunk, "t h w c -> t c h w")

                # Resize to [704, 1280] for each frame
                chunk = torchvision.transforms.functional.resize(chunk, [args.height, args.width])

                # Expand dims: (T, C, H, W) -> (B=1, C, T, H, W)
                chunk = rearrange(chunk, "(b t) c h w -> b c t h w", b=1)

                # Convert to bf16 and normalize from [0, 255] to [-1, 1]
                chunk = chunk.to(device="cuda", dtype=torch.bfloat16, non_blocking=True) / 127.5 - 1.0

                # Condition Latent (for Video2World training)
                conditioning_chunk_len = 9
                conditioning_chunk = chunk[:, :, :conditioning_chunk_len, ...]

                # Encode video
                latent = vae.encode(chunk).cpu()  # shape: (1, latent_channels, T//factor, H//factor, W//factor)

                # Encode conditioning frames
                conditioning_latent, _ = create_condition_latent_from_input_frames(
                    vae, conditioning_chunk, conditioning_chunk_len
                )
                conditioning_latent = conditioning_latent.cpu()

                # Encode text
                out = encode_for_batch(tokenizer, text_encoder, [instruction])[0]
                encoded_text = torch.tensor(out, dtype=torch.bfloat16)

                # Pad T5 embedding to t5_embeding_max_length
                L, C_ = encoded_text.shape
                t5_embed = torch.zeros(1, t5_embeding_max_length, C_, dtype=torch.bfloat16)
                t5_embed[0, :L] = encoded_text

                # Save data to folder
                torch.save(latent[0], os.path.join(args.output_path, f"{cnt}.video_latent.pth"))
                torch.save(conditioning_latent[0], os.path.join(args.output_path, f"{cnt}.conditioning_latent.pth"))
                torch.save(t5_embed[0], os.path.join(args.output_path, f"{cnt}.t5_text_embeddings.pth"))

                # Create a T5 text mask of all ones
                torch.save(
                    torch.ones(512, dtype=torch.bfloat16), os.path.join(args.output_path, f"{cnt}.t5_text_mask.pth")
                )

                # Save metadata
                info = {
                    "height": H,
                    "width": W,
                    "fps": meta["video_fps"],
                    "num_frames": chunk_duration,
                    "video_path": os.path.basename(video_path),
                    "start_frame": start_idx,
                }
                with open(os.path.join(args.output_path, f"{cnt}.info.json"), "w") as json_file:
                    json.dump(info, json_file)
                    print(f"Saved metadata to {args.output_path}/{cnt}.info.json")

                cnt += 1


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
