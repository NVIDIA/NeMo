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
import json
import os
import pickle
import random
import zipfile
from pathlib import Path
from typing import List

import ffmpeg
import torch
import torchvision
import torchvision.transforms.functional as transforms_F
from cosmos1.models.diffusion.nemo.post_training.prepare_dataset import create_condition_latent_from_input_frames
from cosmos1.utils import log
from einops import rearrange
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import T5EncoderModel, T5TokenizerFast

from nemo.collections.diffusion.data.camera_ctrl_utils import (
    estimate_pose_list_to_plucker_embedding,
    normalize_camera_trajectory_to_unit_sphere,
)
from nemo.collections.diffusion.models.model import DiT7BCameraCtrlConfig


def get_parser():
    parser = argparse.ArgumentParser(description="Process some configurations.")
    parser.add_argument("--tokenizer_dir", type=str, default="", help="Path to the VAE model")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Path to DL3DV dataset",
    )
    parser.add_argument(
        "--video_processing_cache_path",
        type=str,
        default="video_processing_cache",
        help="Path to cache where frames will be extracted and used to form mp4 files",
    )
    parser.add_argument(
        "--path_to_caption_dict",
        type=str,
        default=None,
        help="Path to a pickled python dictionary where the keys are the DL3DV sample hash followed by a chunk num "
        "(e.g., '614d0e91913409a64d6d6c06a9d0c944d62d93b30b1d2d60493efd40ed4a5f9/2'), "
        "and the values are the estimated captions for that chunk",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="camera_ctrl_dataset_cached",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=1,
        help="Number of random chunks to sample per video",
    )
    parser.add_argument("--height", type=int, default=704, help="Height to resize video")
    parser.add_argument("--width", type=int, default=1280, help="Width to resize video")
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed for randomly selecting chunks",
    )
    parser.add_argument(
        "--num_zip_files",
        type=int,
        default=-1,
        help="The number of zip files to use for creating the camera control dataset. "
        "Default will be all zip files found recursively from the "
        "'--dataset_path' directory ",
    )
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
    dit_config = DiT7BCameraCtrlConfig(
        vae_path=tokenizer_dir,
    )
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


def prepare_raw_sample_from_zip(
    input_zip_file: Path,
    video_processing_cache_path: Path,
) -> List[Path]:
    """
    Takes in a path to a zip file from the DL3DV-ALL-4K subset and performs the following:
        1. Creates a .mp4 file from the png frames and writes it to
           video_processing_cache_path/video_files/<sample_hash>.mp4,
           where <sample_hash> is the hash provided by DL3DV
        2. Extracts the transforms.json file to video_processing_cache_path/transform_files/<sample_hash>_
    """
    # Make the output dirs
    output_mp4_dir = video_processing_cache_path / "video_files"
    output_mp4_dir.mkdir(exist_ok=True)
    extracted_dir = video_processing_cache_path / "extracted_frames"
    extracted_dir.mkdir(exist_ok=True)
    transforms_dir = video_processing_cache_path / "transform_files"
    transforms_dir.mkdir(exist_ok=True)

    # First prepare video (mp4) with ffmpeg (convert to 1280,720)
    input_hash = input_zip_file.stem
    output_mp4_file = output_mp4_dir / f"{input_hash}.mp4"
    if not output_mp4_file.exists():
        # Get the hash of the zip file
        # Extract the pngs from the zip file if they haven't been already
        if not Path(extracted_dir / f"{input_hash}").exists():
            with zipfile.ZipFile(str(input_zip_file), "r") as zip_fp:
                for ifile in zip_fp.infolist():
                    if ifile.filename.lower().endswith(".png"):
                        zip_fp.extract(ifile, str(Path(extracted_dir)))
                    elif ifile.filename.lower().endswith(".json"):
                        zip_fp.extract(ifile, str(Path(transforms_dir)))

        # Prepare the transform file
        transform_file = Path(transforms_dir / f"{input_hash}/transforms.json")
        # Handle the case for no sub dir in the zip
        if not transform_file.exists():
            transform_file = Path(transforms_dir / "transforms.json")
        output_transform_file = transform_file.rename(transforms_dir / f"{input_hash}_transforms.json")

        # Remove the directory if it exists
        if Path(transforms_dir / f"{input_hash}").exists():
            Path(transforms_dir / f"{input_hash}").rmdir()

        # Handle the case in which zip does not have a sub dir
        if not Path(extracted_dir / f"{input_hash}").exists():
            Path(extracted_dir / f"{input_hash}/images").mkdir(exist_ok=True, parents=True)
            Path(extracted_dir / "images").rename(extracted_dir / f"{input_hash}/images")

        # Build the input video file
        input_pattern = str(extracted_dir / f"{input_hash}/images/*.png")
        (
            ffmpeg.input(input_pattern, pattern_type="glob", framerate=7)
            .filter("scale", 1280, 720)
            .output(str(output_mp4_file), vcodec="libx264", preset="fast", crf=23)
            .overwrite_output()
            .run()
        )
    else:
        output_transform_file = Path(transforms_dir / f"{input_hash}_transforms.json")
        if not output_transform_file.exists():
            output_transform_file.rename(transforms_dir / f"{input_hash}_transforms.json")

    return output_mp4_file, output_transform_file


def aspect_aware_resize_and_pad(
    video_chunk: torch.Tensor,
    H_target: int,
    W_target: int,
) -> List[torch.Tensor]:
    # [720, 1280]
    _, _, H, W = video_chunk.shape

    # Aspect-ratio-aware resizing
    # [720, 1280] -> [704, 1252]
    scaling_ratio = H_target / H
    H_resized = int(scaling_ratio * H + 0.5)
    W_resized = int(scaling_ratio * W + 0.5)
    video_chunk = transforms_F.resize(
        video_chunk,
        size=(H_resized, W_resized),
        interpolation=transforms_F.InterpolationMode.BICUBIC,
        antialias=True,
    )

    # Reflection padding
    # [704, 1252] -> [704, 1280]
    padding_left = int((W_target - W_resized) / 2)
    padding_right = W_target - W_resized - padding_left
    padding_top = int((H_target - H_resized) / 2)
    padding_bottom = H_target - H_resized - padding_top
    padding_vals = [padding_left, padding_top, padding_right, padding_bottom]
    video_chunk = transforms_F.pad(video_chunk, padding_vals, padding_mode="reflect")

    # Construct the padding mask
    padding_mask = torch.ones((1, H_target, W_target), dtype=torch.bfloat16)
    padding_mask[:, padding_top : (padding_top + H), padding_left : (padding_left + W)] = 0
    image_size = torch.tensor(
        [[H_target, W_target, H_resized, W_resized]] * 1,
        dtype=torch.float16,
    )

    return video_chunk, padding_mask, image_size


def create_plucker_embeddings_from_pose_list(
    camera_data: dict,
    start_idx: int,
    num_frames: int,
    image_size: torch.tensor,
    latent_compression_ratio_h: int = 8,
    latent_compression_ratio_w: int = 8,
):
    """
    Computes the plücker embeddings from an input list of poses
    """
    # Prepare the instrinsics and extrinsics
    intrinsic = [
        [camera_data["fl_x"], 0, camera_data["cx"]],
        [0, camera_data["fl_y"], camera_data["cy"]],
        [0, 0, 1],
    ]

    pose_list = []
    for i, iframe in enumerate(camera_data["frames"]):
        if i > start_idx and i <= start_idx + num_frames:
            rotation, translation = [], []
            # Construct the 3 X 3 rotation matrix and the 3 X 1 translation vector
            for j, row in enumerate(iframe["transform_matrix"]):
                if j < 3:
                    rotation.append(row[:3])
                    translation.append(row[-1])
            pose_list.append(
                {
                    "intrinsics": intrinsic,
                    "rotation": rotation,
                    "translation": translation,
                }
            )

    normalize_camera_trajectory_to_unit_sphere(pose_list)
    plucker_coords, plucker_embeddings_h, plucker_embeddings_w = estimate_pose_list_to_plucker_embedding(
        pose_list,
        latent_compression_ratio_h,
        latent_compression_ratio_w,
        image_size,
    )
    plucker_embeddings = rearrange(
        plucker_coords,
        "b (h w) c -> b c h w",
        h=plucker_embeddings_h,
        w=plucker_embeddings_w,
    )

    return plucker_embeddings.unsqueeze(0), plucker_embeddings_h, plucker_embeddings_w


def main(args):
    # Set up output directory
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.video_processing_cache_path, exist_ok=True)

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

    random.seed(args.seed)

    # Check if dataset_path is correct
    files = list(Path(args.dataset_path).rglob("*.zip"))
    if len(files) == 0:
        raise ValueError(f"Dataset path {args.dataset_path} does not contain any .zip files.")

    if args.num_zip_files == -1:
        num_zip_files = len(files)
    else:
        num_zip_files = args.num_zip_files
    files = random.sample(files, num_zip_files)

    captions = None
    if args.path_to_caption_dict is not None:
        with open(args.path_to_caption_dict, "rb") as captions_fp:
            captions = pickle.load(captions_fp)

    # Process each video in the dataset folder
    with torch.no_grad():
        for zip_path in tqdm(files):
            # Read video (T x H x W x C)
            video_path, transform_path = prepare_raw_sample_from_zip(
                zip_path,
                Path(args.video_processing_cache_path),
            )
            video, _, meta = torchvision.io.read_video(video_path)
            T, H, W, C = video.shape

            with open(transform_path, "r") as transform_fp:
                camera_data = json.load(transform_fp)
                num_camera_frames = len(camera_data["frames"])

            # Check to make sure we have a quality sample
            if num_camera_frames != T:
                log.info(f"Video camera_data is not aligned with video frames. Skipping {video_path}.")
                continue

            # Skip videos shorter than one chunk
            if T < chunk_duration:
                log.info(f"Video {video_path} is shorter than {chunk_duration} frames. Skipped.")
                continue

            # Sample random segments
            for _ in range(args.num_chunks):
                start_idx = random.randint(0, T - chunk_duration)
                chunk = video[start_idx : start_idx + chunk_duration]  # (chunk_duration, H, W, C)

                # Rearrange dimensions: (T, H, W, C) -> (T, C, H, W)
                chunk = rearrange(chunk, "t h w c -> t c h w")

                # Aspect-aware resizing and padding to [704, 1280] for each frame
                chunk, padding_mask, image_size = aspect_aware_resize_and_pad(
                    chunk,
                    args.height,
                    args.width,
                )

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

                # Compute the Plücker embeddings from the camera data
                (
                    plucker_embeddings,
                    plucker_embeddings_h,
                    plucker_embeddings_w,
                ) = create_plucker_embeddings_from_pose_list(
                    camera_data,
                    start_idx,
                    chunk_duration,
                    image_size[0],
                )

                padding_mask = padding_mask.unsqueeze(0)  # 1,1,H_input,W_input
                padding_mask = torch.nn.functional.interpolate(
                    padding_mask, size=(plucker_embeddings_h, plucker_embeddings_w), mode="nearest"
                )
                plucker_embeddings = plucker_embeddings * (1 - padding_mask)

                # Get the corresponding caption for the 0th chunk of the video
                if captions is not None:
                    sample_hash = zip_path.stem
                    key = f"{sample_hash}/0"
                    if key in captions:
                        prompt = captions[f"{sample_hash}/0"]
                    else:
                        log.info(f"Caption doesn't exist in provided captions file. Skipping {sample_hash}.")
                        continue
                else:
                    prompt = "A video of a camera moving"

                # Encode text
                out = encode_for_batch(tokenizer, text_encoder, [prompt])[0]
                encoded_text = torch.tensor(out, dtype=torch.bfloat16)

                # Pad T5 embedding to t5_embeding_max_length
                L, C_ = encoded_text.shape
                t5_embed = torch.zeros(1, t5_embeding_max_length, C_, dtype=torch.bfloat16)
                t5_embed[0, :L] = encoded_text

                # Save data to folder
                torch.save(latent[0], os.path.join(args.output_path, f"{cnt}.video_latent.pth"))
                torch.save(conditioning_latent[0], os.path.join(args.output_path, f"{cnt}.conditioning_latent.pth"))
                torch.save(plucker_embeddings[0], str(Path(args.output_path) / f"{cnt}.plucker_embeddings.pth"))
                torch.save(image_size, str(Path(args.output_path) / f"{cnt}.image_size.pth"))
                torch.save(padding_mask[0], str(Path(args.output_path) / f"{cnt}.padding_mask.pth"))
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
                    "caption": prompt,
                    "start_frame": start_idx,
                }
                with open(os.path.join(args.output_path, f"{cnt}.info.json"), "w") as json_file:
                    json.dump(info, json_file)

                cnt += 1


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
