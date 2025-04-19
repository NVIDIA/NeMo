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
import random
import shutil
from pathlib import Path

import torch
from cosmos1.models.autoregressive.nemo.utils import read_input_videos
from cosmos1.models.autoregressive.tokenizer.discrete_video import DiscreteVideoFSQJITTokenizer
from cosmos1.models.common.t5_text_encoder import CosmosT5TextEncoder
from einops import rearrange
from huggingface_hub import snapshot_download

BOV_TOKEN = 64000
PAD_ID = 64002


def _get_text_prompt_embeddings(prompt, text_encoder):
    prompt_embedding, prompt_mask = text_encoder.encode_prompts([prompt])
    return prompt_embedding.squeeze()


def _get_video_tokens(filepath, video_tokenizer, data_resolution, num_frames, tokenizer_compression_factor):
    input_video = read_input_videos(filepath, data_resolution, num_frames).cuda()
    batch_size, channels, frames, height, width = input_video.shape
    latent_shape = (
        (frames - 1) // tokenizer_compression_factor[0] + 1,
        height // tokenizer_compression_factor[1],
        width // tokenizer_compression_factor[2],
    )
    T, H, W = latent_shape
    video_tokenizer.latent_chunk_duration = T
    quantized_out, _ = video_tokenizer.encode(input_video, pixel_chunk_duration=None)
    indices = video_tokenizer.fsq_quantizer.codes_to_indices(quantized_out.permute(0, 2, 3, 4, 1))
    indices = rearrange(indices, "B T H W -> (B T H W)")
    video_tokens = torch.IntTensor([BOV_TOKEN] + indices.tolist() + [PAD_ID] * 64)
    return video_tokens


def main(args):
    text_encoder = CosmosT5TextEncoder().cuda()

    if args.encoder_path.startswith("nvidia/"):
        args.encoder_path = os.path.join(snapshot_download(args.encoder_path), "encoder.jit")
    if args.decoder_path.startswith("nvidia/"):
        args.decoder_path = os.path.join(snapshot_download(args.decoder_path), "decoder.jit")

    tokenizer_compression_factor = list(map(int, args.tokenizer_compression_factor.split(",")))
    assert len(tokenizer_compression_factor) == 3, "Tokenizer compression factor must be a tuple of 3 integers"

    data_resolution = [args.height, args.width]
    num_frames = args.num_context_frames

    video_tokenizer = DiscreteVideoFSQJITTokenizer(
        enc_fp=args.encoder_path,
        dec_fp=args.decoder_path,
        name="discrete_video_fsq",
        pixel_chunk_duration=num_frames,
    ).cuda()

    def save_tensors(jsonl_contents, split):
        assert (
            len(jsonl_contents) > 0
        ), f"Ensure length of the {split} split is atleast 1. Modify split string accordingly or add more data points"
        for idx, jsonl_content in enumerate(jsonl_contents):
            json_data = json.loads(jsonl_content)
            assert "prompt" in json_data, "Expected key prompt with text prompt in the input jsonl file"
            assert (
                "visual_input" in json_data
            ), "Expected key visual_input with path to video/image in the input jsonl file"

            video_filename = json_data["visual_input"]
            prompt = json_data["prompt"]
            prompt_embedding = _get_text_prompt_embeddings(prompt, text_encoder)
            video_tokens = _get_video_tokens(
                video_filename, video_tokenizer, data_resolution, num_frames, tokenizer_compression_factor
            )

            torch.save(prompt_embedding, f"{args.output_dir}/{split}_prompt_{idx}.pt")
            torch.save(video_tokens, f"{args.output_dir}/{split}_video_{idx}.pt")

    with open(args.input_jsonl, "r") as f:
        jsonl_file_contents = list(f)

    random.shuffle(jsonl_file_contents)

    num_files = len(jsonl_file_contents)
    train_split, test_split, val_split = [int(split) for split in args.split_string.split(",")]
    assert train_split != 0, "train split in split string is 0. Please make it positive"
    assert test_split != 0, "test split in split string is 0. Please make it positive"
    assert val_split != 0, "val split in split string is 0. Please make it positive"
    total = train_split + test_split + val_split
    (
        train_fraction,
        test_fraction,
    ) = (
        train_split * num_files // total,
        test_split * num_files // total,
    )
    train_jsonl_file_contents = jsonl_file_contents[:train_fraction]
    test_jsonl_file_contents = jsonl_file_contents[train_fraction : (train_fraction + test_fraction)]
    val_jsonl_file_contents = jsonl_file_contents[(train_fraction + test_fraction) :]

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir, ignore_errors=True)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    save_tensors(train_jsonl_file_contents, "train")
    save_tensors(test_jsonl_file_contents, "test")
    save_tensors(val_jsonl_file_contents, "val")

    metadata = {
        "train_samples": len(train_jsonl_file_contents),
        "test_samples": len(test_jsonl_file_contents),
        "val_samples": len(val_jsonl_file_contents),
    }

    with open(f"{args.output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some configurations.")
    parser.add_argument("--width", default=1024, type=int, help="The width of the video")
    parser.add_argument("--height", default=640, type=int, help="The height of the video")
    parser.add_argument("--num_context_frames", default=33, type=int, help="The number of frames in the video")
    parser.add_argument(
        "--tokenizer_compression_factor", default="8,16,16", type=str, help="The compression factor of the tokenizer"
    )
    parser.add_argument(
        "--input_jsonl",
        required=True,
        type=str,
        help="The path to the a jsonl file. Each line of the file should be a dictionary with two keys. visual_input is a key with the video file as value, and prompt is a key , with the text prompt as value. ",
    )
    parser.add_argument(
        "--encoder_path", default="nvidia/Cosmos-1.0-Tokenizer-DV8x16x16", type=str, help="The path to encoder"
    )
    parser.add_argument(
        "--decoder_path", default="nvidia/Cosmos-1.0-Tokenizer-DV8x16x16", type=str, help="The path to the decoder"
    )
    parser.add_argument("--split_string", default="4,1,1", type=str, help="The train/test/val split")
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="The directory to store the prompt embeddings and video tokens",
    )
    args = parser.parse_args()

    with torch.no_grad():
        main(args)
