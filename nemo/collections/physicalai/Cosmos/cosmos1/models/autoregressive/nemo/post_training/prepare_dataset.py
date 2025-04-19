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

import os
import re
from argparse import ArgumentParser
from glob import glob

import torch
from cosmos1.models.autoregressive.nemo.utils import read_input_videos
from cosmos1.models.autoregressive.tokenizer.discrete_video import DiscreteVideoFSQJITTokenizer
from cosmos1.utils import log
from einops import rearrange
from huggingface_hub import snapshot_download

from nemo.collections.nlp.data.language_modeling.megatron import indexed_dataset

CHUNK_SIZE = 250  # Number of videos per chunk

NUM_GPUS = torch.cuda.device_count()


def main(args):
    # --------------------------------------------------------------------------
    # 1) Initialize distributed (if launched with torchrun).
    #    If you plan to run single-GPU only, you could skip this entirely.
    # --------------------------------------------------------------------------
    # Local rank is used for picking the GPU (e.g., rank 0 -> GPU 0, rank 1 -> GPU 1, etc.)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"]) % NUM_GPUS
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    # Download the tokenizer JIT models to local cache if needed
    if args.encoder_path.startswith("nvidia/"):
        args.encoder_path = os.path.join(snapshot_download(args.encoder_path), "encoder.jit")
    if args.decoder_path.startswith("nvidia/"):
        args.decoder_path = os.path.join(snapshot_download(args.decoder_path), "decoder.jit")

    NUM_CONTEXT_FRAMES = int(args.num_context_frames)
    # Validate tokenizer compression factor format
    if not re.match(r"^\d+,\d+,\d+$", args.tokenizer_compression_factor):
        raise ValueError("Invalid tokenizer_compression_factor format. Expected format like '8,16,16' or '4,8,8'")
    TOKENIZER_COMPRESSION_FACTOR = [int(x) for x in args.tokenizer_compression_factor.split(",")]
    # --------------------------------------------------------------------------
    # Instantiate the tokenizer and move it to the correct GPU
    # --------------------------------------------------------------------------
    video_tokenizer = DiscreteVideoFSQJITTokenizer(
        enc_fp=args.encoder_path,
        dec_fp=args.decoder_path,
        name="discrete_video_fsq",
        pixel_chunk_duration=NUM_CONTEXT_FRAMES,
        compression_ratio=TOKENIZER_COMPRESSION_FACTOR,
    ).to(device)

    # --------------------------------------------------------------------------
    # Gather all mp4 filepaths and sort them
    # --------------------------------------------------------------------------
    filepaths_final = sorted(glob(f"{args.input_videos_dir}/*.mp4"))[:10000]
    total_files = len(filepaths_final)
    if total_files == 0:
        if rank == 0:  # Print once
            log.warning(f"No .mp4 files found in {args.input_videos_dir}.")
        return

    if rank == 0:
        log.info(f"Found {total_files} .mp4 files in {args.input_videos_dir}.")
        log.info(f"Chunk size = {CHUNK_SIZE}, total chunks = {((total_files-1)//CHUNK_SIZE)+1}.")

    # --------------------------------------------------------------------------
    # Loop over chunks, but only process the chunks that match our rank
    # Example: If there are 10 chunks total and world_size=4,
    #   rank 0 processes chunks 0,4,8
    #   rank 1 processes chunks 1,5,9
    #   rank 2 processes chunks 2,6
    #   rank 3 processes chunks 3,7
    # --------------------------------------------------------------------------
    num_chunks = (total_files - 1) // CHUNK_SIZE + 1
    log.info(f"World size: {world_size}, rank: {rank}, num_chunks: {num_chunks}")
    for chunk_idx in range(num_chunks):
        # Skip chunks not assigned to this rank
        if chunk_idx % world_size != rank:
            continue
        # print amount of GPU memory used
        start_index = chunk_idx * CHUNK_SIZE
        chunk_filepaths = filepaths_final[start_index : start_index + CHUNK_SIZE]

        # Prepare output directory for this chunk
        chunk_dir = os.path.join(args.output_prefix, str(chunk_idx)) if num_chunks > 1 else (args.output_prefix + "/")
        bin_file = os.path.join(chunk_dir, ".bin")
        idx_file = os.path.join(chunk_dir, ".idx")

        # Skip if dataset already exists
        if os.path.exists(bin_file) and os.path.exists(idx_file):
            log.info(f"[Rank {rank}] Skipping chunk {chunk_idx}, data already exists.")
            continue

        if os.path.exists(bin_file):
            log.info(f"[Rank {rank}] Deleting existing bin file {bin_file}")
            os.remove(bin_file)
        elif os.path.exists(idx_file):
            log.info(f"[Rank {rank}] Deleting existing idx file {idx_file}")
            os.remove(idx_file)

        # ----------------------------------------------------------------------
        # Create a new builder for this chunk
        # ----------------------------------------------------------------------
        os.makedirs(chunk_dir, exist_ok=True)
        builder = indexed_dataset.make_builder(
            bin_file,
            impl="mmap",
            chunk_size=64,
            pad_id=0,
            retrieval_db=None,
            vocab_size=64000,
            stride=64,
        )

        log.info(f"[Rank {rank}] Processing chunk {chunk_idx}, number of files = {len(chunk_filepaths)}")

        for idx_in_chunk, filepath in enumerate(chunk_filepaths, 1):
            try:
                # Move the video input to GPU
                input_video = read_input_videos(
                    filepath, data_resolution=[int(args.height), int(args.width)], num_frames=NUM_CONTEXT_FRAMES
                ).to(device)
                input_video = input_video[:, :, :NUM_CONTEXT_FRAMES, :, :]

                batch_size, channels, frames, height, width = input_video.shape
                latent_shape = (
                    (frames - 1) // TOKENIZER_COMPRESSION_FACTOR[0] + 1,
                    height // TOKENIZER_COMPRESSION_FACTOR[1],
                    width // TOKENIZER_COMPRESSION_FACTOR[2],
                )
                T, H, W = latent_shape

                # Temporarily update the tokenizer's latent_chunk_duration
                video_tokenizer.latent_chunk_duration = T

                quantized_out, _ = video_tokenizer.encode(input_video, pixel_chunk_duration=None)
                indices = video_tokenizer.fsq_quantizer.codes_to_indices(quantized_out.permute(0, 2, 3, 4, 1))

                # Flatten to 1D
                indices = rearrange(indices, "B T H W -> (B T H W)").detach().cpu()
                builder.add_item(torch.IntTensor(indices))
                builder.end_document()

                log.info(
                    f"[Rank {rank}] Chunk {chunk_idx}, file {idx_in_chunk}/{len(chunk_filepaths)}: "
                    f"{os.path.basename(filepath)} processed successfully."
                )

            except Exception as e:
                log.error(f"[Rank {rank}] Error processing {filepath}: {e}")

        # Finalize .idx and .bin for this chunk
        builder.finalize(idx_file)
        log.info(f"[Rank {rank}] Stored .bin and .idx files in {chunk_dir}")

    if rank == 0:
        log.info("All ranks have finished processing.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_videos_dir", required=True, type=str, help="Path to the input .mp4 files")
    parser.add_argument("--width", default=1024, type=int, help="Width of the input videos")
    parser.add_argument("--height", default=640, type=int, help="Height of the input videos")
    parser.add_argument(
        "--encoder_path",
        default="nvidia/Cosmos-1.0-Tokenizer-DV8x16x16",
        type=str,
        help="Hugging Face repo or local path to encoder",
    )
    parser.add_argument(
        "--decoder_path",
        default="nvidia/Cosmos-1.0-Tokenizer-DV8x16x16",
        type=str,
        help="Hugging Face repo or local path to decoder",
    )
    parser.add_argument(
        "--num_context_frames",
        default=33,
        type=int,
        help="Number of context frames to use for the tokenizer",
    )
    parser.add_argument(
        "--tokenizer_compression_factor",
        default="8,16,16",
        type=str,
        help="Tokenizer compression factor",
    )
    parser.add_argument(
        "--output_prefix",
        required=True,
        type=str,
        help="Directory to write chunked .idx and .bin files (e.g. /path/to/output)",
    )
    args = parser.parse_args()

    with torch.no_grad():
        main(args)
