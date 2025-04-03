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

from pathlib import Path
from typing import Literal

import huggingface_hub
import torch
import typer
from cosmos1.models.autoregressive.nemo.post_training.action_control.prepare_dataset import (
    Split,
    VideoDataset,
    download_bridge_data,
    get_annotations,
    get_default_output_prefix,
)
from tqdm import tqdm

from nemo.collections.diffusion.vae.video_vae import VideoJITTokenizer


def create_tokenizer(tokenizer_tag: str = "nvidia/Cosmos-1.0-Tokenizer-CV8x8x8"):
    """Creates a DiscreteVideoFSQJITTokenizer from a Hugging Face Hub tokenizer tag."""

    tokenizer_path = Path(huggingface_hub.snapshot_download(tokenizer_tag))

    name = "cosmos_tokenizer"
    enc_fp = tokenizer_path / "encoder.jit"
    dec_fp = tokenizer_path / "decoder.jit"
    video_mean_std_fp = tokenizer_path / "mean_std.pt"

    image_vae = VideoJITTokenizer(
        str(tokenizer_path),
        str(enc_fp),
        str(dec_fp),
        name,
        str(video_mean_std_fp),
        pixel_chunk_duration=1,
        spatial_compression_factor=8,
        temporal_compression_factor=8,
    )

    return image_vae.cuda()


def get_tokenized_frames(
    dataset_dir: Path,
    dataset_split: Literal["train", "val", "test"],
    tokenizer_tag: str,
    batch_size: int,
    num_workers: int,
):
    """Tokenizes the video frames from the IRASim bridge dataset and saves them to the local filesystem."""

    video_tokenizer = create_tokenizer(tokenizer_tag)
    dataset = VideoDataset(dataset_dir, dataset_split)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=lambda x: torch.concat(x, dim=0),
    )

    def iter_tokenized_batches():
        for batch in tqdm(dataloader):
            # Convert to bf16 and normalize from [0, 255] to [-1, 1]
            batch = batch.to(device="cuda", dtype=torch.bfloat16, non_blocking=True) / 127.5 - 1.0
            latent_frames = video_tokenizer.encode(batch)
            yield latent_frames.detach().to("cpu")

    with torch.no_grad():
        return torch.cat(list(iter_tokenized_batches()), dim=0)


def main(
    tokenizer_tag: str = "nvidia/Cosmos-1.0-Tokenizer-CV8x8x8",
    output_prefix: str | None = None,
    dataset_split: Split = Split.train,
    dataset_dir: str | None = None,
    batch_size: int = 10,
    num_workers: int = 10,
):
    """Prepare the bridge dataset for diffusion post-training.

    This script will download the bridge dataset from lf-robot-opensource.bytetos.com, extract the annotations and
    video frames, and tokenize the video frames using the specified tokenizer. The resulting files will be saved
    to the local filesystem.

    Args:
        tokenizer_tag: The tag of the tokenizer to be used in tokenizing the video frames.
        output_prefix: The prefix to be used for the output files. If omitted, the output files will be saved to
            the huggingface cache directory.
        dataset_split: The split of the dataset to be processed.
        dataset_dir: The path to the extracted contents of bridge_train_data.tar.gz, in the format provided by IRASim.
            If omitted, the ~30Gb dataset will be downloaded from lf-robot-opensource.bytetos.com.
        batch_size: The batch size (number of clips) to use during tokenization.
        num_workers: The number of worker processes to use when processing the dataset.
    """

    if dataset_dir is None:
        dataset_path = download_bridge_data() / "opensource_robotdata" / "bridge"
    else:
        dataset_path = Path(dataset_dir)

    if output_prefix is None:
        output_path = get_default_output_prefix(dataset_split.value, subfolder="diffusion")
    else:
        output_path = Path(output_prefix)

    all_annotations = get_annotations(dataset_path, dataset_split.value, batch_size, num_workers)
    latent_fames = get_tokenized_frames(dataset_path, dataset_split.value, tokenizer_tag, batch_size, num_workers)

    torch.save(all_annotations["state"], output_path / "state.pt")
    torch.save(all_annotations["action"], output_path / "actions.pt")
    torch.save(all_annotations["continuous_gripper_state"], output_path / "gripper.pt")
    torch.save(latent_fames, output_path / "tokenized-frames.pt")


if __name__ == "__main__":
    typer.run(main)
