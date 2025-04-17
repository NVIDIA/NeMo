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

import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Literal, TypedDict

import huggingface_hub
import numpy as np
import pooch
import torch
import torchvision
import typer
from cosmos1.models.autoregressive.tokenizer.discrete_video import DiscreteVideoFSQJITTokenizer
from einops import rearrange
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_tokenizer(tokenizer_tag: str = "nvidia/Cosmos-1.0-Tokenizer-DV8x16x16"):
    """Creates a DiscreteVideoFSQJITTokenizer from a Hugging Face Hub tokenizer tag."""

    tokenizer_path = Path(huggingface_hub.snapshot_download(tokenizer_tag))

    # In the action-control finetuning task, we predict next frames as a function of the previous
    # frame and a action vector. We set the pixel_chunk_duration to 1 to indicate that we're
    # tokenizing indidivual still frames rather than entire videos.
    video_tokenizer = DiscreteVideoFSQJITTokenizer(
        enc_fp=str(tokenizer_path / "encoder.jit"),
        dec_fp=str(tokenizer_path / "decoder.jit"),
        name="discrete_video_fsq",
        pixel_chunk_duration=1,
    ).cuda()

    video_tokenizer.latent_chunk_duration = 1

    return video_tokenizer


def download_bridge_data() -> Path:
    """Downloads the bridge dataset (if not available in the hf cache) and extracts it to the local filesystem."""

    logger.info("Downloading bridge dataset...")

    ds_path = huggingface_hub.cached_assets_path("cosmos", namespace="action-control", subfolder="datasets")
    if (ds_path / "039134bac1ecbf2f26fe22b3a32078d9-bridge_train_data.tar.gz.untar").exists():
        logger.info("Bridge dataset already downloaded, skipping download.")
        return ds_path / "039134bac1ecbf2f26fe22b3a32078d9-bridge_train_data.tar.gz.untar"

    processor = pooch.Untar()
    _ = pooch.retrieve(
        url="https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/opensource_IRASim_v1/bridge_train_data.tar.gz",
        known_hash=None,  # Open-source dataset hash known to change sometimes.
        processor=processor,
        path=ds_path,
        progressbar=True,
    )
    # Validating the hash of the downloaded file can take a while, so we skip it here. If you want
    # to validate the hash, uncomment the known_hash line above and remove the known_hash=None line.
    logger.info("Bridge dataset downloaded successfully.")

    return Path(processor.extract_dir)  # type: ignore


class ClipDataset(torch.utils.data.Dataset):
    """A dataset of clips in the IRASim bridge structure, where each clip is identified by an integer ID."""

    def __init__(
        self,
        bridge_data_root_dir: str | os.PathLike,
        split: Literal["train", "val", "test"] = "train",
    ):
        """Initializes the dataset.

        Args:
            bridge_data_root_dir: The root directory of the bridge dataset.
            split: The split of the dataset to be used.
        """

        video_files = (Path(bridge_data_root_dir) / "videos" / split).glob("*/rgb.mp4")
        self.split = split
        self.bridge_data_root_dir = Path(bridge_data_root_dir)
        self.clip_ids = sorted([int(path.parent.stem) for path in video_files])

    def __len__(self) -> int:
        return len(self.clip_ids)


class VideoDataset(ClipDataset):
    """A dataset to decode video frames from the IRASim bridge dataset."""

    def __getitem__(self, i: int) -> torch.Tensor:
        """Gets all the frames from a single video clip by its ID.

        Args:
            i: The index of the clip in the dataset.

        Returns:
            A tensor of shape (T, C, 1, H, W) containing the video frames.
        """

        video_file = self.bridge_data_root_dir / "videos" / self.split / f"{self.clip_ids[i]}" / "rgb.mp4"

        # We use torchvision to perform the mp4 decoding, which means this operation is CPU-bound.
        # By placing this in a dataset class and subsequently using a multiprocess dataloader, we can attempt to
        # make this operation not the bottleneck in video tokenization.
        video, _, _ = torchvision.io.read_video(str(video_file), pts_unit="sec")

        # Here we rearrange the video tensor from a single video clip to a batch of individual
        # frames, by moving the temporal dimension to the batch dimension and subsequently using a
        # collate function that concatenates the frames.
        frames = rearrange(video, "t h w c -> t c 1 h w")
        return frames


class AnnotationBatch(TypedDict):
    """A single parsed clip's annotations"""

    state: torch.Tensor
    action: torch.Tensor
    continuous_gripper_state: torch.Tensor


class AnnotationDataset(ClipDataset):
    """A dataset to load annotations, primarily the action vector, from the IRASim bridge dataset."""

    def __getitem__(self, i: int) -> AnnotationBatch:
        """Gets the annotations for a single video clip by its ID.

        Args:
            i: The index of the clip in the dataset.

        Returns:
            A dictionary containing the state, action, and continuous gripper state annotations.
        """

        state_file = self.bridge_data_root_dir / "videos" / self.split / f"{self.clip_ids[i]}" / "state.npy"
        annotation_file = self.bridge_data_root_dir / "annotation" / self.split / f"{self.clip_ids[i]}.json"
        with open(annotation_file, "rt") as f:
            annotations = json.load(f)

        # Before returning the action tensor, we append a row of NaNs to the end of the tensor,
        # since the action here represents the movement of the arm between frame i and frame i+1.
        # The action tensor is therefore 1 row shorter than the number of frames in the video, and
        # adding these NaNs lets us easily align these tensors and find valid training samples.
        return {
            "state": torch.Tensor(np.load(state_file)),
            "action": torch.cat([torch.Tensor(annotations["action"]), np.nan * torch.ones((1, 7))], dim=0),
            "continuous_gripper_state": torch.Tensor(annotations["continuous_gripper_state"]),
        }


def dict_cat_collate(batch: list[AnnotationBatch]) -> AnnotationBatch:
    """Collate function to concatenate the annotations in a batch of clips."""

    return {
        "state": torch.cat([d["state"] for d in batch], dim=0),
        "action": torch.cat([d["action"] for d in batch], dim=0),
        "continuous_gripper_state": torch.cat([d["continuous_gripper_state"] for d in batch], dim=0),
    }


def get_annotations(
    dataset_dir: Path,
    dataset_split: Literal["train", "val", "test"],
    batch_size: int,
    num_workers: int,
) -> AnnotationBatch:
    """Saves the annotations from the IRASim bridge dataset to the local filesystem."""

    dataset = AnnotationDataset(dataset_dir, dataset_split)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=dict_cat_collate,
    )

    return dict_cat_collate(list(dataloader))


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
            _, indices = video_tokenizer.encode(batch, pixel_chunk_duration=None)
            yield indices.detach().to("cpu")

    # Here we're not being very memory efficient, as we're storing all the tokenized frames in
    # memory before saving them to disk. For the bridge dataset with the 16x16 tokenizer, this
    # results in only about a 4 Gb object.
    with torch.no_grad():
        return torch.cat(list(iter_tokenized_batches()), dim=0)


class Split(str, Enum):
    train = "train"
    val = "val"
    test = "test"


def get_default_output_prefix(dataset_split: Literal["train", "val", "test"], subfolder: str | None = None) -> Path:
    """Returns the default directory for serializing the bridge dataset.

    Args:
        dataset_split: The split of the dataset to be processed.
        subfolder: The subfolder to use in HF_HOME/assets/cosmos/action-control. If not provided, the default
            subfolder "autoregressive" will be used.
    """
    if subfolder is None:
        subfolder = "autoregressive"

    output_path = (
        huggingface_hub.cached_assets_path("cosmos", namespace="action-control", subfolder=subfolder)
        / "bridge"
        / dataset_split
    )
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def main(
    tokenizer_tag: str = "nvidia/Cosmos-1.0-Tokenizer-DV8x16x16",
    output_prefix: str | None = None,
    dataset_split: Split = Split.train,
    dataset_dir: str | None = None,
    batch_size: int = 10,
    num_workers: int = 10,
):
    """Prepare the bridge dataset for autoregressive post-training.

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
        output_path = get_default_output_prefix(dataset_split.value)
    else:
        output_path = Path(output_prefix)

    logger.info("Serializing annotations...")
    all_annotations = get_annotations(dataset_path, dataset_split.value, batch_size, num_workers)

    logger.info("Serializing tokenized frames...")
    all_tokenized_frames = get_tokenized_frames(
        dataset_path, dataset_split.value, tokenizer_tag, batch_size, num_workers
    )

    torch.save(all_annotations["state"], output_path / "state.pt")
    torch.save(all_annotations["action"], output_path / "actions.pt")
    torch.save(all_annotations["continuous_gripper_state"], output_path / "gripper.pt")
    torch.save(all_tokenized_frames, output_path / "tokenized-frames.pt")


if __name__ == "__main__":
    typer.run(main)
