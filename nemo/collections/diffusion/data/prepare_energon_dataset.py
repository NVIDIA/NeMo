# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0115,C0116,C0301

import os
import pickle
from typing import Callable, List

import nemo_run as run
import numpy as np
import torch
import torch.distributed as dist
import webdataset as wds


def get_start_end_idx_for_this_rank(dataset_size, rank, world_size):
    """
    Calculate the start and end indices for a given rank in a distributed setting.

    Args:
        dataset_size (int): The total size of the dataset.
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.

    Returns:
        tuple: A tuple containing the start index (int) and end index (int) for the given rank.
    """
    split_size = dataset_size // world_size
    start_idx = rank * split_size
    # The last rank takes the remainder
    end_idx = start_idx + split_size if rank != world_size - 1 else dataset_size
    return start_idx, end_idx


def dummy_process_func(input):
    """
    Generates a sample dictionary containing random image latent tensor, text embedding,
    and metadata based on the provided input key.

    Args:
        input (str): The key to be used in the sample dictionary.

    Returns:
        dict: A dictionary containing the following keys:
            - "__key__": The input key.
            - ".pth": A randomly generated image latent tensor with shape (3, 1, 720, 1280) and dtype torch.bfloat16.
            - ".pickle": A pickled numpy array representing a random text embedding with shape (512, 2048).
            - ".json": A dictionary containing metadata with keys:
                - "image_height": The height of the image (720).
                - "image_width": The width of the image (1280).
    """
    C, T, H, W = 3, 1, 720, 1280
    image_latent = torch.randn(C, T, H, W, dtype=torch.bfloat16)
    text_embedding = np.random.randn(512, 2048)
    sample = {
        "__key__": input,
        ".pth": image_latent,
        ".pickle": pickle.dumps(text_embedding),
        ".json": {
            "image_height": H,
            "image_width": W,
        },
    }
    return sample


@torch.no_grad()
@run.cli.entrypoint
def prepare(process_func: Callable, inputs: List[str], output_dir: str = 'output'):
    """
    distributed prepration webdataset using the provided processing function, and writes the processed samples to tar files.

    Args:
        process_func (Callable): A function that processes a single input and returns the processed sample.
        inputs (List[str]): A list of input file paths or data entries to be processed.
        output_dir (str, optional): The directory where the output tar files will be saved. Defaults to 'output'.
    """
    rank = dist.get_rank()
    world_size = torch.distributed.get_world_size()

    start_idx, end_idx = get_start_end_idx_for_this_rank(len(inputs), rank, world_size)
    os.makedirs(output_dir, exist_ok=True)
    output_tar = os.path.join(output_dir, f"rank{rank}-%06d.tar")
    with wds.ShardWriter(output_tar, maxcount=10000) as sink:
        for i in range(start_idx, end_idx):
            sample = process_func(inputs[i])
            # Write the sample to the tar file
            sink.write(sample)


@run.cli.factory(target=prepare)
def prepare_dummy_image_dataset() -> run.Partial:
    recipe = run.Partial(
        prepare,
        process_func=dummy_process_func,
        inputs=list(str(i) + '.jpg' for i in range(10000)),
    )
    return recipe


if __name__ == '__main__':
    dist.init_process_group("nccl")
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    run.cli.main(prepare, default_factory=prepare_dummy_image_dataset)
