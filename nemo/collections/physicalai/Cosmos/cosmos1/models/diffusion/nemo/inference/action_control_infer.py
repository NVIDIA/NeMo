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

import matplotlib.pyplot as plt
import numpy as np
import torch
from cosmos1.models.autoregressive.nemo.post_training.action_control.prepare_dataset import Split
from cosmos1.models.diffusion.nemo.post_training.action_control.prepare_dataset import create_tokenizer

from nemo import lightning as nl
from nemo.collections.diffusion.datamodule import ActionControlDiffusionDataset
from nemo.collections.diffusion.models.model import (
    DiT7BVideo2WorldActionConfig,
    DiT14BVideo2WorldActionConfig,
    DiTModel,
)
from nemo.collections.diffusion.sampler.cosmos.cosmos_extended_diffusion_pipeline import ExtendedDiffusionPipeline

DEFAULT_AUGMENT_SIGMA_LIST = 0.001


def create_trainer(tensor_model_parallel_size: int, context_parallel_size: int):
    """Initialize model parallel strategy.
    Here, we only use CP2 because action control only has T=2 frames.

    Args:
        tensor_model_parallel_size: number of the tensor model parallelism
        context_parallel_size: context parallel size
    Returns:
        Trainer: The trainer object configured with fit strategy
    """

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=context_parallel_size,
        pipeline_dtype=torch.bfloat16,
    )

    # Initialize ptl trainer
    trainer = nl.Trainer(
        devices=tensor_model_parallel_size * context_parallel_size,
        max_steps=1,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    )
    return trainer


def setup_diffusion_pipeline(
    nemo_ckpt: str, tensor_model_parallel_size: int, context_parallel_size: int, model_size: str
):
    """
    Initialize the diffusion pipeline from the checkpoint.

    Args:
        nemo_ckpt: Model checkpoint file
        tensor_model_parallel_size: number of the tensor model parallelism
        context_parallel_size: context parallel size
        model_size: the size of the model. One of 7B or 14B.

    Returns:
        ExtendedDiffusionPipeline: diffusion pipeline with the checkpointed model.
    """

    # Initialize the model with the action config
    if model_size == "7B":
        dit_config = DiT7BVideo2WorldActionConfig()
    elif model_size == "14B":
        dit_config = DiT14BVideo2WorldActionConfig()
    else:
        raise ValueError("Invalid model size. Choose '7B' or '14B'.")

    dit_model = DiTModel(dit_config)

    trainer = create_trainer(tensor_model_parallel_size, context_parallel_size)

    # Convert trainer to fabric for inference
    fabric = trainer.to_fabric()
    fabric.strategy.checkpoint_io.save_ckpt_format = "zarr"
    fabric.strategy.checkpoint_io.validate_access_integrity = False
    model = fabric.load_model(nemo_ckpt, dit_model).to(device="cuda", dtype=torch.bfloat16)

    # Initialize the diffusion pipeline
    diffusion_pipeline = ExtendedDiffusionPipeline(
        net=model.module, conditioner=model.conditioner, sampler_type="RES", seed=42
    )

    return diffusion_pipeline


def run_diffusion_inference(
    diffusion_pipeline: ExtendedDiffusionPipeline,
    index: int = 0,
    dataset_split: Split = Split.val,
    output_dir: str = "outputs/",
):
    """
    Generate an example action control prediction from the given diffusion pipeline.
    Args:
        diffusion_pipeline:diffusion pipeline with the checkpointed model.
        index: The index of the item to use for the prediction.
        dataset_split: The dataset split to use for the prediction.
        output_dir: The directory to save the output images to.

    """

    augment_sigma = DEFAULT_AUGMENT_SIGMA_LIST

    # Initialize data batch
    acd = ActionControlDiffusionDataset(subfolder="diffusion", split=dataset_split.value)
    data_batch = acd.collate_fn([acd[index]])
    data_batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in data_batch.items()}
    data_batch["inference_fwd"] = True

    # Generate the next frame from the diffusion pipeline.
    sample = diffusion_pipeline.generate_samples_from_batch(
        data_batch,
        guidance=7,
        state_shape=data_batch["video"].squeeze(0).size(),
        num_condition_t=data_batch["num_condition_t"],
        condition_latent=data_batch["video"],
        condition_video_augment_sigma_in_inference=augment_sigma,
    )

    # Post-processing and save image
    if torch.distributed.get_rank() == 0:
        sigma_data = 0.5

        # Initialize the tokenizer
        vae = create_tokenizer()

        predicted_sample = sample[:, :, 1:2]
        first_sample = data_batch["video"][:, :, 0:1]
        next_sample = data_batch["video"][:, :, 1:2]

        # Stack the input frame, ground truth next frame, and predicted next frame.
        all_frames = torch.cat([first_sample, next_sample, predicted_sample])

        # Decode the frames with the tokenizer into into RGB images together.
        decoded_image = vae.decode(all_frames.cuda() / sigma_data).squeeze(2).detach()
        grid = (1.0 + decoded_image).clamp(0, 2) / 2
        grid = (grid.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy().astype(np.uint8)

        # Create a single image with the three frames side by side.
        _, ax = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
        ax[0].imshow(grid[0])
        ax[1].imshow(grid[1])
        ax[2].imshow(grid[2])
        text_labels = ["Input Frame", "Next Frame (Ground Truth)", "Next Frame (Predicted)"]
        for i, text in enumerate(text_labels):
            ax[i].set_title(text)

        output_path = Path(output_dir) / f"diffusion_control_{dataset_split.value}_{index}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)


def main(
    nemo_ckpt: str,
    dataset_split: Split = Split.val,
    index: int = 0,
    output_dir: str = "outputs/",
    tensor_model_parallel_size: int = 4,
    context_parallel_size: int = 2,
    model_size: str = "14B",
):
    """
    Generate an example diffusion prediction from a post-trained checkpoint.

    Args:
        checkpoint_directory: A path to a post-trained checkpoint directory.
        dataset_split: The dataset split to use for the prediction.
        index: The index of the item to use for the prediction.
        output_dir: The directory to save the output images to.
        tensor_model_parallel_size: number of the tensor model parallelism
        context_parallel_size: context parallel size
        model_size: the size of the model. One of 7B or 14B.

    """

    diffusion_pipeline = setup_diffusion_pipeline(
        nemo_ckpt, tensor_model_parallel_size, context_parallel_size, model_size
    )
    run_diffusion_inference(diffusion_pipeline, index, dataset_split.val, output_dir)


if __name__ == "__main__":
    import typer

    typer.run(main)
