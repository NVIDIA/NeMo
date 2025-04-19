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

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from cosmos1.models.autoregressive.nemo.inference.general import MockMCoreTokenizer
from cosmos1.models.autoregressive.nemo.inference.inference_controller import CosmosActionGenerationController
from cosmos1.models.autoregressive.nemo.post_training.action_control.action_control_dataset import ActionControlDataset
from cosmos1.models.autoregressive.nemo.post_training.action_control.prepare_dataset import Split, create_tokenizer
from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.engines.mcore_engine import MCoreEngine

from nemo import lightning as nl
from nemo.collections.llm.inference.base import _setup_trainer_and_restore_model
from nemo.lightning import io
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir

LATENT_SHAPE = [1, 30, 40]  # For the nvidia/Cosmos-1.0-Tokenizer-DV8x16x16


BOV_TOKEN = 64000
from einops import rearrange


def create_trainer():
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        setup_optimizers=False,
        store_optimizer_states=False,
    )

    trainer = nl.Trainer(
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        strategy=strategy,
        num_sanity_val_steps=0,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
        ),
    )

    return trainer


def main(
    checkpoint_directory: str,
    temperature: float = 1,
    top_p: float = 0.0,
    dataset_split: Split = Split.val,
    index: int = 0,
    output_dir: str = "outputs/",
):
    """Generate an example action control prediction from a post-trained checkpoint.

    Args:
        checkpoint_directory: A path to a post-trained checkpoint directory.
        temperature: The temperature to use for the prediction.
        top_p: The top-p value to use for the prediction.
        dataset_split: The dataset split to use for the prediction.
        index: The index of the item to use for the prediction.
        output_dir: The directory to save the output images to.
    """

    num_tokens_to_generate = math.prod(LATENT_SHAPE)

    # Create the mcore inference engine from the restored checkpoint to handle efficient caching of
    # repeated auto-regressive token generation.
    model: io.TrainerContext = io.load_context(path=ckpt_to_context_subdir(checkpoint_directory), subpath="model")
    trainer = create_trainer()
    _setup_trainer_and_restore_model(path=Path(checkpoint_directory), trainer=trainer, model=model)
    inference_wrapped_model = model.get_inference_wrapper(torch.bfloat16, inference_batch_times_seqlen_threshold=1000)
    action_generation_controller = CosmosActionGenerationController(
        inference_wrapped_model=inference_wrapped_model, tokenizer=MockMCoreTokenizer()
    )
    mcore_engine = MCoreEngine(text_generation_controller=action_generation_controller, max_batch_size=1)

    # Create the dataset and sample a single item from it.
    dataset = ActionControlDataset(subfolder="autoregressive", split=dataset_split.value, shuffle=False)
    sample = dataset[index]
    input_frame = torch.cat([torch.tensor([BOV_TOKEN]), sample["current_frame"].flatten()])
    prompts: list[list[int]] = [input_frame.tolist()]
    actions = sample["action"].unsqueeze(0).cuda()  # shape([1, 7]), dtype=torch.bfloat16

    # Generate the next frame by passing in the current frame and the action to take and
    # subsequently generating prod(LATENT_SHAPE) tokens. Since we're only using a batch size of 1
    # here, we take the first item.
    results = mcore_engine.generate(
        prompts=prompts,
        add_BOS=False,
        encoder_prompts=actions,  # type: ignore
        common_inference_params=CommonInferenceParams(
            num_tokens_to_generate=num_tokens_to_generate,
            top_p=top_p,
            temperature=temperature,
        ),
    )[0]

    # Reshape the generated tokens to the correct shape for an output frame.
    predicted_frame = results.generated_tokens.reshape(1, 30, 40).detach().cpu()

    # Stack the input frame, ground truth next frame, and predicted next frame and decode them into RGB images together.
    all_frames = torch.stack([sample["current_frame"], sample["next_frame"], predicted_frame])
    video_tokenizer = create_tokenizer()
    decoded_image = video_tokenizer.decode(all_frames.cuda(), pixel_chunk_duration=None).detach()
    decoded_image = (decoded_image + 1) * 127.5
    decoded_image = rearrange(decoded_image, "b c 1 h w -> b h w c").to(device="cpu", dtype=torch.uint8)

    # Create a single image with the three frames side by side.
    text_labels = ["Input Frame", "Next Frame (Ground Truth)", "Next Frame (Predicted)"]
    _, ax = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
    ax[0].imshow(decoded_image[0])
    ax[1].imshow(decoded_image[1])
    ax[2].imshow(decoded_image[2])
    for i, text in enumerate(text_labels):
        ax[i].set_title(text)
    output_path = Path(output_dir) / f"action_control_ar_{dataset_split.value}_{index}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)


if __name__ == "__main__":
    import typer

    typer.run(main)
