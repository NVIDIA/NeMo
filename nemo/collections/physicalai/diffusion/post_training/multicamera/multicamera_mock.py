# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import nemo_run as run
import torch
from huggingface_hub import snapshot_download

from nemo.collections import llm
from nemo.collections.diffusion.datamodule import DiTDataModule
from nemo.collections.diffusion.train import pretrain
from nemo.collections.physicalai.diffusion.post_training.multicamera.dit_multi_camera import MultiCameraDiT7BConfig
from nemo.lightning.pytorch.strategies.utils import RestoreConfig


class MultiCameraDiTVideoLatentMockDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, seq_len=21760):
        self.length = num_samples if num_samples > 0 else 1 << 32
        self.seq_len = seq_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        t = 16
        h = 34
        w = 40
        seq_len = t * h * w
        video_latent = torch.randn(16, t, h, w).to(dtype=torch.uint8)
        loss_mask = torch.ones(seq_len, dtype=torch.bfloat16)
        noise_latent = torch.rand_like(video_latent, dtype=torch.bfloat16)
        timesteps = torch.randn(1, dtype=torch.bfloat16)
        text_embedding = torch.randn(512, 1024, dtype=torch.bfloat16)

        return {
            "video": video_latent,
            "noise_latent": noise_latent,
            "timesteps": timesteps,
            "t5_text_embeddings": text_embedding,
            "t5_text_mask": torch.ones(512, dtype=torch.bfloat16),
            "image_size": torch.tensor([[34, 40, 34, 40]] * 1, dtype=torch.bfloat16),
            "fps": torch.tensor([30] * 1, dtype=torch.bfloat16),
            "num_frames": torch.tensor([16] * 1, dtype=torch.bfloat16),
            "padding_mask": torch.zeros((1, 1, 34, 40), dtype=torch.bfloat16),
            "loss_mask": loss_mask,
        }

    def _collate_fn(self, batch):
        """
        A default implementation of a collation function.
        Users should override this method to define custom data loaders.
        """
        return torch.utils.data.dataloader.default_collate(batch)

    def collate_fn(self, batch):
        """Method that user pass as functor to DataLoader.

        The method optionally performs neural type checking and add types to the outputs.

        Please note, subclasses of Dataset should not implement `input_types`.

        # Usage:
        dataloader = torch.utils.data.DataLoader(
                ....,
                collate_fn=dataset.collate_fn,
                ....
        )

        Returns
        -------
            Collated batch, with or without types.
        """
        return self._collate_fn(batch)


@run.cli.factory(target=llm.train)
def cosmos_multicamera_diffusion_7b_text2world_finetune() -> run.Partial:
    # Model setup
    recipe = pretrain()
    recipe.model.config = run.Config(MultiCameraDiT7BConfig)
    recipe.trainer.strategy.ckpt_load_strictness = False

    # Trainer setup
    recipe.trainer.max_steps = 1000
    recipe.optim.config.lr = 1e-6

    # Tensor / Sequence parallelism
    recipe.trainer.strategy.tensor_model_parallel_size = 8
    recipe.trainer.strategy.sequence_parallel = True
    recipe.trainer.strategy.ckpt_async_save = False

    # FSDP
    recipe.trainer.strategy.ddp.with_megatron_fsdp_code_path = True
    recipe.trainer.strategy.ddp.data_parallel_sharding_strategy = "MODEL_AND_OPTIMIZER_STATES"
    recipe.trainer.strategy.ddp.overlap_param_gather = True
    recipe.trainer.strategy.ddp.overlap_grad_reduce = True
    recipe.model.config.use_cpu_initialization = True

    # Data setup
    recipe.data = DiTDataModule(
        dataset=MultiCameraDiTVideoLatentMockDataset, path=1000, micro_batch_size=1, global_batch_size=1
    )

    # Checkpoint load
    recipe.resume.restore_config = run.Config(RestoreConfig, load_artifacts=False)
    recipe.resume.restore_config.path = os.path.join(
        snapshot_download("nvidia/Cosmos-1.0-Diffusion-7B-Text2World", allow_patterns=["nemo/*"]), "nemo"
    )  # path to diffusion model checkpoint
    recipe.resume.resume_if_exists = False

    # Directory to save checkpoints / logs
    recipe.log.log_dir = "nemo_experiments/cosmos_multicamera_diffusion_7b_text2world_finetune"

    return recipe


if __name__ == "__main__":
    run.cli.main(llm.train, default_factory=cosmos_multicamera_diffusion_7b_text2world_finetune)
