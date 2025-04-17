# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import pytorch_lightning as pl
import torch.distributed
import torch.utils.checkpoint
from einops import rearrange
from megatron.energon import DefaultTaskEncoder
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.diffusion.data.diffusion_energon_datamodule import DiffusionDataModule
from nemo.collections.physicalai.tokenizer.augmentors import TemporalRandomCrop
from nemo.collections.physicalai.tokenizer.data.augmentors.image.cropping import RandomCrop
from nemo.collections.physicalai.tokenizer.data.augmentors.image.normalize import Normalize
from nemo.collections.physicalai.tokenizer.data.utils import get_crop_size_info
from nemo.collections.physicalai.tokenizer.tokenizer_model import MASK_KEY, VIDEO_KEY, TokenizerModel
from nemo.lightning.io.mixin import IOMixin
from nemo.lightning.pytorch.callbacks import ModelCheckpoint, PreemptionCallback
from nemo.lightning.pytorch.optim.pytorch import PytorchOptimizerModule
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils.exp_manager import TimingCallback


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 100000000

    def __getitem__(self, idx):
        input_t = torch.randn([2, 3, 33, 256, 256], dtype=torch.bfloat16, device='cuda')
        mask_t = torch.ones_like(input_t, requires_grad=False, dtype=torch.bfloat16, device='cuda')
        return {VIDEO_KEY: input_t, MASK_KEY: mask_t}

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


class FakeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        seq_length: int = 2048,
        micro_batch_size: int = 1,
        global_batch_size: int = 8,
        num_workers: int = 1,
        pin_memory: bool = True,
        use_train_split_for_val: bool = False,
        task_encoder=None,
    ) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
        )

    def setup(self, stage: str = "") -> None:
        self._train_ds = FakeDataset()

    def train_dataloader(self):
        if not hasattr(self, "_train_ds"):
            self.setup()
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self):
        if not hasattr(self, "_train_ds"):
            self.setup()
        return self._create_dataloader(self._train_ds)

    def _create_dataloader(self, dataset, **kwargs):
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=dataset.collate_fn,
            **kwargs,
        )


class ImageTaskEncoder(DefaultTaskEncoder, IOMixin):
    """Image task encoder that crops and normalizes the image."""

    def __init__(
        self, *, encoded_sample_type=None, raw_batch_type=None, batch_type=None, crop_height=256, temporal_crop=49
    ):
        super().__init__(encoded_sample_type=encoded_sample_type, raw_batch_type=raw_batch_type, batch_type=batch_type)

        crop_sizes = get_crop_size_info(crop_height)
        self.spatial_crop = RandomCrop(input_keys=[VIDEO_KEY], args={"size": crop_sizes})
        self.temporal_crop = TemporalRandomCrop(temporal_crop)
        self.normalize = Normalize(input_keys=[VIDEO_KEY], args={"mean": 0.5, "std": 0.5})

    def encode_sample(self, sample):
        """
        Encode a single image sample by cropping and shifting its values.

        Args:
            sample: An image sample.

        Returns:
            The transformed image sample.
        """
        sample = super().encode_sample(sample)
        sample.image.frames = rearrange(sample.image.frames, 't c h w -> c t h w')

        mask_t = torch.ones_like(
            sample.image.frames, requires_grad=False, dtype=torch.bfloat16, device=sample.image.frames.device
        )

        data = {VIDEO_KEY: sample.image.frames, MASK_KEY: mask_t, 'aspect_ratio': '1,1'}
        data = self.spatial_crop(data)
        begin_index, end_index = self.temporal_crop(data[VIDEO_KEY].shape[1])
        data[VIDEO_KEY] = data[VIDEO_KEY][:, begin_index:end_index]
        data = self.normalize(data)
        data[VIDEO_KEY] = data[VIDEO_KEY].to(torch.bfloat16)
        return data


@run.cli.factory(target=llm.train)
def train_tokenizer() -> run.Partial:
    return run.Partial(
        llm.train,
        model=run.Config(
            TokenizerModel,
            jit_ckpt_pth=None,
            model="Cosmos-1.0-Tokenizer-CV8x8x8",
        ),
        data=run.Config(
            DiffusionDataModule,
            path=None,
            task_encoder=run.Config(ImageTaskEncoder),
            global_batch_size=8,
            micro_batch_size=1,
            num_workers=1,
        ),
        trainer=run.Config(
            nl.Trainer,
            devices='auto',
            num_nodes=int(os.environ.get('SLURM_NNODES', 1)),
            accelerator="gpu",
            strategy="ddp_find_unused_parameters_true",
            num_sanity_val_steps=0,
            limit_val_batches=1,
            val_check_interval=100,
            max_epochs=10000,
            precision='bf16',
            logger=WandbLogger(project='cosmos-tokenizer') if "WANDB_API_KEY" in os.environ else None,
            log_every_n_steps=1,
            use_distributed_sampler=False,
            callbacks=[
                run.Config(
                    ModelCheckpoint,
                    monitor='global_step',
                    filename='{global_step}',
                    every_n_train_steps=100,
                    save_top_k=3,
                    mode='max',
                    always_save_context=False,
                    save_context_on_train_end=False,
                ),
                run.Config(PreemptionCallback),
                run.Config(TimingCallback),
            ],
        ),
        optim=run.Config(
            PytorchOptimizerModule,
            optimizer_fn=run.Partial(
                torch.optim.AdamW,
                lr=1e-4,
                betas=(0.5, 0.999),
                eps=1e-8,
                weight_decay=0.01,
                fused=True,
            ),
        ),
        tokenizer=None,
        resume=run.Config(
            nl.AutoResume,
            resume_if_exists=True,
            resume_ignore_no_checkpoint=True,
            resume_past_end=True,
        ),
        model_transform=None,
    )


if __name__ == "__main__":
    run.cli.main(llm.train, default_factory=train_tokenizer)
