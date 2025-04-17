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
from functools import partial

import nemo_run as run
from huggingface_hub import snapshot_download
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from platformdirs import user_cache_path
from torch.utils.data import DataLoader

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.diffusion.train import pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.physicalai.datasets.dataverse_dataset.driving_dataloader.alpamayo_dataloader import (
    InfiniteDataVerse,
)
from nemo.collections.physicalai.datasets.dataverse_dataset.driving_dataloader.config_dataverse import DATAVERSE_CONFIG
from nemo.collections.physicalai.datasets.dataverse_dataset.driving_dataloader.dataloader_utils import (
    dict_collation_fn,
)
from nemo.collections.physicalai.diffusion.post_training.multicamera.dit_multi_camera import (
    MultiCameraDiT7BConfig,
    MultiCameraDiTModel,
)
from nemo.lightning.pytorch.callbacks import ModelCheckpoint, PreemptionCallback
from nemo.lightning.pytorch.strategies.utils import RestoreConfig


class SimpleDataModule(MockDataModule):
    def __init__(self, *args, dataset=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset

    def setup(self, stage: str = "") -> None:
        self._train_ds = self.dataset()
        self._validation_ds = self.dataset()
        self._test_ds = self.dataset()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if not hasattr(self, "_train_ds"):
            self.setup()
        return self._create_dataloader(self._train_ds, num_workers=8)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if not hasattr(self, "_validation_ds"):
            self.setup()
        return self._create_dataloader(self._validation_ds, num_workers=0)

    def _create_dataloader(self, dataset, num_workers=0, **kwargs) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=dict_collation_fn,
            **kwargs,
        )


@run.cli.factory(target=llm.train)
def cosmos_multicamera_diffusion_7b_text2world_finetune() -> run.Partial:
    # Model setup
    recipe = pretrain()
    recipe.model = run.Config(
        MultiCameraDiTModel,
        config=run.Config(
            MultiCameraDiT7BConfig,
            n_cameras=6,
            camera_condition_dim=6,
            add_repeat_frame_embedding=True,
            # concat_traj_embedding=True,
            # traj_condition_dim=12,
            vae_path=snapshot_download("nvidia/Cosmos-1.0-Tokenizer-CV8x8x8"),
            pixel_chunk_duration=57,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        ),
    )
    recipe.trainer.strategy.ckpt_load_strictness = False
    recipe.trainer.val_check_interval = 100
    recipe.trainer.limit_val_batches = 1

    # Trainer setup
    recipe.trainer.max_steps = 30000

    # Optim setup
    recipe.optim.config.lr = 1e-4
    recipe.optim.config.weight_decay = 0.3
    recipe.optim.config.adam_eps = 1e-8
    recipe.optim.config.adam_beta1 = 0.9
    recipe.optim.config.adam_beta2 = 0.999
    recipe.optim.lr_scheduler = run.Config(
        nl.lr_scheduler.WarmupHoldPolicyScheduler, warmup_steps=1000, min_lr=1.0e-6, hold_steps=1e9
    )

    # Tensor / Sequence parallelism
    recipe.trainer.strategy.tensor_model_parallel_size = 8
    recipe.trainer.strategy.context_parallel_size = 1
    recipe.trainer.strategy.sequence_parallel = True
    recipe.trainer.strategy.ckpt_async_save = False
    recipe.trainer.strategy.save_ckpt_format = "torch_dist"

    # FSDP
    # recipe.trainer.strategy.ddp.with_megatron_fsdp_code_path = True
    # recipe.trainer.strategy.ddp.data_parallel_sharding_strategy = "MODEL_AND_OPTIMIZER_STATES"
    recipe.trainer.strategy.ddp.overlap_param_gather = False
    recipe.trainer.strategy.ddp.overlap_grad_reduce = False
    recipe.model.config.use_cpu_initialization = True

    recipe.trainer.callbacks = [
        run.Config(
            ModelCheckpoint,
            monitor="reduced_train_loss",
            dirpath="nemo_experiments/cosmos_multicamera_diffusion_7b_text2world_finetune/default/experiment_dir",
            filename="{epoch}-{step}",
            every_n_train_steps=100,
            save_top_k=5,
            always_save_context=True,
            save_weights_only=False,
        ),
        run.Config(PreemptionCallback),
    ]

    # Data setup
    recipe.data = run.Config(
        SimpleDataModule,
        micro_batch_size=1,
        global_batch_size=1,
        dataset=partial(InfiniteDataVerse, **DATAVERSE_CONFIG["alpamayo_v2_traj_qwen_24fps_6_cameras_frame_repeat"]),
    )

    recipe.resume = run.Config(
        nl.AutoResume,
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
        resume_past_end=True,
        resume_from_directory="nemo_experiments/cosmos_multicamera_diffusion_7b_text2world_finetune/default/experiment_dir",
    )

    # Checkpoint load
    recipe.resume.restore_config = run.Config(
        RestoreConfig,
        path=os.path.join(
            snapshot_download("nvidia/Cosmos-1.0-Diffusion-7B-Text2World", allow_patterns=["nemo/*"]), "nemo"
        ),  # path to diffusion model checkpoint
        load_model_state=True,
        load_optim_state=False,
        load_artifacts=False,
    )
    return recipe


@run.cli.factory(target=llm.train)
def cosmos_multicamera_diffusion_7b_text2world_overfit() -> run.Partial:
    recipe = cosmos_multicamera_diffusion_7b_text2world_finetune()

    # Path to overfit data
    dataset_path = str(user_cache_path("AV-V2.2"))
    # dataset_path = "./overfit_data"
    tar_dirs_training = [f"{dataset_path}/trainv2-2-chunk-00"]

    # Custom Dataset Config
    ds_config = DATAVERSE_CONFIG["alpamayo_v2_traj_qwen_24fps_6_cameras_frame_repeat"]
    ds_config["dataset_cfg"]["params"]["tar_dirs"] = tar_dirs_training
    ds_config["dataset_cfg"]["params"]["uuid_dirs"] = None
    ds_config["dataset_cfg"]["params"]["t5_dirs"] = f"{dataset_path}/alpamayo_caption_t5/qwen_t5_tars/"

    # Data setup
    recipe.data = run.Config(
        SimpleDataModule,
        micro_batch_size=1,
        global_batch_size=1,
        dataset=partial(InfiniteDataVerse, **ds_config),
    )

    return recipe


@run.cli.factory(target=llm.train)
def cosmos_multicamera_diffusion_7b_text2world_finetune_w_traj() -> run.Partial:
    # Model setup
    recipe = cosmos_multicamera_diffusion_7b_text2world_finetune()

    recipe.model.config.concat_traj_embedding = True
    recipe.model.config.traj_condition_dim = 12

    return recipe


@run.cli.factory(target=llm.train)
def cosmos_multicamera_diffusion_7b_text2world_finetune_w_traj_debug() -> run.Partial:
    # Model setup
    recipe = cosmos_multicamera_diffusion_7b_text2world_finetune_w_traj()

    recipe.model.config.concat_traj_embedding = True
    recipe.model.config.traj_condition_dim = 12
    recipe.model.config.num_layers = 1
    recipe.resume.restore_config = None
    return recipe


@run.cli.factory(target=llm.train)
def cosmos_multicamera_diffusion_7b_image2world_finetune() -> run.Partial:
    # Model setup
    recipe = cosmos_multicamera_diffusion_7b_text2world_finetune()

    recipe.model = run.Config(
        MultiCameraDiTModel,
        config=run.Config(
            MultiCameraDiT7BConfig,
            n_cameras=6,
            camera_condition_dim=6,
            add_repeat_frame_embedding=True,
            # concat_traj_embedding=True,
            # traj_condition_dim=12,
            vae_path=snapshot_download("nvidia/Cosmos-1.0-Tokenizer-CV8x8x8"),
            pixel_chunk_duration=57,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        ),
    )

    # Checkpoint load
    recipe.resume.restore_config.path = os.path.join(
        snapshot_download("nvidia/Cosmos-1.0-Diffusion-7B-Video2World", allow_patterns=["nemo/*"]), "nemo"
    )  # path to diffusion model checkpoint

    return recipe


@run.cli.factory(target=llm.train)
def cosmos_multicamera_diffusion_7b_image2world_finetune_w_traj() -> run.Partial:
    # Model setup
    recipe = cosmos_multicamera_diffusion_7b_image2world_finetune()

    recipe.model.config.concat_traj_embedding = True
    recipe.model.config.traj_condition_dim = 32

    return recipe


if __name__ == "__main__":
    run.cli.main(llm.train, default_factory=cosmos_multicamera_diffusion_7b_text2world_finetune)
