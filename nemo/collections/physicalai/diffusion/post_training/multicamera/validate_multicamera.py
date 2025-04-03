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

from functools import partial
import os
from nemo.collections.diffusion.models.model import DiTModel, dit_data_step, dynamic_import
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.physicalai.datasets.dataverse_dataset.driving_dataloader.alpamayo_dataloader import DrivingVideoDataLoader, InfiniteDataVerse, get_driving_dataset
from nemo.collections.physicalai.datasets.dataverse_dataset.driving_dataloader.config_dataverse import DATAVERSE_CONFIG
from nemo.collections.physicalai.diffusion.post_training.multicamera.dit_multi_camera import MultiCameraDiT7BConfig, MultiCameraDiTModel, VideoExtendMultiCameraDiTCrossAttentionModel7B
import torch
from huggingface_hub import snapshot_download
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.diffusion.datamodule import DiTDataModule
from nemo.collections.diffusion.train import pretrain
from nemo.lightning.pytorch.strategies.utils import RestoreConfig
import nemo_run as run
from torch.utils.data import DataLoader
from nemo.collections.physicalai.datasets.dataverse_dataset.driving_dataloader.dataloader_utils import dict_collation_fn
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from huggingface_hub import snapshot_download
from einops import rearrange
from multicamera import cosmos_multicamera_diffusion_7b_text2world_finetune, SimpleDataModule

@run.cli.factory(target=llm.validate)
def cosmos_multicamera_diffusion_7b_text2world_validate() -> run.Partial:
    recipe = cosmos_multicamera_diffusion_7b_text2world_finetune()
    
    # Checkpoint load
    recipe.resume.restore_config.path = os.path.join(
        snapshot_download("nvidia/Cosmos-1.0-Diffusion-7B-Video2World", allow_patterns=["nemo/*"]), "nemo"
    )  # path to diffusion model checkpoint
    
    return run.Partial(
        llm.validate,
        model=recipe.model,
        data=recipe.data,
        trainer=recipe.trainer,
        log=recipe.log,
        optim=recipe.optim,
        tokenizer=None,
        resume=recipe.resume,
        model_transform=None,
    )

if __name__ == "__main__":
    run.cli.main(llm.validate, default_factory=cosmos_multicamera_diffusion_7b_text2world_validate)
