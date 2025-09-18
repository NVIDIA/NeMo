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

from nemo.collections.common.data.fallback import FallbackDataset
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.tokenizers import TokenizerSpec


class HybridSALMTDTDataModule(LightningDataModule):
    """
    A Lightning DataModule specialized for Hybrid SALM-TDT models.
    
    This DataModule handles the separation of speech and non-speech data,
    creating appropriate dataloaders for each type of data.
    
    The typical structure of the YAML config used to initialize this module looks like:
    
    .. code-block:: yaml
    
        data:
          train_ds:
            input_cfg: path/to/input_cfg.yaml
            num_workers: 2
            batch_size: 4
            speech_ratio: 0.5  # Optional: ratio of speech data in batches
            
          validation_ds:
            datasets:
              speech_val:
                cuts_path: path/to/speech_validation.cuts
              non_speech_val:
                cuts_path: path/to/non_speech_validation.cuts
            batch_size: 4
    
    Args:
        cfg: a DictConfig instance, typically corresponding to `data` namespace in YAML configs.
        tokenizer: a tokenizer instance, typically NeMo's AutoTokenizer wrapping HF's AutoTokenizer.
        dataset: a torch.utils.data.Dataset instance, expected to define __getitem__ that accepts
            a lhotse.CutSet. It converts metadata + raw data to a batch of PyTorch tensors.
    """

    def __init__(self, cfg, tokenizer: TokenizerSpec, dataset: torch.utils.data.Dataset, tdt_tokenizer: TokenizerSpec = None) -> None:
        super().__init__()
        self.cfg = cfg
        with open_dict(self.cfg):
            for k in ("validation_ds", "test_ds"):
                if k in self.cfg:
                    getattr(self.cfg, k).force_finite = True
                    # Don't force map dataset for hybrid SALM-TDT as it's designed for iterable datasets
                    # getattr(self.cfg, k).force_map_dataset = True
        self.tokenizer = tokenizer
        self.tdt_tokenizer = tdt_tokenizer
        self.dataset = dataset

    def train_dataloader(self):
        """Create training dataloader using the hybrid dataset."""
        if "train_ds" not in self.cfg:
            return None
        
        # Get speech ratio from config, default to 0.5
        speech_ratio = self.cfg.train_ds.get('speech_ratio', 0.5)
        
        # Update dataset with speech ratio if it supports it
        if hasattr(self.dataset, 'speech_ratio'):
            self.dataset.speech_ratio = speech_ratio
        
        
        return get_lhotse_dataloader_from_config(
            config=self.cfg.train_ds,
            global_rank=self._get_dp_rank(),
            world_size=self._get_world_size(),
            dataset=FallbackDataset(self.dataset),
            tokenizer=self.tokenizer,
            tdt_tokenizer=self.tdt_tokenizer,
        )

    def val_dataloader(self):
        """Create validation dataloader with separate speech and non-speech datasets."""
        if "validation_ds" not in self.cfg:
            return None
        cfg = self.cfg.validation_ds
        return self._build_test_dataloader(cfg)

    def test_dataloader(self):
        """Create test dataloader with separate speech and non-speech datasets."""
        if "test_ds" not in self.cfg:
            return None
        cfg = self.cfg.test_ds
        return self._build_test_dataloader(cfg)

    def _build_test_dataloader(self, cfg: DictConfig) -> torch.utils.data.DataLoader | CombinedLoader:
        """Build test/validation dataloader with support for separate speech and non-speech datasets."""
        # Single validation/test dataloader (legacy support)
        if "datasets" not in cfg:
            with open_dict(cfg):
                cfg.force_finite = True
                cfg.force_map_dataset = True
                # Don't force map dataset for hybrid SALM-TDT as it's designed for iterable datasets
                # cfg.force_map_dataset = True
            return get_lhotse_dataloader_from_config(
                config=cfg,
                global_rank=self._get_dp_rank(),
                world_size=self._get_world_size(),
                dataset=self.dataset,
                tokenizer=self.tokenizer,
                tdt_tokenizer=self.tdt_tokenizer,
            )

        # Multiple validation/test dataloaders with speech/non-speech separation
        base_cfg = cfg.copy()
        with open_dict(base_cfg):
            del base_cfg.datasets
        
        dloaders = {}
        for name, item in cfg.datasets.items():
            with open_dict(base_cfg):
                item = OmegaConf.merge(base_cfg, item)
            dloaders[name] = self._build_test_dataloader(item)
        
        return CombinedLoader(dloaders, mode="max_size")

    def _get_dp_rank(self):
        """Get data parallel rank."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if (
                hasattr(self.trainer, "model")
                and hasattr(self.trainer.model, "device_mesh")
                and self.trainer.model.device_mesh is not None
            ):  # model parallelism
                return self.trainer.model.device_mesh.get_coordinate()[0]
            else:
                return torch.distributed.get_rank()  # plain ol' DDP
        else:
            return 0  # 1 GPU

    def _get_world_size(self):
        """Get world size for distributed training."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if (
                hasattr(self.trainer, "model")
                and hasattr(self.trainer.model, "device_mesh")
                and self.trainer.model.device_mesh is not None
            ):  # model parallelism
                return self.trainer.model.device_mesh.shape[0]
            else:  # plain ol' DDP
                return torch.distributed.get_world_size()
        else:
            return 1  # 1 GPU
