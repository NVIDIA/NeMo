# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import itertools
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from nemo.core import ModelPT
from nemo.utils import logging
from nemo.utils.decorators import experimental
from nemo.collections.tts.models.fastpitch import TextTokenizer, TextTokenizerConfig, G2PConfig, FastPitchModel


@experimental
class VoiceboxModel(ModelPT):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super.__init__(cfg, trainer)

    def prepare_data(self) -> None:
        """ Pytorch Lightning hook.

        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#prepare-data

        The following code is basically for transcribed LibriLight.
        """
        from lhotse.recipes.utils import manifests_exist

        os.makedirs(self._cfg.manifests_dir, exist_ok=True)
        for subset in self.subsets:
            if not manifests_exist(subset, self._cfg.manifests_dir, ["cuts"], "librilight"):
                logging.info(f"Downloading {subset} subset.")
                os.system(f"wget -P {self._cfg.manifests_dir} -c https://huggingface.co/datasets/pkufool/libriheavy/resolve/main/libriheavy_cuts_{subset}.jsonl.gz")
            else:
                logging.info(f"Skipping download, {subset} subset exists.")

    _get_default_text_tokenizer_conf = FastPitchModel._get_default_text_tokenizer_conf
    _setup_normalizer = FastPitchModel._setup_normalizer
    _setup_tokenizer = FastPitchModel._setup_tokenizer
    

    def _setup_dataloader_from_config(self, config: Optional[Dict]) -> DataLoader[Any]:
        """Modified from https://github.com/pzelasko/NeMo/blob/feature/lhotse-integration/nemo/collections/asr/models/hybrid_rnnt_ctc_bpe_models.py#L129
        """
        from nemo.collections.asr.data.lhotse.dataloader import get_lhotse_dataloader_from_config
        from nemo.collections.tts.data.text_to_speech_lhotse import LhotseTextToSpeechDataset

        assert config.get("use_lhotse")

        # Note:
        #    Lhotse Dataset only maps CutSet -> batch of tensors, but does not actually
        #    contain any data or meta-data; it is passed to it by a Lhotse sampler for
        #    each sampler mini-batch.
        return get_lhotse_dataloader_from_config(
            config,
            global_rank=self.global_rank,
            world_size=self.world_size,
            dataset=LhotseTextToSpeechDataset(
                tokenizer=self.tokenizer, noise_cuts=config.get("lhotse", {}).get("noise_cuts")
            ),
        )
    
    def setup_training_data(self, train_data_config: DictConfig | Dict):
        return super().setup_training_data(train_data_config)