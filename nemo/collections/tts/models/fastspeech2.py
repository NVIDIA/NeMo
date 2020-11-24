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

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from hydra.utils import instantiate
from torch import nn
from omegaconf import MISSING, DictConfig, OmegaConf, open_dict

from nemo.collections.asr.parts import parsers
from nemo.collections.tts.models.base import SpectrogramGenerator, TextToWaveform
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.utils import logging


@dataclass
class PreprocessorParams:
    pad_value: float = MISSING


@dataclass
class Preprocessor:
    cls: str = MISSING
    params: PreprocessorParams = PreprocessorParams()


@dataclass
class FastSpeech2Config:
    fastspeech2: Dict[Any, Any] = MISSING
    preprocessor: Preprocessor = Preprocessor()
    # TODO: may need more params
    train_ds: Optional[Dict[Any, Any]] = None
    validation_ds: Optional[Dict[Any, Any]] = None


class FastSpeech2Model(SpectrogramGenerator):
    """FastSpeech 2 model used to convert between text (phonemes) and mel-spectrograms."""

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        super().__init__(cfg=cfg, trainer=trainer)

        schema = OmegaConf.structured(FastSpeech2Config)
        # ModelPT ensures that cfg is a DictConfig, but do this second check in case ModelPT changes
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
        # Ensure passed cfg is compliant with schema
        OmegaConf.merge(cfg, schema)

    # @property
    # def input_types(self):
    #     return {"text": NeuralType(('B', 'T'), TokenIndex()), "text_lengths": NeuralType(('B'), LengthsType())}

    # @property
    # def output_types(self):
    #     # May need to condition on OperationMode.training vs OperationMode.validation
    #     pass

    @typecheck()
    def forward(self, *, text, text_lens):
        pass

    # @typecheck(
    #     input_types={"text": NeuralType(('B', 'T'), TokenIndex()), "text_lengths": NeuralType(('B'), LengthsType())},
    #     output_types={"spec": NeuralType(('B', 'D', 'T'), MelSpectrogramType())},
    # )
    def generate_spectrogram(self, tokens: torch.Tensor) -> torch.Tensor:
        # TODO
        pass

    def parse(self, str_input: str) -> torch.Tensor:
        # TODO
        pass

    def __setup_dataloader_from_config(self, cfg, shuffle_should_be: bool = True, name: str = "train"):
        if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
            raise ValueError(f"No dataset for {name}")
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloder_params for {name}")
        if shuffle_should_be:
            if 'shuffle' not in cfg.dataloader_params:
                logging.warning(
                    f"Shuffle should be set to True for {self}'s {name} dataloader but was not found in its "
                    "config. Manually setting to True"
                )
                with open_dict(cfg.dataloader_params):
                    cfg.dataloader_params.shuffle = True
            elif not cfg.dataloader_params.shuffle:
                logging.error(f"The {name} dataloader for {self} has shuffle set to False!!!")
        elif not shuffle_should_be and cfg.dataloader_params.shuffle:
            logging.error(f"The {name} dataloader for {self} has shuffle set to True!!!")

        dataset = instantiate(cfg.dataset)
        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def setup_training_data(self, cfg):
        self._train_dl = self.__setup_dataloader_from_config(cfg)

    def setup_validation_data(self, cfg):
        pass


# TODO: FastSpeech 2s (TextToWaveform)
