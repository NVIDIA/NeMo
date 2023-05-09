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

import copy
import json
import os
from math import ceil
from typing import Dict, List, Optional, Union

import joblib
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from tqdm.auto import tqdm

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import AudioToCharDALIDataset, DALIOutputs
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer import WER, CTCDecoding, CTCDecodingConfig
from nemo.collections.asr.models.asr_model import ASRModel, ExportableEncDecModel
from nemo.collections.asr.parts.mixins import ASRModuleMixin, InterCTCMixin
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.mixins import AccessMixin
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, LogprobsType, NeuralType, SpectrogramType
from nemo.utils import logging

__all__ = ['ConfidenceEnsembleModel']


class ConfidenceEnsembleModel(ModelPT):
    def __init__(
        self,
        cfg: DictConfig,
        trainer: 'Trainer' = None,
        models: Optional[List[str]] = None,
        model_selection_block_path: Optional[str] = None,
    ):
        super().__init__(cfg=cfg, trainer=trainer)

        # either we load all models from ``models`` init parameter
        # or all of them are specified in the config alongside the num_models key
        # ideally, we'd like to directly store all models in a list, but that
        # is not currently supported by the submodule logic

        if 'num_models' in self.cfg:
            self.num_models = self.cfg.num_models
            for idx, model_cfg in enumerate(self.cfg.models):
                cfg_field = f"model{idx}"
                self.register_nemo_submodule(
                    name=cfg_field, config_field=cfg_field, model=ASRModel(model_cfg, trainer=trainer),
                )
        else:
            self.num_models = len(models)
            self.cfg.num_models = self.num_models
            for idx, model in enumerate(models):
                cfg_field = f"model{idx}"
                if model.endswith(".nemo"):
                    self.register_nemo_submodule(
                        name=cfg_field, config_field=cfg_field, model=ASRModel.restore_from(model, trainer=trainer),
                    )
                else:
                    self.register_nemo_submodule(
                        cfg_field, config_field=cfg_field, model=ASRModel.from_pretrained(model),
                    )

        # registering the model selection block as an artifact
        if model_selection_block_path:
            self.register_artifact("model_selection_block", model_selection_block_path)
            self.model_selection_block = joblib.load(model_selection_block_path)
        else:  # or loading from checkpoint if not specified
            model_selection_block_path = self.register_artifact("model_selection_block", cfg.model_selection_block)
            self.model_selection_block = joblib.load(model_selection_block_path)

    def list_available_models(self):
        pass

    def setup_training_data(self):
        pass

    def setup_validation_data(self):
        pass

    @torch.no_grad()
    def transcribe(  # TODO: rnnt takes different parameters?
        self,
        *args,
        **kwargs,
        # paths2audio_files: List[str],
        # batch_size: int = 4,
        # logprobs: bool = False,
        # return_hypotheses: bool = False,
        # num_workers: int = 0,
        # channel_selector: Optional[ChannelSelectorType] = None,
        # augmentor: DictConfig = None,
        # verbose: bool = True,
    ) -> List[str]:
        """Confidence-ensemble transcribe method.

        Consists of the following steps:

            1. Run all models (TODO: in parallel)
            2. Compute confidence for each model
            3. Use logistic regression to pick the "most confident" model
            4. Return the output of that model
        """
        hypotheses = []
        for model in self.models:
            hypotheses.append(model.transcribe(*args, **kwargs))

        from IPython import embed

        embed()
