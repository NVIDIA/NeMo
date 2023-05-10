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
import numpy as np
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
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceConfig, get_confidence_aggregation_bank
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.mixins import AccessMixin
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, LogprobsType, NeuralType, SpectrogramType
from nemo.utils import logging, model_utils

__all__ = ['ConfidenceEnsembleModel']


class ConfidenceEnsembleModel(ModelPT):
    def __init__(
        self, cfg: DictConfig, trainer: 'Trainer' = None, models: Optional[List[str]] = None,
    ):
        # TODO: put models inside the config
        super().__init__(cfg=cfg, trainer=trainer)

        # either we load all models from ``models`` init parameter
        # or all of them are specified in the config alongside the num_models key
        #
        # ideally, we'd like to directly store all models in a list, but that
        # is not currently supported by the submodule logic
        # so to access all the models, we do something like
        #
        # for model_idx in range(self.num_models):
        #    model = getattr(self, f"model{model_idx}")

        if 'num_models' in self.cfg:
            self.num_models = self.cfg.num_models
            for idx in range(self.num_models):
                cfg_field = f"model{idx}"
                model_cfg = self.cfg[cfg_field]
                model_class = model_utils.import_class_by_path(model_cfg['target'])
                self.register_nemo_submodule(
                    name=cfg_field, config_field=cfg_field, model=model_class(model_cfg, trainer=trainer),
                )
        else:
            self.num_models = len(models)
            OmegaConf.set_struct(self.cfg, False)
            self.cfg.num_models = self.num_models
            OmegaConf.set_struct(self.cfg, True)
            for idx, model in enumerate(models):
                cfg_field = f"model{idx}"
                # TODO: map location cpu
                if model.endswith(".nemo"):
                    self.register_nemo_submodule(
                        name=cfg_field, config_field=cfg_field, model=ASRModel.restore_from(model, trainer=trainer),
                    )
                else:
                    self.register_nemo_submodule(
                        cfg_field, config_field=cfg_field, model=ASRModel.from_pretrained(model),
                    )

        model_selection_block_path = self.register_artifact("model_selection_block", cfg.model_selection_block)
        self.model_selection_block = joblib.load(model_selection_block_path)
        self.confidence = ConfidenceConfig(**self.cfg.confidence)

        # making sure each model has correct confidence settings in the decoder strategy
        for model_idx in range(self.num_models):
            model = getattr(self, f"model{model_idx}")
            decoding_cfg = model.cfg.decoding
            decoding_cfg.confidence_cfg = self.confidence
            # TODO: is there a way to handle hybrid model change flexibly here?
            model.change_decoding_strategy(decoding_cfg)

    # TODO: hybrid later (no switch from ctc to rnnt)

    def list_available_models(self):
        pass

    def setup_training_data(self):
        pass

    def setup_validation_data(self):
        pass

    def change_attention_model(self, *args, **kwargs):
        """Pass-through to the ensemble models."""
        for model_idx in range(self.num_models):
            getattr(self, f"model{model_idx}").change_attention_model(*args, **kwargs)

    def change_decoding_strategy(self, decoding_cfg: DictConfig = None, **kwargs):
        """Pass-through to the ensemble models.

        The only change here is that we always require frame-confidence
        to be returned.
        """
        decoding_cfg.confidence_cfg = self.confidence
        for model_idx in range(self.num_models):
            getattr(self, f"model{model_idx}").change_decoding_strategy(decoding_cfg, **kwargs)

    # TODO: keep a single class, use common arguments explicitly, any optional model-specific go to kwargs and inspect later
    # TODO: assume only common arguments for now
    # TODO: return_hypotheses always True
    @torch.no_grad()
    def transcribe(  # TODO: rnnt takes different parameters?
        self,
        *args,  # TODO: remove args
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
        # TODO: lots of duplicate code with building ensemble script
        aggr_func = get_confidence_aggregation_bank()[self.confidence.aggregation]
        confidences = []
        all_transcriptions = []
        for model_idx in range(self.num_models):
            model = getattr(self, f"model{model_idx}")
            transcriptions = model.transcribe(*args, **kwargs)

            model_confidences = []
            for transcription in transcriptions:
                if isinstance(transcription.frame_confidence[0], list):
                    # NeMo Transducer API returns list of lists for confidences
                    conf_values = [conf_value for confs in transcription.frame_confidence for conf_value in confs]
                else:
                    conf_values = transcription.frame_confidence
                model_confidences.append(aggr_func(conf_values))
            confidences.append(model_confidences)
            all_transcriptions.append(transcriptions)

        # transposing with zip(*list)
        features = np.array(list(zip(*confidences)))
        model_indices = self.model_selection_block.predict(features)
        final_transcriptions = []
        for transcrption_idx in range(len(all_transcriptions[0])):
            final_transcriptions.append(all_transcriptions[model_indices[transcrption_idx]][transcrption_idx])

        return final_transcriptions
