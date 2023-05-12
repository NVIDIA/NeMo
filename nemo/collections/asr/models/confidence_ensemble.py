# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer

from nemo.collections.asr.models.asr_model import ASRModel
from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel
from nemo.collections.asr.parts.utils.asr_confidence_utils import (
    ConfidenceConfig,
    ConfidenceMethodConfig,
    get_confidence_aggregation_bank,
    get_confidence_measure_bank,
)
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.core.classes import ModelPT
from nemo.utils import model_utils


@dataclass
class ConfidenceSpec:
    exclude_blank: bool
    aggregation: str
    confidence_type: str
    alpha: float

    def to_confidence_config(self) -> ConfidenceConfig:
        if self.confidence_type == 'max_prob':
            name = 'max_prob'
            entropy_type = 'tsallis'  # can be any
            entropy_norm = 'lin'  # can be any
        else:
            name, entropy_type, entropy_norm = self.confidence_type.split("_")
        return ConfidenceConfig(
            exclude_blank=self.exclude_blank,
            aggregation=self.aggregation,
            method_cfg=ConfidenceMethodConfig(
                name=name, entropy_type=entropy_type, temperature=self.alpha, entropy_norm=entropy_norm,
            ),
        )


def get_filtered_logprobs(transcription, exclude_blank):
    if isinstance(transcription.alignments, list):  # Transducer
        filtered_logprobs = []
        for alignment in transcription.alignments:
            for align_elem in alignment:
                if exclude_blank and align_elem[1].item() != align_elem[0].shape[-1] - 1:
                    filtered_logprobs.append(align_elem[0])
                filtered_logprobs.append(align_elem[0])
        if not filtered_logprobs:  # for the edge-case of all blanks
            filtered_logprobs.append(align_elem[0])
        filtered_logprobs = torch.stack(filtered_logprobs)
        if torch.cuda.is_available():  # by default logprobs are placed on cpu in nemo
            filtered_logprobs = filtered_logprobs.cuda()
    else:  # CTC
        logprobs = transcription.y_sequence
        if torch.cuda.is_available():  # by default logprobs are placed on cpu in nemo
            logprobs = logprobs.cuda()
        if exclude_blank:  # filtering blanks
            labels = logprobs.argmax(dim=-1)
            filtered_logprobs = logprobs[labels != logprobs.shape[1] - 1]
        else:
            filtered_logprobs = logprobs
    return filtered_logprobs


def compute_confidence(transcription, confidence_cfg: ConfidenceConfig):
    filtered_logprobs = get_filtered_logprobs(transcription, confidence_cfg.exclude_blank)
    vocab_size = filtered_logprobs.shape[1]
    aggr_func = get_confidence_aggregation_bank()[confidence_cfg.aggregation]
    if confidence_cfg.method_cfg.name == "max_prob":
        conf_type = "max_prob"
        alpha = 1.0
    else:
        conf_type = f"entropy_{confidence_cfg.method_cfg.entropy_type}_{confidence_cfg.method_cfg.entropy_norm}"
        alpha = confidence_cfg.method_cfg.temperature
    conf_func = get_confidence_measure_bank()[conf_type]

    conf_value = aggr_func(conf_func(filtered_logprobs, v=vocab_size, t=alpha)).cpu().item()
    return conf_value


class ConfidenceEnsembleModel(ModelPT):
    def __init__(
        self, cfg: DictConfig, trainer: 'Trainer' = None,
    ):
        super().__init__(cfg=cfg, trainer=trainer)

        # either we load all models from ``load_models`` cfg parameter
        # or all of them are specified in the config as modelX alongside the num_models key
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
            self.num_models = len(cfg.load_models)
            with open_dict(self.cfg):
                self.cfg.num_models = self.num_models
            for idx, model in enumerate(cfg.load_models):
                cfg_field = f"model{idx}"
                if model.endswith(".nemo"):
                    self.register_nemo_submodule(
                        name=cfg_field,
                        config_field=cfg_field,
                        model=ASRModel.restore_from(model, trainer=trainer, map_location="cpu"),
                    )
                else:
                    self.register_nemo_submodule(
                        cfg_field, config_field=cfg_field, model=ASRModel.from_pretrained(model, map_location="cpu"),
                    )

        # registering model selection block - this is expected to be a joblib-saved
        # pretrained sklearn pipeline containing standardization + logistic regression
        # trained to predict "most-confident" model index from the confidence scores of all models
        model_selection_block_path = self.register_artifact("model_selection_block", cfg.model_selection_block)
        self.model_selection_block = joblib.load(model_selection_block_path)
        self.confidence_cfg = ConfidenceConfig(**self.cfg.confidence)

        # making sure each model has correct temperature setting in the decoder strategy
        for model_idx in range(self.num_models):
            model = getattr(self, f"model{model_idx}")
            # for now we assume users are direclty responsible for matching
            # decoder type when building ensemlbe with inference type
            # TODO: add automatic checks for errors
            if isinstance(model, EncDecHybridRNNTCTCModel):
                self.update_decoding_parameters(model.cfg.decoding)
                model.change_decoding_strategy(model.cfg.decoding, decoder_type="rnnt")
                self.update_decoding_parameters(model.cfg.aux_ctc.decoding)
                model.change_decoding_strategy(model.cfg.aux_ctc.decoding, decoder_type="ctc")
            else:
                self.update_decoding_parameters(model.cfg.decoding)
                model.change_decoding_strategy(model.cfg.decoding)

    def update_decoding_parameters(self, decoding_cfg):
        """Updating temperature/preserve_alignment/preserve_frame_confidence parameters of the config."""
        with open_dict(decoding_cfg):
            decoding_cfg.temperature = self.cfg.temperature
            decoding_cfg.preserve_alignments = True
            if 'confidence_cfg' in decoding_cfg:
                decoding_cfg.confidence_cfg.preserve_frame_confidence = True
            else:
                decoding_cfg.confidence_cfg = ConfidenceConfig(preserve_frame_confidence=True)

    def list_available_models(self):
        return []

    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        """Pass-through to the ensemble models.

        Note that training is not actually supported for this class!
        """
        for model_idx in range(self.num_models):
            getattr(self, f"model{model_idx}").setup_training_data(train_data_config)

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        """Pass-through to the ensemble models."""
        for model_idx in range(self.num_models):
            getattr(self, f"model{model_idx}").setup_validation_data(val_data_config)

    def change_attention_model(
        self, self_attention_model: str = None, att_context_size: List[int] = None, update_config: bool = True
    ):
        """Pass-through to the ensemble models."""
        for model_idx in range(self.num_models):
            getattr(self, f"model{model_idx}").change_attention_model(
                self_attention_model, att_context_size, update_config
            )

    def change_decoding_strategy(self, decoding_cfg: DictConfig = None, decoder_type: str = None):
        """Pass-through to the ensemble models.

        The only change here is that we always require expected temperature
        to be set as well as ``decoding_cfg.preserve_alignments = True``
        """
        self.update_decoding_parameters(decoding_cfg)
        for model_idx in range(self.num_models):
            model = getattr(self, f"model{model_idx}")
            if isinstance(model, EncDecHybridRNNTCTCModel):
                model.change_decoding_strategy(decoding_cfg, decoder_type=decoder_type)
            else:
                model.change_decoding_strategy(decoding_cfg)

    @torch.no_grad()
    def transcribe(
        self,
        paths2audio_files: List[str],
        batch_size: int = 4,
        return_hypotheses: bool = False,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
        **kwargs,  # any other model specific parameters are passed directly
    ) -> List[str]:
        """Confidence-ensemble transcribe method.

        Consists of the following steps:

            1. Run all models (TODO: in parallel)
            2. Compute confidence for each model
            3. Use logistic regression to pick the "most confident" model
            4. Return the output of that model
        """
        # TODO: lots of duplicate code with building ensemble script
        confidences = []
        all_transcriptions = []
        # always requiring to return hypothesis
        # TODO: make sure to return text only if was False originally
        return_hypotheses = True
        for model_idx in range(self.num_models):
            model = getattr(self, f"model{model_idx}")
            transcriptions = model.transcribe(
                paths2audio_files=paths2audio_files,
                batch_size=batch_size,
                return_hypotheses=return_hypotheses,
                num_workers=num_workers,
                channel_selector=channel_selector,
                augmentor=augmentor,
                verbose=verbose,
                **kwargs,
            )
            if isinstance(transcriptions, tuple):  # transducers return a tuple
                transcriptions = transcriptions[0]

            model_confidences = []
            for transcription in transcriptions:
                model_confidences.append(compute_confidence(transcription, self.confidence_cfg))
            confidences.append(model_confidences)
            all_transcriptions.append(transcriptions)

        # transposing with zip(*list)
        features = np.array(list(zip(*confidences)))
        model_indices = self.model_selection_block.predict(features)
        final_transcriptions = []
        for transcrption_idx in range(len(all_transcriptions[0])):
            final_transcriptions.append(all_transcriptions[model_indices[transcrption_idx]][transcrption_idx])

        return final_transcriptions

    def list_available_models(self):
        return []
