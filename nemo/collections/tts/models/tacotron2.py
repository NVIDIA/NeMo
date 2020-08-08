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
from typing import Dict, List, Optional

import torch
from omegaconf import MISSING, DictConfig, OmegaConf, open_dict
from torch import nn

from nemo.collections.tts.helpers.helpers import tacotron2_log_to_tb_func
from nemo.collections.tts.losses.tacotron2loss import Tacotron2Loss
from nemo.core.classes import ModelPT, typecheck
from nemo.core.neural_types.elements import (
    AudioSignal,
    EmbeddedTextType,
    LengthsType,
    LogitsType,
    MelSpectrogramType,
    SequenceToSequenceAlignmentType,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging
from nemo.utils.decorators import experimental


@dataclass
class PreprocessorParams:
    pad_value: float = MISSING


@dataclass
class Preprocessor:
    cls: str = MISSING
    params: PreprocessorParams = PreprocessorParams()


@dataclass
class Tacotron2Config:
    preprocessor: Preprocessor = Preprocessor()
    encoder: Dict = MISSING
    decoder: Dict = MISSING
    postnet: Dict = MISSING
    labels: List = MISSING
    train_ds: Optional[Dict] = None
    validation_ds: Optional[Dict] = None


@experimental  # TODO: Need to implement abstract methods: list_available_models
class Tacotron2Model(ModelPT):
    """ Tacotron 2 Model that is used to generate mel spectrograms from text
    """

    # TODO: tensorboard for training
    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        super().__init__(cfg=cfg, trainer=trainer)

        schema = OmegaConf.structured(Tacotron2Config)
        # ModelPT ensures that cfg is a DictConfig, but do this second check in case ModelPT changes
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
        # Ensure passed cfg is compliant with schema
        OmegaConf.merge(cfg, schema)

        self.pad_value = self._cfg.preprocessor.params.pad_value
        self.audio_to_melspec_precessor = Tacotron2Model.from_config_dict(self._cfg.preprocessor)
        self.text_embedding = nn.Embedding(len(cfg.labels) + 3, 512)
        self.encoder = Tacotron2Model.from_config_dict(self._cfg.encoder)
        self.decoder = Tacotron2Model.from_config_dict(self._cfg.decoder)
        self.postnet = Tacotron2Model.from_config_dict(self._cfg.postnet)
        self.loss = Tacotron2Loss()

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'T'), AudioSignal()),
            "audio_len": NeuralType(('B'), LengthsType()),
            "tokens": NeuralType(('B', 'T'), EmbeddedTextType()),
            "token_len": NeuralType(('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "mel_out": NeuralType(('B', 'T', 'D'), MelSpectrogramType()),
            "mel_out_postnet": NeuralType(('B', 'T', 'D'), MelSpectrogramType()),
            "gate_out": NeuralType(('B', 'T'), LogitsType()),
            "mel_target": NeuralType(('B', 'T', 'D'), MelSpectrogramType()),
            "gate_target": NeuralType(('B', 'T'), LogitsType()),
            "target_len": NeuralType(('B'), LengthsType()),
            "alignments": NeuralType(('B', 'T', 'T'), SequenceToSequenceAlignmentType()),
        }

    @typecheck()
    def forward(self, *, audio, audio_len, tokens, token_len):
        spec, spec_len = self.audio_to_melspec_precessor(audio, audio_len)
        token_embedding = self.text_embedding(tokens).transpose(1, 2)
        encoder_embedding = self.encoder(token_embedding=token_embedding, token_len=token_len)
        if self.training:
            spec_dec, gate, alignments = self.decoder(
                memory=encoder_embedding, decoder_inputs=spec, memory_lengths=token_len
            )
        else:
            spec_dec, gate, alignments, _ = self.decoder(memory=encoder_embedding, memory_lengths=token_len)
        spec_postnet = self.postnet(mel_spec=spec_dec)

        max_len = spec.shape[2]
        gate_padded = torch.zeros(spec_len.shape[0], max_len)
        gate_padded = gate_padded.type_as(gate)
        for i, length in enumerate(spec_len):
            gate_padded[i, length.data - 1 :] = 1

        return spec_dec, spec_postnet, gate, spec, gate_padded, spec_len, alignments

    def training_step(self, batch, batch_idx):
        audio, audio_len, tokens, token_len = batch
        spec_dec, spec_postnet, gate, spec, gate_padded, spec_len, _ = self.forward(
            audio=audio, audio_len=audio_len, tokens=tokens, token_len=token_len
        )

        loss = self.loss(
            mel_out=spec_dec,
            mel_out_postnet=spec_postnet,
            gate_out=gate,
            mel_target=spec,
            gate_target=gate_padded,
            target_len=spec_len,
            pad_value=self.pad_value,
        )

        output = {
            'loss': loss,
            'progress_bar': {'training_loss': loss},
            'log': {'loss': loss},
        }
        return output

    def validation_step(self, batch, batch_idx):
        audio, audio_len, tokens, token_len = batch
        spec_dec, spec_postnet, gate, spec, gate_padded, spec_len, alignments = self.forward(
            audio=audio, audio_len=audio_len, tokens=tokens, token_len=token_len
        )

        loss = self.loss(
            mel_out=spec_dec,
            mel_out_postnet=spec_postnet,
            gate_out=gate,
            mel_target=spec,
            gate_target=gate_padded,
            target_len=spec_len,
            pad_value=self.pad_value,
        )
        return {
            "loss": loss,
            "mel_target": spec,
            "mel_postnet": spec_postnet,
            "gate": gate,
            "gate_target": gate_padded,
            "alignments": alignments,
        }

    def validation_epoch_end(self, outputs):
        if self.logger is not None and self.logger.experiment is not None:
            tacotron2_log_to_tb_func(
                self.logger.experiment,
                outputs[0].values(),
                self.global_step,
                tag="val",
                log_images=True,
                add_audio=False,
            )
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def __setup_dataloader_from_config(self, cfg, shuffle_should_be: bool = True, name: str = "train"):
        if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
            raise ValueError(f"No dataset for {name}")  # TODO
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloder_params for {name}")  # TODO
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

        labels = cfg.dataset.params.labels
        dataset = Tacotron2Model.from_config_dict(
            cfg.dataset, bos_id=len(labels), eos_id=len(labels) + 1, pad_id=len(labels) + 2,
        )
        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def setup_training_data(self, cfg):
        self._train_dl = self.__setup_dataloader_from_config(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self.__setup_dataloader_from_config(cfg, shuffle_should_be=False, name="validation")

    @classmethod
    def list_available_models(cls) -> 'Optional[Dict[str, str]]':
        """TODO: Implement me!"""
        pass
