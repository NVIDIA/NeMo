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

import torch
from omegaconf import DictConfig, open_dict

from nemo.collections.tts.helpers.helpers import waveglow_log_to_tb_func
from nemo.core.classes import ModelPT
from nemo.utils.decorators import experimental
from nemo.utils import logging


@experimental
class Waveglow(ModelPT):
    def __init__(self, cfg: 'DictConfig', trainer: 'Trainer' = None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.pad_value = self._cfg.preprocessor.params.pad_value
        self.sigma = 1.0
        self.audio_to_melspec_precessor = Waveglow.from_config_dict(self._cfg.preprocessor)
        self.waveglow = Waveglow.from_config_dict(self._cfg.waveglow)

        self.setup_optimization()

    def loss(self, z, log_s_list, log_det_W_list):
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s)
                log_det_W_total = log_det_W_list[i]
            else:
                log_s_total = log_s_total + torch.sum(log_s)
                log_det_W_total += log_det_W_list[i]

        loss = torch.sum(z * z) / (2 * self.sigma * self.sigma) - log_s_total - log_det_W_total
        return loss / (z.size(0) * z.size(1) * z.size(2))

    def forward(self, audio, audio_len):
        spec, spec_len = self.audio_to_melspec_precessor(audio, audio_len)
        if self.training:
            return self.waveglow((spec, audio))
        else:
            audio_pred = self.waveglow.infer(spec)
        return audio_pred, spec, spec_len

    def training_step(self, batch, batch_idx):
        audio, audio_len = batch
        z, log_s_list, log_det_W_list = self.forward(audio, audio_len)

        loss = self.loss(z=z, log_s_list=log_s_list, log_det_W_list=log_det_W_list)
        output = {
            'loss': loss,
            'progress_bar': {'training_loss': loss},
            'log': {'loss': loss},
        }
        return output

    def validation_step(self, batch, batch_idx):
        audio, audio_len = batch
        audio_pred, spec, spec_len = self.forward(audio, audio_len)
        return {
            "audio_pred": audio_pred,
            "mel_target": spec,
            "mel_len": spec_len,
        }

    def validation_epoch_end(self, outputs):
        waveglow_log_to_tb_func(
            self.logger.experiment,
            outputs[0].values(),
            self.global_step,
            tag="eval",
            mel_fb=self.audio_to_melspec_precessor.fb,
        )
        return {}

    def __setup_dataloader_from_config(self, cfg, shuffle_should_be: bool = True, name: str = "train"):
        logging.debug(isinstance(cfg["dataset"], (dict, DictConfig)))
        if "dataset" not in cfg or not isinstance(cfg["dataset"], (dict, DictConfig)):
            raise ValueError(f"No dataset for {name}")  # TODO
        if "dataloader_params" not in cfg or not isinstance(cfg["dataloader_params"], (dict, DictConfig)):
            raise ValueError(f"No dataloder_params for {name}")  # TODO
        if shuffle_should_be:
            if 'shuffle' not in cfg["dataloader_params"]:
                logging.warning(
                    f"Shuffle should be set to True for {self}'s {name} dataloader but was not found in its "
                    "config. Manually setting to True"
                )
                with open_dict(cfg["dataloader_params"]):
                    cfg["dataloader_params"]["shuffle"] = True
            elif not cfg["dataloader_params"]["shuffle"]:
                logging.error(f"The {name} dataloader for {self} has shuffle set to False!!!")
        elif not shuffle_should_be and cfg["dataloader_params"]["shuffle"]:
            logging.error(f"The {name} dataloader for {self} has shuffle set to True!!!")

        dataset = Waveglow.from_config_dict(cfg["dataset"])
        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg["dataloader_params"])

    def setup_training_data(self, cfg):
        self._train_dl = self.__setup_dataloader_from_config(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self.__setup_dataloader_from_config(cfg, shuffle_should_be=False, name="validation")

    @classmethod
    def list_available_models(cls) -> 'Optional[Dict[str, str]]':
        pass

    @classmethod
    def from_pretrained(cls, name: str):
        pass

    def export(self, **kwargs):
        pass
