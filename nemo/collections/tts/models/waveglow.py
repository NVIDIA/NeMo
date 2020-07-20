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

from nemo.core.classes import ModelPT
from nemo.utils.decorators import experimental
from nemo.collections.tts.helpers.helpers import waveglow_log_to_tb_func
from nemo.collections.tts.data.datalayers import AudioDataset


@experimental
class WaveglowPTL(ModelPT):
    def __init__(self, cfg: 'DictConfig', trainer: 'Trainer' = None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.pad_value = self._cfg.preprocessor.params.pad_value
        self.sigma = 1.0
        self.audio_to_melspec_precessor = WaveglowPTL.from_config_dict(self._cfg.preprocessor)
        self.waveglow = WaveglowPTL.from_config_dict(self._cfg.waveglow)

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

    def setup_training_data(self, config):
        if 'shuffle' not in config:
            config['shuffle'] = True
        dataset = AudioDataset(
            manifest_filepath=config["manifest_filepath"],
            n_segments=config["n_segments"],
            min_duration=config.get("min_duration", 0.1),
            max_duration=config.get("max_duration", None),
            trim=config.get("trim_silence", False),
        )
        self._train_dl = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.get("batch_size", False),
            shuffle=config["shuffle"],
            collate_fn=dataset._collate_fn,
        )

    def setup_validation_data(self, config):
        if 'shuffle' not in config:
            config['shuffle'] = False
        dataset = AudioDataset(
            manifest_filepath=config["manifest_filepath"],
            n_segments=config["n_segments"],
            min_duration=config.get("min_duration", 0.1),
            max_duration=config.get("max_duration", None),
            trim=config.get("trim_silence", False),
        )
        self._validation_dl = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.get("batch_size", False),
            shuffle=config["shuffle"],
            collate_fn=dataset._collate_fn,
        )

    @classmethod
    def list_available_models(cls) -> 'Optional[Dict[str, str]]':
        pass

    @classmethod
    def from_pretrained(cls, name: str):
        pass

    def export(self, **kwargs):
        pass
