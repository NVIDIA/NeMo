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
from torch import nn
from torch.nn.functional import pad

from nemo.collections.asr.data.audio_to_text import AudioToTextDataset
from nemo.collections.asr.parts.features import WaveformFeaturizer
from nemo.collections.tts.helpers.helpers import get_mask_from_lengths, tacotron2_log_to_tb_func
from nemo.core.classes import ModelPT
from nemo.utils.decorators import experimental


@experimental
class Tacotron2PTL(ModelPT):
    # TODO: tensorboard for training
    def __init__(self, cfg: 'DictConfig', trainer: 'Trainer' = None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.pad_value = self._cfg.preprocessor.params.pad_value
        self.audio_to_melspec_precessor = Tacotron2PTL.from_config_dict(self._cfg.preprocessor)
        self.text_embedding = nn.Embedding(len(cfg.train_ds.labels) + 3, 512)
        self.encoder = Tacotron2PTL.from_config_dict(self._cfg.encoder)
        self.decoder = Tacotron2PTL.from_config_dict(self._cfg.decoder)
        self.postnet = Tacotron2PTL.from_config_dict(self._cfg.postnet)

        # After defining all torch.modules, create optimizer and scheduler
        self.setup_optimization()

    def loss(
        self, mel_out, mel_out_postnet, gate_out, mel_target, gate_target, target_len,
    ):
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        max_len = mel_target.shape[2]

        if max_len < mel_out.shape[2]:
            # Predicted len is larger than reference
            # Need to slice
            mel_out = mel_out.narrow(2, 0, max_len)
            mel_out_postnet = mel_out_postnet.narrow(2, 0, max_len)
            gate_out = gate_out.narrow(1, 0, max_len).contiguous()
        elif max_len > mel_out.shape[2]:
            # Need to do padding
            pad_amount = max_len - mel_out.shape[2]
            mel_out = pad(mel_out, (0, pad_amount), value=self.pad_value)
            mel_out_postnet = pad(mel_out_postnet, (0, pad_amount), value=self.pad_value)
            gate_out = pad(gate_out, (0, pad_amount), value=1e3)
            max_len = mel_out.shape[2]

        mask = ~get_mask_from_lengths(target_len, max_len=max_len)
        mask = mask.expand(mel_target.shape[1], mask.size(0), mask.size(1))
        mask = mask.permute(1, 0, 2)
        mel_out.data.masked_fill_(mask, self.pad_value)
        mel_out_postnet.data.masked_fill_(mask, self.pad_value)
        gate_out.data.masked_fill_(mask[:, 0, :], 1e3)

        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss

    def forward(self, audio, audio_len, tokens, token_len):
        spec, spec_len = self.audio_to_melspec_precessor(audio, audio_len)
        token_embedding = self.text_embedding(tokens).transpose(1, 2)
        encoder_embedding = self.encoder(token_embedding, token_len)
        if self.training:
            spec_dec, gate, alignments = self.decoder(encoder_embedding, spec, memory_lengths=token_len)
        else:
            spec_dec, gate, alignments, _ = self.decoder.infer(encoder_embedding, memory_lengths=token_len)
        spec_postnet = self.postnet(spec_dec)

        max_len = spec.shape[2]
        gate_padded = torch.zeros(spec_len.shape[0], max_len)
        gate_padded = gate_padded.type_as(gate)
        for i, length in enumerate(spec_len):
            gate_padded[i, length.data - 1 :] = 1

        return spec_dec, spec_postnet, gate, spec, gate_padded, spec_len, alignments

    def training_step(self, batch, batch_idx):
        audio, audio_len, tokens, token_len = batch
        spec_dec, spec_postnet, gate, spec, gate_padded, spec_len, _ = self.forward(
            audio, audio_len, tokens, token_len
        )

        loss = self.loss(
            mel_out=spec_dec,
            mel_out_postnet=spec_postnet,
            gate_out=gate,
            mel_target=spec,
            gate_target=gate_padded,
            target_len=spec_len,
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
            audio, audio_len, tokens, token_len
        )

        loss = self.loss(
            mel_out=spec_dec,
            mel_out_postnet=spec_postnet,
            gate_out=gate,
            mel_target=spec,
            gate_target=gate_padded,
            target_len=spec_len,
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
        tacotron2_log_to_tb_func(
            self.logger.experiment, outputs[0].values(), self.global_step, tag="eval", log_images=True, add_audio=False
        )
        return {}

    @staticmethod
    def __setup_dataloader_from_config(config: 'Optional[Dict]'):
        featurizer = WaveformFeaturizer(sample_rate=config['sample_rate'], int_values=config.get('int_values', False))
        dataset = AudioToTextDataset(
            manifest_filepath=config['manifest_filepath'],
            labels=config['labels'],
            featurizer=featurizer,
            bos_id=len(config['labels']),
            eos_id=len(config['labels']) + 1,
            pad_id=len(config['labels']) + 2,
            max_duration=config.get('max_duration', None),
            min_duration=config.get('min_duration', None),
            normalize=False,
            trim=False,
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=dataset._collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=config['shuffle'],
            num_workers=config.get('num_workers', 0),
        )

    def setup_training_data(self, config):
        if 'shuffle' not in config:
            config['shuffle'] = True
        self._train_dl = self.__setup_dataloader_from_config(config)

    def setup_validation_data(self, config):
        if 'shuffle' not in config:
            config['shuffle'] = False
        self._validation_dl = self.__setup_dataloader_from_config(config)

    @classmethod
    def list_available_models(cls) -> 'Optional[Dict[str, str]]':
        pass

    @classmethod
    def from_pretrained(cls, name: str):
        pass

    def export(self, **kwargs):
        pass
