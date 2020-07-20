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


from nemo.core.classes import ModelPT
import nemo.collections.asr as nemo_asr
import nemo.collections.tts as nemo_tts
from nemo.collections.tts.helpers.helpers import get_mask_from_lengths, tacotron2_log_to_tb_func
from nemo.utils.decorators import experimental


@experimental
class Tacotron2PTL(ModelPT):
    # TODO: tensorboard for training
    def __init__(self, cfg: 'DictConfig', trainer: 'Trainer' = None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.pad_value = self._cfg.preprocessor.params.pad_value
        self.audio_to_melspec_precessor = Tacotron2PTL.from_config_dict(self._cfg.preprocessor)
        # nemo_tts.data.processors.FilterbankFeatures(
        #     sample_rate=22050,
        #     n_window_size=1024,
        #     n_window_stride=256,
        #     normalize=None,
        #     n_fft=1024,
        #     preemph=None,
        #     nfilt=80,
        #     lowfreq=0,
        #     highfreq=None,
        #     log=True,
        #     log_zero_guard_type="clamp",
        #     log_zero_guard_value=1e-5,
        #     dither=0.0,
        #     pad_to=8,
        #     frame_splicing=1,
        #     pad_value=self.pad_value,
        #     mag_power=1.0,
        #     stft_conv=True,
        # )
        self.text_embedding = nn.Embedding(len(cfg.train_ds.labels) + 3, 512)
        # self.encoder = nemo_tts.modules.tacotron2.Encoder(5, 512, 3)
        self.encoder = Tacotron2PTL.from_config_dict(self._cfg.encoder)
        self.decoder = Tacotron2PTL.from_config_dict(self._cfg.decoder)
        # self.decoder = nemo_tts.modules.tacotron2.Decoder(
        #     n_mel_channels=80,
        #     n_frames_per_step=1,
        #     encoder_embedding_dim=512,
        #     gate_threshold=0.5,
        #     prenet_dim=256,
        #     max_decoder_steps=1000,
        #     decoder_rnn_dim=1024,
        #     p_decoder_dropout=0.1,
        #     p_attention_dropout=0.1,
        #     attention_rnn_dim=1024,
        #     attention_dim=128,
        #     attention_location_n_filters=32,
        #     attention_location_kernel_size=31,
        #     prenet_p_dropout=0.5,
        #     early_stopping=True,
        # )
        # self.postnet = nemo_tts.modules.tacotron2.Postnet(
        self.postnet = Tacotron2PTL.from_config_dict(self._cfg.postnet)
        #     n_mel_channels=80,
        #     postnet_embedding_dim=512,
        #     postnet_kernel_size=5,
        #     postnet_n_convolutions=5,
        #     p_dropout=0.5,
        # )

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

    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        audio, audio_len, tokens, token_len = batch
        spec, spec_len = self.audio_to_melspec_precessor(audio, audio_len)
        token_embedding = self.text_embedding(tokens).transpose(1, 2)
        encoder_embedding = self.encoder(token_embedding, token_len)
        spec_dec, gate, alignments = self.decoder(encoder_embedding, spec, memory_lengths=token_len)
        spec_postnet = self.postnet(spec_dec)

        max_len = spec.shape[2]
        gate_padded = torch.FloatTensor(spec_len.shape[0], max_len)
        gate_padded.zero_()
        for i, length in enumerate(spec_len):
            gate_padded[i, length.data - 1 :] = 1
        gate_padded = gate_padded.cuda()

        loss = self.loss(
            mel_out=spec_dec,
            mel_out_postnet=spec_postnet,
            gate_out=gate,
            mel_target=spec,
            gate_target=gate_padded,
            target_len=spec_len,
        )

        output = {
            'loss': loss,  # required
            'progress_bar': {'training_loss': loss},  # optional (MUST ALL BE TENSORS)
            'log': {'loss': loss},
        }
        # return a dict
        return output

    def validation_step(self, batch, batch_idx):
        audio, audio_len, tokens, token_len = batch
        spec, spec_len = self.audio_to_melspec_precessor.forward(audio, audio_len)
        token_embedding = self.text_embedding(tokens).transpose(1, 2)
        encoder_embedding = self.encoder(token_embedding, token_len)
        spec_dec, gate, alignments, mel_len = self.decoder.infer(encoder_embedding, memory_lengths=token_len)
        # mel_output, gate_output, alignments, mel_len
        spec_postnet = self.postnet(spec_dec)

        max_len = spec.shape[2]
        gate_padded = torch.FloatTensor(spec_len.shape[0], max_len)
        gate_padded.zero_()
        for i, length in enumerate(spec_len):
            gate_padded[i, length.data - 1 :] = 1
        gate_padded = gate_padded.cuda()

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
        featurizer = nemo_asr.parts.features.WaveformFeaturizer(
            sample_rate=config['sample_rate'], int_values=config.get('int_values', False)
        )
        dataset = nemo_tts.data.datalayers.AudioToTextDataset(
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
