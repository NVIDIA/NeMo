# Copyright 2020 NVIDIA. All Rights Reserved.
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
import argparse
import math
import os
from functools import partial

import torch
from torch.nn.functional import pad
from pytorch_lightning.core.lightning import LightningModule
from ruamel.yaml import YAML
from torch import nn
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers

import nemo.collections.tts.jason as nemo_tts_jason
import nemo.collections.asr as nemo_asr
from nemo.collections.tts.jason.helpers.helpers import tacotron2_log_to_tb_func, get_mask_from_lengths
from nemo.utils import logging

class Tacotron2PTL(LightningModule):
    # TODO: gradient norm, learning rate scheduler, tensorboard for training
    def __init__(self, labels, train_dataset, eval_dataset):
        super().__init__()
        self.labels = labels
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.pad_value = -11.42
        self.featurizer = nemo_asr.parts.features.WaveformFeaturizer(22050)
        self.audio_to_melspec_precessor = nemo_tts_jason.data.processors.FilterbankFeatures(
            sample_rate=22050,
            n_window_size=1024,
            n_window_stride=256,
            normalize=None,
            n_fft=1024,
            preemph=None,
            nfilt=80,
            lowfreq=0,
            highfreq=None,
            log=True,
            log_zero_guard_type="clamp",
            log_zero_guard_value=1e-5,
            dither=0.,
            pad_to=8,
            frame_splicing=1,
            pad_value=self.pad_value,
            mag_power=1.0,
            stft_conv=True,
        )
        self.text_embedding = nn.Embedding(len(labels) + 3, 512)
        self.encoder = nemo_tts_jason.tacotron2.Encoder(5, 512, 3)
        self.decoder = nemo_tts_jason.tacotron2.Decoder(
            n_mel_channels=80,
            n_frames_per_step=1,
            encoder_embedding_dim=512,
            gate_threshold=0.5,
            prenet_dim=256,
            max_decoder_steps=1000,
            decoder_rnn_dim=1024,
            p_decoder_dropout=0.1,
            p_attention_dropout=0.1,
            attention_rnn_dim=1024,
            attention_dim=128,
            attention_location_n_filters=32,
            attention_location_kernel_size=31,
            prenet_p_dropout=0.5,
            early_stopping=True,
        )
        self.postnet = nemo_tts_jason.tacotron2.Postnet(
            n_mel_channels=80,
            postnet_embedding_dim=512,
            postnet_kernel_size=5,
            postnet_n_convolutions=5,
            p_dropout=0.5,
        )

    def loss (
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

        # loss = None
        # logger_logs = {}
        output = {
            'loss': loss, # required
            'progress_bar': {'training_loss': loss}, # optional (MUST ALL BE TENSORS)
            # 'log': logger_logs
        }
        # return a dict
        return output

    def train_dataloader(self):
        dataset = nemo_tts_jason.data.datalayers.AudioToTextDataset(
            manifest_filepath=self.train_dataset,
            labels=self.labels,
            featurizer=self.featurizer,
            bos_id=len(self.labels),
            eos_id=len(self.labels) + 1,
            pad_id=len(self.labels) + 2,
            min_duration=0.1,
            max_duration=None,
            normalize=False,
            trim=False,
        )
        return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=dataset._collate_fn)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-6)

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
        tacotron2_log_to_tb_func(self.logger.experiment, outputs[0].values(), self.global_step, tag="eval", log_images=True, add_audio=False)
        return {}

    def val_dataloader(self):
        dataset = nemo_tts_jason.data.datalayers.AudioToTextDataset(
            manifest_filepath=self.eval_dataset,
            labels=self.labels,
            featurizer=self.featurizer,
            bos_id=len(self.labels),
            eos_id=len(self.labels) + 1,
            pad_id=len(self.labels) + 2,
            min_duration=0.1,
            max_duration=None,
            normalize=False,
            trim=False,
        )
        return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=dataset._collate_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", default=None, type=str, help="working directory for experiment")
    parser.add_argument("--train_dataset", default=None, type=str, help="working directory for experiment")
    parser.add_argument("--eval_datasets", default=None, type=str, help="working directory for experiment")
    args = parser.parse_args()
    labels = [
        ' ', '!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?', 'A', 'B',
        'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
        'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', 'a', 'b',
        'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
        'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
    ]
    tb_logger = pl_loggers.TensorBoardLogger(args.work_dir)
    model = Tacotron2PTL(labels, args.train_dataset, args.eval_datasets)
    trainer = Trainer(gpus=1, num_nodes=1, logger=tb_logger)
    trainer.fit(model)

if __name__ == '__main__':
    main()
