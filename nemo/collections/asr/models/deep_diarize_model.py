# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import math
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from x_transformers import Decoder

from nemo.collections.asr.data.deep_diarize.inference_data import RTTMDataset
from nemo.collections.asr.data.deep_diarize.train_data import MultiStreamDataLoader, RTTMStreamingSegmentsDataset
from nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessor
from nemo.collections.asr.modules.deep_diarize_transformer import TransformerXL
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling
from nemo.core import PretrainedModelInfo
from nemo.core.classes.modelPT import ModelPT


class DeepDiarizeModel(ModelPT):
    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        pass

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer)
        self.save_hyperparameters()
        self.num_speakers = self.cfg.num_speakers

        # apply the same sub-sampling as conformer to reduce the time dimension.
        self.sub_sample_layer = ConvSubsampling(
            subsampling="striding",
            subsampling_factor=self.cfg.subsampling,
            conv_channels=self.cfg.d_model,
            feat_in=self.cfg.feat_in,
            feat_out=self.cfg.d_model,
            is_causal=False,
        )

        self.transformer_model = TransformerXL(
            max_seq_len=self.sub_sample_length,
            max_mem_len=self.sub_sample_length,
            dim_in=self.cfg.d_model,
            dim_out=self.cfg.num_speakers,
            attn_layers=Decoder(
                dim=self.cfg.hidden_dim,
                depth=self.cfg.n_layers,
                heads=self.cfg.n_heads,
                dropout=self.cfg.dropout,
                emb_dropout=self.cfg.emb_dropout,
                rel_pos_bias=True,
            ),
        )

        self.loss = torch.nn.BCELoss()
        self.sigmoid = torch.nn.Sigmoid()
        self.apply(self._init_weights)
        self.mems = None

    def training_step(self, batch, batch_idx):
        train_x, train_lengths, y, _, sample_ids, offsets = batch
        seg, model_lengths = self.sub_sample_layer(train_x, lengths=train_lengths)
        logits, self.mems = self.transformer_model(seg, mems=self.mems)
        logits = self.sigmoid(logits)
        loss = self.loss(logits, y)
        self.logger.log_metrics(
            {"train_loss": loss, "learning_rate": self.lr_schedulers().get_last_lr()[0],},
            step=self.trainer.global_step,
        )
        return loss

    def divide(self, num, max_length):
        n_chunks = math.ceil(num / max_length)
        remainder = num % max_length
        lengths = [max_length if num - (max_length * (x + 1)) >= 0 else remainder for x in range(n_chunks)]
        return lengths

    def pad(self, chunk):
        return F.pad(chunk.transpose(1, 2), pad=(0, self.train_sequence_length - chunk.size(1)), value=0).transpose(
            1, 2
        )

    def validation_step(self, batch, batch_idx):
        inputs, input_lengths, y_splits, y = batch
        mems = None
        logits = None
        for chunk, length, y_split in zip(inputs, input_lengths, y_splits):
            seg, model_lengths = self.sub_sample_layer(chunk, lengths=length.unsqueeze(0))
            chunk_logits, mems = self.transformer_model(seg, mems=mems)
            chunk_logits = self.sigmoid(chunk_logits)
            logits = chunk_logits if logits is None else torch.cat((logits, chunk_logits), dim=1)
        loss = self.loss(logits, y.unsqueeze(0))
        self.log('val_loss', loss, sync_dist=True)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def setup_optimizer_param_groups(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)]
        param_groups = [
            {"params": params_decay, "weight_decay": 0.1},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]

        self._optimizer_param_groups = param_groups

    def _setup_preprocessor(self, cfg):
        featurizer = WaveformFeaturizer(sample_rate=cfg.sample_rate, int_values=cfg.int_values, augmentor=None)
        preprocessor = AudioToMelSpectrogramPreprocessor(
            normalize="per_feature",
            window_size=0.025,
            window_stride=self.cfg.preprocessor.window_stride,
            sample_rate=self.cfg.preprocessor.sample_rate,
            features=self.cfg.preprocessor.features,
            n_fft=512,
            frame_splicing=1,
            dither=0.00001,
            pad_to=0,
        )
        # todo: will be written by validation, in this case it will always be the same, but just to be aware.
        self.train_sequence_length = preprocessor.featurizer.get_seq_len(
            (torch.tensor(self.cfg.chunk_seconds * cfg.sample_rate, dtype=torch.float))
        )
        self.sub_sample_length = int(self.train_sequence_length / self.cfg.subsampling)
        return featurizer, preprocessor

    def setup_training_data(self, cfg: Optional[Union[DictConfig, Dict]]):
        featurizer, preprocessor = self._setup_preprocessor(cfg)

        datasets = RTTMStreamingSegmentsDataset.create_streaming_datasets(
            manifest_filepath=cfg.manifest_filepath,
            preprocessor=preprocessor,
            featurizer=featurizer,
            window_stride=self.cfg.preprocessor.window_stride,
            subsampling=self.cfg.subsampling,
            train_segment_seconds=self.cfg.chunk_seconds,
            batch_size=cfg.batch_size,
            max_workers=cfg.num_workers,
            num_calls=cfg.num_samples,
        )
        self._train_dl = MultiStreamDataLoader(datasets)

    def setup_validation_data(self, cfg: Optional[Union[DictConfig, Dict]]):
        featurizer, preprocessor = self._setup_preprocessor(cfg)
        dataset = RTTMDataset(
            manifest_filepath=cfg.manifest_filepath,
            preprocessor=preprocessor,
            featurizer=featurizer,
            window_stride=self.cfg.preprocessor.window_stride,
            subsampling=self.cfg.subsampling,
            segment_seconds=self.cfg.chunk_seconds,
        )
        self._validation_dl = DataLoader(dataset, num_workers=cfg.num_workers, collate_fn=dataset.collate_fn)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        pass
