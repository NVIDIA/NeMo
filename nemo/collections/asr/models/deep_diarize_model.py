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
from typing import Any, Dict, List, Optional, Union

import hydra.utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from x_transformers import Decoder

from nemo.collections.asr.data.deep_diarize.inference_data import RTTMDataset
from nemo.collections.asr.data.deep_diarize.train_data import LocalRTTMStreamingSegmentsDataset, MultiStreamDataLoader
from nemo.collections.asr.data.deep_diarize.train_tarred_data import TarredRTTMStreamingSegmentsDataset
from nemo.collections.asr.metrics.der import DER
from nemo.collections.asr.metrics.running_avg import RunningAverage
from nemo.collections.asr.models.asr_model import ASRModel
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
                shift_mem_down=self.cfg.shift_mem_down,
            ),
        )

        self.loss = torch.nn.BCELoss()
        self.sigmoid = torch.nn.Sigmoid()
        self.apply(self._init_weights)
        self.mems = None
        self.running_loss_avg = RunningAverage()
        self.der, self.der_no_mem = [
            DER(
                seconds_per_frame=self.cfg.preprocessor.window_stride * self.cfg.subsampling,
                min_seconds_for_segment=self.cfg.min_seconds_for_segment,
                threshold=self.cfg.threshold,
                combine_segments_seconds=self.cfg.combine_segments_seconds,
            )
            for x in range(2)
        ]
        self.mask_mem = self.cfg.get('mask_mem', False)

    def _mask_mems(self, mask: List[bool]):
        if self.mems is not None:
            for mem in self.mems:
                mem[mask, :, :] = 0

    def training_step(self, batch, batch_idx):
        train_x, train_lengths, y, mask = batch
        if self.mask_mem:
            self._mask_mems(mask)
        seg, model_lengths = self.sub_sample_layer(train_x, lengths=train_lengths)
        logits, self.mems = self.transformer_model(seg, mems=self.mems)
        logits = self.sigmoid(logits)
        loss = self.loss(logits, y)
        self.running_loss_avg(loss)
        self.logger.log_metrics(
            {
                "train_loss": loss,
                "learning_rate": self.lr_schedulers().get_last_lr()[0],
                'running_train_loss': self.running_loss_avg.compute(),
            },
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
        inputs, input_lengths, y, annotations = batch
        mems, logits, no_mem_logits = None, None, None
        for chunk, length in zip(inputs, input_lengths):
            seg, model_lengths = self.sub_sample_layer(chunk, lengths=length.unsqueeze(0))
            chunk_logits, mems = self.transformer_model(seg, mems=mems)
            chunk_logits = self.sigmoid(chunk_logits)
            logits = chunk_logits if logits is None else torch.cat((logits, chunk_logits), dim=1)

            chunk_logits, _ = self.transformer_model(seg)
            chunk_logits = self.sigmoid(chunk_logits)
            no_mem_logits = chunk_logits if no_mem_logits is None else torch.cat((no_mem_logits, chunk_logits), dim=1)

        loss = self._calculate_val_loss(logits, y)
        no_mem_loss = self._calculate_val_loss(no_mem_logits, y)
        self.log('val_loss', loss, sync_dist=True)
        self.log('no_mem_val_loss', no_mem_loss, sync_dist=True)
        self.der(logits.squeeze(0), annotations)
        self.der_no_mem(no_mem_logits.squeeze(0), annotations)
        self.log('DER', self.der, sync_dist=True)
        self.log('no_mem_DER', self.der_no_mem, sync_dist=True)

    def _calculate_val_loss(self, logits, y):
        loss = self.loss(logits, y.unsqueeze(0))
        # calculate the loss after we flipped the speaker labels as well
        invert_loss = self.loss(logits, torch.flip(y, dims=(-1,)).unsqueeze(0))
        # take the minimum loss
        loss = min(loss, invert_loss)
        return loss

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

    def _setup_preprocessor(self, dataloader_cfg):
        featurizer = WaveformFeaturizer(
            sample_rate=dataloader_cfg.sample_rate, int_values=dataloader_cfg.int_values, augmentor=None
        )
        preprocessor = hydra.utils.instantiate(self.cfg.preprocessor)
        # todo: will be written by validation, in this case it will always be the same, but just to be aware.
        self.train_sequence_length = preprocessor.featurizer.get_seq_len(
            (torch.tensor(self.cfg.chunk_seconds * dataloader_cfg.sample_rate, dtype=torch.float))
        )
        self.sub_sample_length = int(self.train_sequence_length / self.cfg.subsampling)
        return featurizer, preprocessor

    def setup_training_data(self, cfg: Optional[Union[DictConfig, Dict]]):
        featurizer, preprocessor = self._setup_preprocessor(cfg)

        spec_augmentation = ASRModel.from_config_dict(self.cfg.spec_augment)

        cls = TarredRTTMStreamingSegmentsDataset if cfg.tarred else LocalRTTMStreamingSegmentsDataset
        datasets = cls.create_streaming_datasets(
            manifest_filepath=cfg.manifest_filepath,
            preprocessor=preprocessor,
            featurizer=featurizer,
            spec_augmentation=spec_augmentation,
            window_stride=self.cfg.preprocessor.window_stride,
            subsampling=self.cfg.subsampling,
            train_segment_seconds=self.cfg.chunk_seconds,
            batch_size=cfg.batch_size,
            max_workers=cfg.num_workers,
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
        self._validation_dl = DataLoader(
            dataset, num_workers=cfg.num_workers, collate_fn=dataset.collate_fn, shuffle=False
        )

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        pass

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['mems'] = self.mems

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.mems = checkpoint['mems']

    def _move_mems_to_device(self, mems: List[torch.Tensor]) -> List[torch.Tensor]:
        return [x.to(self.device) for x in mems]

    def on_train_start(self) -> None:
        self.mems = self._move_mems_to_device(self.mems) if self.mems is not None else self.mems
