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
import itertools
import pdb
from typing import Any, Dict, List, Optional, Union

import hydra.utils
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchmetrics.functional import permutation_invariant_training
from x_transformers import Decoder

from nemo.collections.asr.data.deep_diarize.inference_data import RTTMDataset
from nemo.collections.asr.data.deep_diarize.train_data import LocalRTTMStreamingSegmentsDataset, MultiStreamDataLoader
from nemo.collections.asr.data.deep_diarize.train_tarred_data import TarredRTTMStreamingSegmentsDataset
from nemo.collections.asr.data.deep_diarize.utils import ContextWindow
from nemo.collections.asr.metrics.der import DER, FA, Confusion, MissedDetection
from nemo.collections.asr.models.asr_model import ASRModel
from nemo.collections.asr.modules.deep_diarize_transformer import TransformerXL
from nemo.collections.asr.modules.transformer import PerceiverEncoder, TransformerDecoder, TransformerEncoder
from nemo.collections.asr.parts.preprocessing import process_augmentations
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling
from nemo.core import PretrainedModelInfo
from nemo.core.classes.modelPT import ModelPT


class DiarizeTransformerXLDecoder(torch.nn.Module):
    def __init__(
        self,
        seq_len: int,
        num_speakers: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        inner_dim: int,
        cat_features: bool,
        dropout: float,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.cat_features = cat_features
        self.num_speakers = num_speakers
        if self.cat_features:
            self.decoder = TransformerEncoder(
                num_layers=n_layers,
                hidden_size=hidden_dim * 2,
                inner_size=inner_dim,
                num_attention_heads=n_heads,
                attn_score_dropout=dropout,
                attn_layer_dropout=dropout,
                ffn_dropout=dropout,
            )
            self.projection = torch.nn.Linear(hidden_dim * 2, 1)
        else:
            self.decoder = TransformerDecoder(
                num_layers=n_layers,
                hidden_size=hidden_dim,
                inner_size=inner_dim,
                num_attention_heads=n_heads,
                attn_score_dropout=dropout,
                attn_layer_dropout=dropout,
                ffn_dropout=dropout,
            )
            self.projection = torch.nn.Linear(hidden_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, attractors: torch.Tensor, encoded_xl_features: torch.Tensor) -> torch.Tensor:
        B, K, H = attractors.size()
        N = encoded_xl_features.size(1)
        # [B, K, H] -> [B * K, N, H]
        attractors = attractors.view(B, K, 1, H).expand(-1, -1, N, -1).reshape(B * K, N, H)
        attractors_mask = torch.ones(B * K, N).to(attractors.device)

        # [B, N, H] -> [B, K, N, H]
        transformer_xl_encoded_features = encoded_xl_features.view(B, 1, N, H).expand(-1, K, -1, -1)
        # [B, K, N, H] -> [B * K, N, H]
        transformer_xl_encoded_features = transformer_xl_encoded_features.reshape(B * K, N, H)
        encoder_mask = torch.ones(B * K, N).to(attractors.device)

        if self.cat_features:
            input = torch.cat((transformer_xl_encoded_features, attractors), dim=-1)
            encoder_mask = torch.ones(B * K, N).to(attractors.device)
            output = self.decoder(encoder_states=input, encoder_mask=encoder_mask,)

            # [B * K, N, H] -> [B * K, N] reshape to [B, N, K]
            activities = self.projection(output).view(B, K, N).transpose(1, 2)
            return activities.sigmoid()

        output = self.decoder(
            decoder_states=attractors,
            decoder_mask=attractors_mask,
            encoder_states=transformer_xl_encoded_features,
            encoder_mask=encoder_mask,
            encoder_diagonal=0,
        )

        # [B * K, N, H] -> [B * K, N] reshape to [B, N, K]
        activities = self.projection(output).view(B, K, N).transpose(1, 2)
        return self.sigmoid(activities)


class DeepDiarizeModel(ModelPT):
    @property
    def feat_in(self) -> int:
        return self.cfg.feat_in * ((self.cfg.context_size * 2) + 1)

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        pass

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer)
        self.save_hyperparameters()
        self.num_speakers = self.cfg.num_speakers

        # apply the same sub-sampling as conformer to reduce the time dimension.
        if self.cfg.conv_subsampling:
            self.sub_sample_layer = ConvSubsampling(
                subsampling="striding",
                subsampling_factor=self.cfg.subsampling,
                conv_channels=self.cfg.d_model,
                feat_in=self.feat_in,
                feat_out=self.cfg.d_model,
                is_causal=False,
            )
        else:
            self.sub_sample_layer = torch.nn.Linear(self.feat_in, self.cfg.d_model)

        self.transformer_feature_encoder = TransformerXL(
            max_seq_len=self.sub_sample_length,
            max_mem_len=self.sub_sample_length,
            dim_in=self.cfg.d_model,
            dim_out=self.cfg.d_model,
            attn_layers=Decoder(
                dim=self.cfg.encoder.hidden_dim,
                depth=self.cfg.encoder.n_layers,
                heads=self.cfg.encoder.n_heads,
                dropout=self.cfg.encoder.dropout,
                emb_dropout=self.cfg.encoder.emb_dropout,
                rel_pos_bias=True,
                shift_mem_down=self.cfg.encoder.shift_mem_down,
            ),
        )
        self.eda_module = PerceiverEncoder(
            num_layers=self.cfg.eda.n_layers,
            hidden_size=self.cfg.d_model,
            num_attention_heads=self.cfg.eda.n_heads,
            inner_size=self.cfg.eda.inner_dim,
            hidden_steps=self.num_speakers,
            attn_score_dropout=self.cfg.eda.dropout,
            attn_layer_dropout=self.cfg.eda.dropout,
            ffn_dropout=self.cfg.eda.dropout,
        )
        self.fc = torch.nn.Linear(self.cfg.d_model, 1)

        self.decoder = DiarizeTransformerXLDecoder(
            seq_len=self.sub_sample_length,
            num_speakers=self.num_speakers,
            hidden_dim=self.cfg.d_model,
            n_heads=self.cfg.decoder.n_heads,
            n_layers=self.cfg.decoder.n_layers,
            inner_dim=self.cfg.decoder.inner_dim,
            dropout=self.cfg.decoder.dropout,
            cat_features=self.cfg.decoder.cat_features,
        )

        self.loss = torch.nn.BCELoss()
        self.train_loss = torch.nn.BCELoss(reduction='none' if self.cfg.focal else 'mean')
        self.sigmoid = torch.nn.Sigmoid()
        self.apply(self._init_weights)
        self.mems = None

        self.segment_metrics = self._create_der_metrics('segment')
        self.call_metrics = self._create_der_metrics('call')
        self.permutations = None
        self.permutations_indices = None
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.train_attractors = None

    def _create_der_metrics(self, prefix: str = ''):
        shared_params = dict(
            seconds_per_frame=self.cfg.preprocessor.window_stride * self.cfg.subsampling,
            min_seconds_for_segment=self.cfg.min_seconds_for_segment,
            threshold=self.cfg.threshold,
            combine_segments_seconds=self.cfg.combine_segments_seconds,
        )
        metrics = {
            f'{prefix}_der': DER(**shared_params),
            f'{prefix}_missed_detection': MissedDetection(**shared_params),
            f'{prefix}_false_alarm': FA(**shared_params),
            f'{prefix}_confusion': Confusion(**shared_params),
            f'{prefix}_collar_der': DER(**shared_params, collar=0.25),
            f'{prefix}_ignore_overlap_der': DER(**shared_params, collar=0.25, ignore_overlap=True),
        }
        for k, metric in metrics.items():
            setattr(self, k, metric)
        return metrics

    def _mask_mems(self, mask: List[bool]):
        if self.mems is not None:
            for mem in self.mems:
                mem[mask, :, :] = 0

    def _permutation_mask(self, preds, y):
        if self.permutations is None:
            # todo: assumption of two speakers
            self.permutations = torch.empty(preds.size(0), 2, dtype=torch.int).to(self.device)

        for x in range(preds.size(0)):
            with torch.no_grad():
                calc_loss, permutation = permutation_invariant_training(
                    preds[x].unsqueeze(0).transpose(1, 2),
                    y[x].unsqueeze(0).transpose(1, 2),
                    metric_func=self._calculate_train_loss,
                    eval_func='min',
                )
            self.permutations[x] = permutation

        for x, labels in enumerate(y):
            y[x] = torch.index_select(y[x], 1, self.permutations[x])
        return y

    def _shuffle(self, tensor: torch.Tensor, dim: int):
        idx = torch.randperm(tensor.shape[dim])
        return tensor[:, idx]

    def _calculate_train_loss(self, logits, y):
        loss = self.train_loss(logits, y)
        if self.cfg.focal:
            pt = torch.exp(-loss)  # prevents nans when probability 0
            loss = self.cfg.alpha * (1 - pt) ** self.cfg.gamma * loss
            loss = loss.mean()
        return loss

    def _calculate_attractor_loss(self, attractors, y):
        # loss would be calculated between outputs and number of speakers present in segment
        outputs = self.fc(attractors).sigmoid().squeeze()
        speaker_exists = y.transpose(1, 2).sum(2)
        speaker_exists[speaker_exists > 0] = 1
        return self._calculate_train_loss(outputs, speaker_exists)

    def training_step(self, batch, batch_idx):
        train_x, train_lengths, y, memory_reset = batch
        if self.cfg.conv_subsampling:
            train_x, train_lengths = self.sub_sample_layer(train_x, lengths=train_lengths)
        else:
            train_x = self.sub_sample_layer(train_x)
        if self.cfg.mem_enabled:
            self._mask_mems(memory_reset)
            encoded_xl_features, self.mems = self.transformer_feature_encoder(train_x, mems=self.mems)
        else:
            encoded_xl_features, _ = self.transformer_feature_encoder(train_x)
        seq_mask = torch.ones(train_x.size(0), train_x.size(1)).to(self.device)
        # shuffle frames before generating attractors
        attractors, _ = self.eda_module(self._shuffle(train_x, dim=1), seq_mask)
        speaker_outputs = self.decoder(attractors, encoded_xl_features)

        if self.cfg.permute:
            y = self._permutation_mask(speaker_outputs, y)
        loss = self._calculate_train_loss(speaker_outputs, y)
        existence_loss = self._calculate_attractor_loss(attractors, y)

        self.logger.log_metrics(
            {
                "train_loss": loss,
                "existence_loss": existence_loss,
                "learning_rate": self.lr_schedulers().get_last_lr()[0],
            },
            step=self.trainer.global_step,
        )
        return loss + (self.cfg.existence_alpha * existence_loss)

    def _process_attractors(self, chunk_attractors: torch.Tensor, attractors: Optional[torch.Tensor]):
        if attractors is None:
            return chunk_attractors
        if self.permutations_indices is None:
            self.permutation_indices = list(
                torch.tensor(perm, device=self.device) for perm in itertools.permutations(range(self.num_speakers))
            )
        for x in range(attractors.size(0)):
            anchor = attractors[x]
            lowest, lowest_cos = None, 0
            for permutation in self.permutation_indices:
                permutated = torch.index_select(chunk_attractors[x], 0, permutation)
                # is mean the right thing here?
                similarity = self.cos(anchor.float(), permutated.float()).mean()
                if similarity > lowest_cos:
                    lowest = permutated
                    lowest_cos = similarity
            attractors[x] = (anchor + lowest) / 2
        return attractors

    def validation_step(self, batch, batch_idx):
        inputs, input_lengths, y, fr_level_ys, annotations, segment_annotations = batch
        segment_loss = 0
        if self.cfg.mem_enabled:
            mems = [
                torch.zeros(1, self.sub_sample_length, self.cfg.encoder.hidden_dim, device=self.device)
                for x in range(self.cfg.encoder.n_layers)
            ]
        speech_activity, attractors = None, None
        for chunk, fr_level_y, segment_annotation, length in zip(
            inputs, fr_level_ys, segment_annotations, input_lengths
        ):
            if self.cfg.conv_subsampling:
                chunk, length = self.sub_sample_layer(chunk, lengths=length.unsqueeze(0))
            else:
                chunk = self.sub_sample_layer(chunk)

            if self.cfg.mem_enabled:
                encoded_xl_features, mems = self.transformer_feature_encoder(chunk, mems=mems)
            else:
                encoded_xl_features, _ = self.transformer_feature_encoder(chunk)
            seq_mask = torch.ones(chunk.size(0), chunk.size(1)).to(self.device)
            chunk_attractors, _ = self.eda_module(self._shuffle(chunk, dim=1), seq_mask)

            if self.cfg.average_attractors:
                attractors = self._process_attractors(chunk_attractors, attractors)

            speaker_outputs = self.decoder(attractors, encoded_xl_features)

            for k, metric in self.segment_metrics.items():
                metric(speaker_outputs.squeeze(0), segment_annotation)
            segment_loss += self._calculate_val_loss(speaker_outputs, fr_level_y)
            speech_activity = (
                speaker_outputs if speech_activity is None else torch.cat((speech_activity, speaker_outputs), dim=1)
            )
        for name, metric in self.call_metrics.items():
            metric(speech_activity.squeeze(0), annotations)
            self.log(name, metric, sync_dist=True)

        for name, metric in self.segment_metrics.items():
            self.log(name, metric, sync_dist=True)

        call_loss = self._calculate_val_loss(speech_activity, y)
        self.log('val_loss', call_loss, sync_dist=True)
        self.log('segment_val_loss', segment_loss / len(inputs), sync_dist=True)

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
        augmentor = process_augmentations(dataloader_cfg['augmentor']) if 'augmentor' in dataloader_cfg else None
        featurizer = WaveformFeaturizer(
            sample_rate=dataloader_cfg.sample_rate, int_values=dataloader_cfg.int_values, augmentor=augmentor
        )
        preprocessor = hydra.utils.instantiate(self.cfg.preprocessor)
        context_window = ContextWindow(
            context_size=self.cfg.context_size,
            subsampling=self.cfg.subsampling if not self.cfg.conv_subsampling else 1,
        )
        # todo: will be written by validation, in this case it will always be the same, but just to be aware.
        self.train_sequence_length = preprocessor.featurizer.get_seq_len(
            (torch.tensor(self.cfg.chunk_seconds * dataloader_cfg.sample_rate, dtype=torch.float))
        )
        self.sub_sample_length = int(self.train_sequence_length / self.cfg.subsampling) + 1
        return featurizer, preprocessor, context_window

    def setup_training_data(self, cfg: Optional[Union[DictConfig, Dict]]):
        featurizer, preprocessor, context_window = self._setup_preprocessor(cfg)

        spec_augmentation = ASRModel.from_config_dict(self.cfg.spec_augment)

        cls = TarredRTTMStreamingSegmentsDataset if cfg.tarred else LocalRTTMStreamingSegmentsDataset
        datasets = cls.create_streaming_datasets(
            manifest_filepath=cfg.manifest_filepath,
            preprocessor=preprocessor,
            featurizer=featurizer,
            context_window=context_window,
            spec_augmentation=spec_augmentation,
            window_stride=self.cfg.preprocessor.window_stride,
            subsampling=self.cfg.subsampling,
            train_segment_seconds=self.cfg.chunk_seconds,
            batch_size=cfg.batch_size,
            max_workers=cfg.num_workers,
        )
        self._train_dl = MultiStreamDataLoader(datasets)

    def setup_validation_data(self, cfg: Optional[Union[DictConfig, Dict]]):
        featurizer, preprocessor, context_window = self._setup_preprocessor(cfg)
        dataset = RTTMDataset(
            manifest_filepath=cfg.manifest_filepath,
            preprocessor=preprocessor,
            featurizer=featurizer,
            context_window=context_window,
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
