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


# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Facebook, Inc. and its affiliates.
#
import logging
from math import ceil
from typing import Dict, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch import nn

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.losses.wav2vecloss import Wav2VecLoss
from nemo.collections.asr.models.wav2vec.wav2vec_config import Wav2VecEncoderModelConfig
from nemo.collections.asr.modules.wav2vec_modules import GumbelVectorQuantizer, compute_mask_indices
from nemo.collections.asr.parts.perturb import process_augmentations
from nemo.collections.asr.parts.wav2vec import ConvFeatureEncoder, GradMultiply, Wav2VecTransformerEncoder
from nemo.core import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import AudioSignal, EncodedRepresentation, LossType, MaskType, NeuralType
from nemo.core.neural_types.elements import BoolType, FloatType


def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


class Wav2VecEncoderModel(ModelPT):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        self.global_rank = 0
        self.world_size = 0
        self.local_rank = 0
        if trainer is not None:
            self.global_rank = (trainer.node_rank * trainer.num_gpus) + trainer.local_rank
            self.world_size = trainer.num_nodes * trainer.num_gpus
            self.local_rank = trainer.local_rank

        super().__init__(cfg=cfg, trainer=trainer)

        schema = OmegaConf.structured(Wav2VecEncoderModelConfig)
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")

        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg = OmegaConf.merge(schema, cfg)

        feature_enc_layers = cfg.conv_feature_encoder.conv_feature_layers
        self.embed = feature_enc_layers[-1][0]  # Select last conv output layer dimension

        self.feature_extractor = ConvFeatureEncoder(
            conv_layers=feature_enc_layers,
            mode=cfg.conv_feature_encoder.extractor_mode,
            conv_bias=cfg.conv_feature_encoder.conv_bias,
        )

        encoder_embed_dim = cfg.transformer_encoder.encoder.embedding_dim
        self.post_extract_proj = (
            nn.Linear(self.embed, encoder_embed_dim)
            if self.embed != encoder_embed_dim and not cfg.quantizer.quantize_input
            else None
        )

        self.mask_cfg = cfg.masking

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.quantizer = None
        self.input_quantizer = None

        self.n_negatives = cfg.n_negatives
        self.cross_sample_negatives = cfg.cross_sample_negatives
        self.codebook_negatives = cfg.codebook_negatives
        self.negatives_from_everywhere = cfg.negatives_from_everywhere

        final_dim = cfg.final_dim if cfg.final_dim > 0 else encoder_embed_dim
        self.final_dim = final_dim
        if cfg.quantizer.quantize_targets:
            vq_dim = cfg.quantizer.latent_dim if cfg.quantizer.latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.embed,
                num_vars=cfg.quantizer.latent_vars,
                temp=cfg.quantizer.latent_temp,
                groups=cfg.quantizer.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
            )
            self.project_q = nn.Linear(vq_dim, final_dim)
        else:
            self.project_q = nn.Linear(self.embed, final_dim)

        if cfg.quantizer.quantize_input:
            if cfg.quantizer.same_quantizer and self.quantizer is not None:
                vq_dim = final_dim
                self.input_quantizer = self.quantizer
            else:
                vq_dim = cfg.quantizer.latent_dim if cfg.quantizer.latent_dim > 0 else encoder_embed_dim
                self.input_quantizer = GumbelVectorQuantizer(
                    dim=self.embed,
                    num_vars=cfg.quantizer.latent_vars,
                    temp=cfg.quantizer.latent_temp,
                    groups=cfg.quantizer.latent_groups,
                    combine_groups=False,
                    vq_dim=vq_dim,
                    time_first=True,
                )
            self.project_inp = nn.Linear(vq_dim, encoder_embed_dim)

        self.mask_emb = nn.Parameter(torch.FloatTensor(encoder_embed_dim).uniform_())

        self.encoder = Wav2VecTransformerEncoder(cfg.transformer_encoder)
        self.layer_norm = nn.LayerNorm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(nn.Linear(final_dim, final_dim * 2), nn.GLU())

        self.final_proj = nn.Linear(encoder_embed_dim, final_dim)
        self.loss = Wav2VecLoss(
            feature_loss_weight=cfg.loss.feature_loss_weight,
            prob_ppl_weight=cfg.loss.prob_ppl_weight,
            logit_temp=cfg.logit_temp,
        )

    def training_step(self, batch, batch_idx):
        loss, feature_loss, prob_ppl_loss = self._step(batch)

        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])
        self.log('loss', loss, on_epoch=True)
        self.log('feature_loss', feature_loss, on_epoch=True, sync_dist=True)
        self.log('prob_ppl_loss', prob_ppl_loss, on_epoch=True, sync_dist=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, feature_loss, prob_ppl_loss = self._step(batch)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, feature_loss, prob_ppl_loss = self._step(batch)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)

    def _step(self, batch):
        audio_signal, audio_lengths, _, _ = batch

        padding_mask = self._create_padding_mask(audio_lengths)

        self._update_quantizer_temp()
        logits, targets, sampled_negatives, _, features_penalty, prob_ppl_loss, _ = self(
            source=audio_signal, padding_mask=padding_mask
        )
        loss, feature_loss, prob_ppl_loss = self.loss(
            logits=logits,
            targets=targets,
            negatives=sampled_negatives,
            prob_ppl_loss=prob_ppl_loss,
            feature_loss=features_penalty,
        )
        return loss, feature_loss, prob_ppl_loss

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return None

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "source": NeuralType(('B', 'T'), AudioSignal()),
            "padding_mask": NeuralType(('B', 'T'), MaskType(), optional=True),
            "mask": NeuralType(elements_type=BoolType(), optional=True),
            "features_only": NeuralType(elements_type=BoolType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "logits": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "targets": NeuralType(('B', 'T', 'D'), EncodedRepresentation(), optional=True),
            "sampled_negatives": NeuralType(('N', 'B', 'T', 'D'), EncodedRepresentation(), optional=True),
            "padding_mask": NeuralType(('B', 'T'), MaskType(), optional=True),
            "features_penalty": NeuralType(elements_type=LossType(), optional=True),
            "prob_ppl_loss": NeuralType(elements_type=LossType(), optional=True),
            "cur_codebook_temp": NeuralType(elements_type=FloatType(), optional=True),
        }

    @typecheck()
    def forward(self, source, padding_mask=None, mask=True, features_only=False) -> tuple:
        prob_ppl_loss, cur_codebook_temp = None, None

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features_penalty = features.float().pow(2).mean()  # L2 Norm on features

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            extra = padding_mask.size(1) % features.size(1)
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
            padding_mask = padding_mask.all(-1)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        if self.input_quantizer:
            features, prob_ppl_loss, cur_codebook_temp = self.input_quantizer(features)
            features = self.project_inp(features)
        if mask:
            logits, mask_indices = self.apply_mask(features, padding_mask)
            if mask_indices is not None:
                targets = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
            else:
                targets = unmasked_features
        else:
            logits = features
            targets = unmasked_features
            mask_indices = None

        logits = self.encoder(logits, padding_mask=padding_mask)

        if features_only:
            return logits, padding_mask

        if self.quantizer:
            targets, prob_ppl_loss, cur_codebook_temp = self.quantizer(targets)
            targets = self.project_q(targets)

            if self.negatives_from_everywhere:
                neg_cands, *_ = self.quantizer(unmasked_features)
                sampled_negatives, _ = self.sample_negatives(neg_cands, targets.size(1))
                sampled_negatives = self.project_q(sampled_negatives)
            else:
                sampled_negatives, _ = self.sample_negatives(targets, targets.size(1))

            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(
                    targets.size(0) * targets.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, targets.size(0), targets.size(1), -1
                )  # order doesnt matter
                cb_negs = self.project_q(cb_negs)
                sampled_negatives = torch.cat([sampled_negatives, cb_negs], dim=0)
        else:
            targets = self.project_q(targets)

            if self.negatives_from_everywhere:
                sampled_negatives, _ = self.sample_negatives(unmasked_features, targets.size(1))
                sampled_negatives = self.project_q(sampled_negatives)
            else:
                sampled_negatives, _ = self.sample_negatives(targets, targets.size(1))

        logits = logits[mask_indices].view(logits.size(0), -1, logits.size(-1))

        if self.target_glu:
            targets = self.target_glu(targets)
            sampled_negatives = self.target_glu(sampled_negatives)

        logits = self.final_proj(logits)

        return logits, targets, sampled_negatives, padding_mask, features_penalty, prob_ppl_loss, cur_codebook_temp

    def extract_features(self, source, audio_lengths, mask=False):
        padding_mask = self._create_padding_mask(audio_lengths)
        return self(source=source, padding_mask=padding_mask, mask=mask, features_only=True)

    def remove_pretraining_modules(self):
        self.quantizer = None
        self.project_q = None
        self.target_glu = None
        self.final_proj = None

    def _update_quantizer_temp(self):
        if self.quantizer:
            self.quantizer.set_num_updates(self.trainer.global_step)
        if self.input_quantizer:
            self.input_quantizer.set_num_updates(self.trainer.global_step)

    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape
        if self.mask_cfg.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_cfg.mask_prob,
                self.mask_cfg.mask_length,
                self.mask_cfg.mask_type,
                self.mask_cfg.mask_other,
                min_masks=2,
                no_overlap=self.mask_cfg.no_mask_overlap,
                min_space=self.mask_cfg.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            mask_emb = self.mask_emb.type_as(x)
            x[mask_indices] = mask_emb
        else:
            mask_indices = None

        if self.mask_cfg.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_cfg.mask_channel_prob,
                self.mask_cfg.mask_channel_length,
                self.mask_cfg.mask_channel_type,
                self.mask_cfg.mask_channel_other,
                no_overlap=self.mask_cfg.no_mask_channel_overlap,
                min_space=self.mask_cfg.mask_channel_min_space,
            )
            mask_channel_indices = torch.from_numpy(mask_channel_indices).to(x.device).unsqueeze(1).expand(-1, T, -1)
            x[mask_channel_indices] = 0

        return x, mask_indices

    def sample_negatives(self, y, num):

        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        cross_high = tsz * bsz
        high = tsz
        with torch.no_grad():
            assert high > 1, f"{bsz, tsz, fsz}"

            if self.n_negatives > 0:
                tszs = buffered_arange(num).unsqueeze(-1).expand(-1, self.n_negatives).flatten()

                neg_idxs = torch.randint(low=0, high=high - 1, size=(bsz, self.n_negatives * num))
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = buffered_arange(num).unsqueeze(-1).expand(-1, self.cross_sample_negatives).flatten()

                cross_neg_idxs = torch.randint(
                    low=0, high=cross_high - 1, size=(bsz, self.cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(bsz, num, self.n_negatives + self.cross_sample_negatives, fsz).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs

    def _create_padding_mask(self, audio_lengths):
        # Broadcast to vectorize creating the padding mask
        max_len = max(audio_lengths)
        padding_mask = torch.arange(max_len, device=self.device)
        padding_mask = padding_mask.expand(len(audio_lengths), max_len) < audio_lengths.unsqueeze(1)
        # Negate to false where no padding
        padding_mask = ~padding_mask
        return padding_mask

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'is_tarred' in train_data_config and train_data_config['is_tarred']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

    def _setup_dataloader_from_config(self, config: Optional[Dict]):

        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        shuffle = config['shuffle']
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        if config.get('use_dali', False):
            device_id = self.local_rank if device == 'gpu' else None
            dataset = audio_to_text_dataset.get_dali_char_dataset(
                config=config,
                shuffle=shuffle,
                device_id=device_id,
                global_rank=self.global_rank,
                world_size=self.world_size,
                preprocessor_cfg=self._cfg.preprocessor,
            )
            return dataset

        # Instantiate tarred dataset loader or normal dataset loader
        if config.get('is_tarred', False):
            if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
                'manifest_filepath' in config and config['manifest_filepath'] is None
            ):
                logging.warning(
                    "Could not load dataset as `manifest_filepath` was None or "
                    f"`tarred_audio_filepaths` is None. Provided config : {config}"
                )
                return None

            shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
            dataset = audio_to_text_dataset.get_tarred_char_dataset(
                config=config,
                shuffle_n=shuffle_n,
                global_rank=self.global_rank,
                world_size=self.world_size,
                augmentor=augmentor,
            )
            shuffle = False
        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
                return None

            dataset = audio_to_text_dataset.get_char_dataset(config=config, augmentor=augmentor)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=dataset.collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )
