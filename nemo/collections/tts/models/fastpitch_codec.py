# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import contextlib
from einops import rearrange
from pathlib import Path
from typing import List

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.tts.data.text_to_speech_dataset import create_text_to_speech_dataset
from nemo.collections.tts.losses.aligner_loss import BinLoss, ForwardSumLoss
from nemo.collections.tts.losses.fastpitch_codec_loss import AudioTokenLoss, MaskedMSELoss
from nemo.collections.tts.models.audio_codec import AudioCodecModel
from nemo.collections.tts.modules.fastpitch_codec_modules import FastPitchCodecModule
from nemo.collections.tts.parts.utils.callbacks import LoggingCallback
from nemo.collections.tts.parts.utils.helpers import average_features, load_model
from nemo.core import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import (
    Index,
    LengthsType,
    LogitsType,
    LogprobsType,
    PredictionsType,
    ProbsType,
    RegressionValuesType,
    TokenIndex,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging, model_utils
from nemo.utils.decorators import experimental


@experimental
class FastPitchCodecModel(ModelPT):

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        self.text_tokenizer = self._create_tokenizer(cfg.text_tokenizer)

        super().__init__(cfg=cfg, trainer=trainer)

        audio_codec_name = cfg.get("audio_codec_name")
        audio_codec_path = cfg.get("audio_codec_path")
        self.audio_codec = load_model(
            model_type=AudioCodecModel,
            device=str(self.device),
            model_name=audio_codec_name,
            checkpoint_path=audio_codec_path
        )
        self.audio_codec.freeze()

        aligner = instantiate(cfg.alignment_module)

        num_text_emb = len(self.text_tokenizer.tokens)
        pad_token = self.text_tokenizer.pad
        encoder = instantiate(cfg.encoder, n_embed=num_text_emb, padding_idx=pad_token)

        decoder = instantiate(cfg.decoder)
        duration_predictor = instantiate(cfg.duration_predictor)
        pitch_predictor = instantiate(cfg.pitch_predictor)
        energy_predictor = instantiate(cfg.energy_predictor)
        speaker_encoder = instantiate(cfg.get("speaker_encoder", None))

        num_codebooks = cfg.get("n_codebooks")
        codebook_size = cfg.get("codebook_size")

        self.fastpitch = FastPitchCodecModule(
            aligner_module=aligner,
            encoder_module=encoder,
            decoder_module=decoder,
            duration_predictor=duration_predictor,
            pitch_predictor=pitch_predictor,
            energy_predictor=energy_predictor,
            speaker_encoder=speaker_encoder,
            n_codebooks=num_codebooks,
            codebook_size=codebook_size,
        )

        self.dur_loss_scale = cfg.get("dur_loss_scale", 0.1)
        self.pitch_loss_scale = cfg.get("pitch_loss_scale", 0.1)
        self.energy_loss_scale = cfg.get("energy_loss_scale", 0.1)
        self.aligner_loss_scale = cfg.get("aligner_loss_scale", 0.1)
        self.bin_loss_warmup_epochs = cfg.get("bin_loss_warmup_epochs", 100)

        self.audio_token_loss_fn = AudioTokenLoss(num_codebooks=num_codebooks)
        self.pitch_loss_fn = MaskedMSELoss()
        self.duration_loss_fn = MaskedMSELoss()
        self.energy_loss_fn = MaskedMSELoss()
        self.forward_sum_loss_fn = ForwardSumLoss()
        self.bin_loss_fn = BinLoss()

        self.log_config = cfg.get("log_config", None)

    def _create_tokenizer(self, tokenizer_config):
        if "phoneme_dict" in tokenizer_config.g2p:
            tokenizer_config.g2p.phoneme_dict = self.register_artifact(
                'text_tokenizer.g2p.phoneme_dict', tokenizer_config.g2p.phoneme_dict,
            )

        if "heteronyms" in tokenizer_config.g2p:
            tokenizer_config.g2p.heteronyms = self.register_artifact(
                'text_tokenizer.g2p.heteronyms', tokenizer_config.g2p.heteronyms,
            )

        text_tokenizer = instantiate(tokenizer_config)
        return text_tokenizer

    def parse(self, str_input: str) -> torch.tensor:
        if self.training:
            logging.warning("parse() is meant to be called in eval mode.")

        if not hasattr(self.text_tokenizer, "set_phone_prob"):
            text_tokens = self.text_tokenizer.encode(str_input)
        else:
            with self.text_tokenizer.set_phone_prob(prob=1.0):
                text_tokens = self.text_tokenizer.encode(str_input)

        token_tensor = torch.tensor(text_tokens).unsqueeze_(0).long().to(self.device)
        return token_tensor

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_lens": NeuralType(tuple('B'), LengthsType()),
            "pitch": NeuralType(('B', 'T_audio'), RegressionValuesType()),
            "energy": NeuralType(('B', 'T_audio'), RegressionValuesType()),
            "audio_tokens": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "audio_token_lens": NeuralType(tuple('B'), LengthsType()),
            "speaker": NeuralType(tuple('B'), Index(), optional=True),
            "attn_prior": NeuralType(('B', 'T_audio', 'T_text'), ProbsType(), optional=True),
        },
        output_types={
            "audio_tokens_pred": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "audio_logits": NeuralType(('B', 'C', 'W', 'T_audio'), LogitsType()),
            "log_durs": NeuralType(('B', 'T_text'), RegressionValuesType()),
            "log_durs_pred": NeuralType(('B', 'T_text'), PredictionsType()),
            "pitch_avg": NeuralType(('B', 'T_text'), RegressionValuesType()),
            "pitch_pred": NeuralType(('B', 'T_text'), PredictionsType()),
            "energy_avg": NeuralType(('B', 'T_text'), RegressionValuesType()),
            "energy_pred": NeuralType(('B', 'T_text'), PredictionsType()),
            "align_hard": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
            "align_soft": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
            "align_logits": NeuralType(('B', 'S', 'T_audio', 'T_text'), LogprobsType())
        }
    )
    def forward(
        self,
        text,
        text_lens,
        pitch,
        energy,
        audio_tokens,
        audio_token_lens,
        speaker=None,
        attn_prior=None,
    ):
        # [batch_size, code_dim, audio_token_len]
        audio_codes = self.audio_codec.dequantize(tokens=audio_tokens, tokens_len=audio_token_lens)

        # [batch_size, text_len], [batch_size, audio_token_len, text_len], ...
        durs, align_hard, align_soft, align_logits = self.fastpitch.get_alignments(
            text=text,
            text_lens=text_lens,
            audio_codes=audio_codes,
            audio_code_lens=audio_token_lens,
            attn_prior=attn_prior,
            speaker=speaker
        )
        log_durs = torch.log(1.0 + durs.float())

        pitch = rearrange(pitch, 'B T_audio -> B 1 T_audio')
        energy = rearrange(energy, 'B T_audio -> B 1 T_audio')
        pitch_avg = average_features(features=pitch, durs=durs)
        energy_avg = average_features(features=energy, durs=durs)
        pitch_avg = rearrange(pitch_avg, 'B 1 T_text -> B T_text')
        energy_avg = rearrange(energy_avg, 'B 1 T_text -> B T_text')

        audio_token_pred, audio_token_logits, log_durs_pred, pitch_pred, energy_pred = self.fastpitch(
            text=text,
            text_lens=text_lens,
            durs=durs,
            pitch=pitch_avg,
            energy=energy_avg,
            speaker=speaker,
        )
        return (
            audio_token_pred,
            audio_token_logits,
            log_durs,
            log_durs_pred,
            pitch_avg,
            pitch_pred,
            energy_avg,
            energy_pred,
            align_hard,
            align_soft,
            align_logits
        )

    def training_step(self, batch_dict, batch_idx):
        text = batch_dict.get("text")
        text_lens = batch_dict.get("text_lens")
        pitch = batch_dict.get("pitch")
        energy = batch_dict.get("energy")
        audio_tokens = batch_dict.get("audio_tokens")
        audio_token_lens = batch_dict.get("audio_token_lens")
        speaker = batch_dict.get("speaker_id", None)
        attn_prior = batch_dict.get("align_prior_matrix", None)

        (
            _,
            audio_token_logits,
            log_durs,
            log_durs_pred,
            pitch_avg,
            pitch_pred,
            energy_avg,
            energy_pred,
            align_hard,
            align_soft,
            align_logits
        ) = self(
            text=text,
            text_lens=text_lens,
            pitch=pitch,
            energy=energy,
            audio_tokens=audio_tokens,
            audio_token_lens=audio_token_lens,
            speaker=speaker,
            attn_prior=attn_prior,
        )

        audio_token_loss = self.audio_token_loss_fn(
            logits=audio_token_logits, target_tokens=audio_tokens, target_len=audio_token_lens
        )

        dur_loss = self.duration_loss_fn(predicted=log_durs_pred, target=log_durs.detach(), target_len=text_lens)
        train_dur_loss = self.dur_loss_scale * dur_loss

        pitch_loss = self.pitch_loss_fn(predicted=pitch_pred, target=pitch_avg.detach(), target_len=text_lens)
        train_pitch_loss = self.pitch_loss_scale * pitch_loss

        energy_loss = self.energy_loss_fn(predicted=energy_pred, target=energy_avg.detach(), target_len=text_lens)
        train_energy_loss = self.energy_loss_scale * energy_loss

        ctc_loss = self.forward_sum_loss_fn(attn_logprob=align_logits, in_lens=text_lens, out_lens=audio_token_lens)
        train_ctc_loss = self.aligner_loss_scale * ctc_loss

        bin_loss = self.bin_loss_fn(hard_attention=align_hard, soft_attention=align_soft)
        bin_loss_weight = 1.0 * min(self.current_epoch / self.bin_loss_warmup_epochs, 1.0)
        train_bin_loss = bin_loss_weight * self.aligner_loss_scale * bin_loss

        loss = audio_token_loss + train_dur_loss + train_pitch_loss + train_energy_loss + train_ctc_loss + train_bin_loss

        metrics = {
            "t_audio_token_loss": audio_token_loss,
            "t_dur_loss": dur_loss,
            "t_pitch_loss": pitch_loss,
            "t_energy_loss": energy_loss,
            "t_ctc_loss": ctc_loss,
            "t_bin_loss": bin_loss,
        }
        self.log_dict(metrics, on_step=True, sync_dist=True)
        self.log("t_loss", audio_token_loss, prog_bar=True, logger=False, sync_dist=True)

        return loss

    def validation_step(self, batch_dict, batch_idx):
        text = batch_dict.get("text")
        text_lens = batch_dict.get("text_lens")
        pitch = batch_dict.get("pitch")
        energy = batch_dict.get("energy")
        audio_tokens = batch_dict.get("audio_tokens")
        audio_token_lens = batch_dict.get("audio_token_lens")
        speaker = batch_dict.get("speaker_id", None)
        attn_prior = batch_dict.get("align_prior_matrix", None)

        (
            audio_token_pred,
            audio_token_logits,
            log_durs,
            log_durs_pred,
            pitch_avg,
            pitch_pred,
            energy_avg,
            energy_pred,
            align_hard,
            align_soft,
            align_logits
        ) = self(
            text=text,
            text_lens=text_lens,
            pitch=pitch,
            energy=energy,
            audio_tokens=audio_tokens,
            audio_token_lens=audio_token_lens,
            speaker=speaker,
            attn_prior=attn_prior,
        )

        audio_token_loss = self.audio_token_loss_fn(
            logits=audio_token_logits, target_tokens=audio_tokens, target_len=audio_token_lens
        )
        dur_loss = self.duration_loss_fn(predicted=log_durs_pred, target=log_durs, target_len=text_lens)
        pitch_loss = self.pitch_loss_fn(predicted=pitch_pred, target=pitch_avg, target_len=text_lens)
        energy_loss = self.energy_loss_fn(predicted=energy_pred, target=energy_avg, target_len=text_lens)
        loss = audio_token_loss + dur_loss + pitch_loss + energy_loss

        metrics = {
            "val_loss": loss,
            "val_audio_token_loss": audio_token_loss,
            "val_dur_loss": dur_loss,
            "val_pitch_loss": pitch_loss,
            "val_energy_loss": energy_loss,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

    def _setup_train_dataloader(self, dataset_config, dataloader_params):
        dataset = create_text_to_speech_dataset(
            dataset_type=dataset_config.dataset_type,
            text_tokenizer=self.text_tokenizer,
            global_rank=self.trainer.global_rank,
            world_size=self.trainer.world_size,
            dataset_args=dataset_config.dataset_args,
            is_train=True
        )

        sampler = dataset.get_sampler(dataloader_params.batch_size, world_size=self.trainer.world_size)
        return torch.utils.data.DataLoader(
            dataset, collate_fn=dataset.collate_fn, sampler=sampler, **dataloader_params
        )

    def _setup_test_dataloader(self, dataset_config, dataloader_params):
        dataset = create_text_to_speech_dataset(
            dataset_type=dataset_config.dataset_type,
            text_tokenizer=self.text_tokenizer,
            global_rank=self.trainer.global_rank,
            world_size=self.trainer.world_size,
            dataset_args=dataset_config.dataset_args,
            is_train=False
        )
        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **dataloader_params)

    def setup_training_data(self, cfg):
        self._train_dl = self._setup_train_dataloader(
            dataset_config=cfg.dataset, dataloader_params=cfg.dataloader_params
        )

    def setup_validation_data(self, cfg):
        self._validation_dl = self._setup_test_dataloader(
            dataset_config=cfg.dataset, dataloader_params=cfg.dataloader_params
        )

    def setup_test_data(self, cfg):
        """Omitted."""
        pass

    def configure_callbacks(self):
        if not self.log_config:
            return []

        data_loader = self._setup_test_dataloader(
            dataset_config=self.log_config.dataset, dataloader_params=self.log_config.dataloader_params
        )
        generators = instantiate(self.log_config.generators)
        log_dir = Path(self.log_config.log_dir) if self.log_config.log_dir else None
        log_callback = LoggingCallback(
            generators=generators,
            data_loader=data_loader,
            log_epochs=self.log_config.log_epochs,
            epoch_frequency=self.log_config.epoch_frequency,
            output_dir=log_dir,
            loggers=self.trainer.loggers,
            log_tensorboard=self.log_config.log_tensorboard,
            log_wandb=self.log_config.log_wandb,
        )

        return [log_callback]

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        return []
