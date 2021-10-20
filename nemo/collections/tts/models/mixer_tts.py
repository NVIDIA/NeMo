# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List, Optional

import numpy as np
import omegaconf
import torch
import transformers
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.nn import functional as F
from transformers import AlbertTokenizer

from nemo.collections.tts.helpers.helpers import (
    binarize_attention_parallel,
    get_mask_from_lengths,
    plot_pitch_to_numpy,
    plot_spectrogram_to_numpy,
)
from nemo.collections.tts.losses.aligner_loss import BinLoss, ForwardSumLoss
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.collections.tts.modules.fastpitch import average_pitch, regulate_len
from nemo.collections.tts.torch.data import MixerTTSDataset
from nemo.collections.tts.torch.tts_tokenizers import EnglishPhonemesTokenizer, EnglishCharsTokenizer
from nemo.core.classes import typecheck
from nemo.utils import logging


class MixerTTSModel(SpectrogramGenerator):
    """MixerTTS pipeline."""

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        super().__init__(cfg=cfg, trainer=trainer)
        typecheck.set_typecheck_enabled(enabled=False)
        cfg = self._cfg

        self.tokenizer = instantiate(cfg.train_ds.dataset.text_tokenizer)
        num_tokens = len(self.tokenizer.tokens)
        self.tokenizer_unk = self.tokenizer.oov

        self.pitch_loss_scale, self.durs_loss_scale = cfg.pitch_loss_scale, cfg.durs_loss_scale
        self.mel_loss_scale = cfg.get("mel_loss_scale", 1.0)

        self.aligner = instantiate(cfg.alignment_module)
        self.forward_sum_loss = ForwardSumLoss()
        self.bin_loss = BinLoss()
        self.add_bin_loss = False
        self.bin_loss_scale = 0.0
        self.bin_loss_start_ratio = cfg.bin_loss_start_ratio
        self.bin_loss_warmup_epochs = cfg.bin_loss_warmup_epochs

        self.conditioning_on_nlp_model_text_encoder = cfg.get("conditioning_on_nlp_model_text_encoder", False)
        self.conditioning_on_nlp_model_text_decoder = cfg.get("conditioning_on_nlp_model_text_decoder", False)
        self.extract_nlp_features_via_nlp_aligner = cfg.get("extract_nlp_features_via_nlp_aligner", False)
        self.use_only_nlp_features_for_encoder = cfg.get("use_only_nlp_features_for_encoder", False)

        nlp_cond_flags = [
            self.conditioning_on_nlp_model_text_encoder,
            self.conditioning_on_nlp_model_text_decoder,
            self.extract_nlp_features_via_nlp_aligner,
            self.use_only_nlp_features_for_encoder
        ]
        assert sum(nlp_cond_flags) <= 1, "use only one NLP conditioning flag"
        self.is_cond_on_nlp = sum(nlp_cond_flags) == 1

        if self.is_cond_on_nlp:
            self.nlp_padding_value = (
                self._train_dl.dataset.nlp_padding_value
                if self._train_dl is not None
                else self._get_nlp_padding_value(cfg.train_ds.dataset.nlp_model)
            )
            self.nlp_model_text_proj = self._get_nlp_embeddings(cfg.train_ds.dataset.nlp_model)
            self.nlp_model_text_proj.weight.requires_grad = True

            if self.conditioning_on_nlp_model_text_encoder or self.conditioning_on_nlp_model_text_decoder or self.use_only_nlp_features_for_encoder:
                self.nlp_emb = instantiate(cfg.nlp_emb, in_channels=self.nlp_model_text_proj.weight.shape[1])

            if self.extract_nlp_features_via_nlp_aligner:
                self.nlp_aligner = instantiate(
                    cfg.nlp_aligner, n_nlp_channels=self.nlp_model_text_proj.weight.shape[1]
                )

        if cfg.encoder._target_ == "nemo.collections.tts.modules.mixer_tts.TTSMixerModule":
            self.encoder_type = "mixer"
            self.encoder = instantiate(
                cfg.encoder,
                num_tokens=num_tokens if not self.use_only_nlp_features_for_encoder else -1,
                padding_idx=self.tokenizer.pad
            )
            self.symbol_emb = self.encoder.to_embed
        elif cfg.encoder._target_ == "nemo.collections.tts.modules.transformer.FFTransformerEncoder":
            self.encoder_type = "transformer"
            self.encoder = instantiate(
                cfg.encoder,
                n_embed=num_tokens if not self.use_only_nlp_features_for_encoder else -1,
                padding_idx=self.tokenizer.pad
            )
            self.symbol_emb = self.encoder.word_emb
        else:
            raise ValueError

        self.duration_predictor = instantiate(cfg.duration_predictor)

        self.pitch_mean, self.pitch_std = float(cfg.pitch_mean), float(cfg.pitch_std)
        self.pitch_predictor = instantiate(cfg.pitch_predictor)
        self.pitch_emb = instantiate(cfg.pitch_emb)

        self.preprocessor = instantiate(cfg.preprocessor)

        self.decoder = instantiate(cfg.decoder)

        if cfg.decoder._target_ == "nemo.collections.tts.modules.mixer_tts.TTSMixerModule":
            self.decoder_type = "mixer"
        elif cfg.decoder._target_ == "nemo.collections.tts.modules.transformer.FFTransformerDecoder":
            self.decoder_type = "transformer"
        else:
            raise ValueError

        self.proj = nn.Linear(self.decoder.d_model, cfg.n_mel_channels)

    def load_state_dict(self, state_dict, strict=True):
        new_sd = {}
        for k, v in state_dict.items():
            new_sd[k] = v

        # Fix for conv1d/conv2d/NHWC
        curr_sd = self.state_dict()
        for key in new_sd:
            len_diff = len(new_sd[key].size()) - len(curr_sd[key].size())
            if len_diff == -1:
                if "linear" in key:
                    new_sd[key] = new_sd[key].unsqueeze(-1)
                else:
                    new_sd[key] = new_sd[key].unsqueeze(-2)
            elif len_diff == 1:
                new_sd[key] = new_sd[key].squeeze(-1)

        super().load_state_dict(new_sd, strict=strict)

    def _get_nlp_embeddings(self, nlp_model):
        if nlp_model == "albert":
            return transformers.AlbertModel.from_pretrained('albert-base-v2').embeddings.word_embeddings
        else:
            raise NotImplementedError(
                f"{nlp_model} nlp model is not supported. Only albert is supported at this moment."
            )

    def _get_nlp_padding_value(self, nlp_model):
        if nlp_model == "albert":
            return transformers.AlbertTokenizer.from_pretrained('albert-base-v2')._convert_token_to_id('<pad>')
        else:
            raise NotImplementedError(
                f"{nlp_model} nlp model is not supported. Only albert is supported at this moment."
            )

    def _metrics(
        self,
        true_durs,
        true_text_len,
        pred_durs,
        true_pitch,
        pred_pitch,
        true_spect=None,
        pred_spect=None,
        true_spect_len=None,
        attn_logprob=None,
        attn_soft=None,
        attn_hard=None,
        attn_hard_dur=None,
        nlp_len=None
    ):
        text_mask = get_mask_from_lengths(true_text_len if not self.use_only_nlp_features_for_encoder else nlp_len)
        mel_mask = get_mask_from_lengths(true_spect_len)
        loss = 0.0

        # dur loss and metrics
        durs_loss = F.mse_loss(pred_durs, (true_durs + 1).float().log(), reduction='none')
        durs_loss = durs_loss * text_mask.float()
        durs_loss = durs_loss.sum() / text_mask.sum()

        durs_pred = pred_durs.exp() - 1
        durs_pred = torch.clamp_min(durs_pred, min=0)
        durs_pred = durs_pred.round().long()

        acc = ((true_durs == durs_pred) * text_mask).sum().float() / text_mask.sum() * 100
        acc_dist_1 = (((true_durs - durs_pred).abs() <= 1) * text_mask).sum().float() / text_mask.sum() * 100
        acc_dist_3 = (((true_durs - durs_pred).abs() <= 3) * text_mask).sum().float() / text_mask.sum() * 100

        pred_spect = pred_spect.transpose(1, 2)

        # mel loss
        mel_loss = F.mse_loss(pred_spect, true_spect, reduction='none').mean(dim=-2)
        mel_loss = mel_loss * mel_mask.float()
        mel_loss = mel_loss.sum() / mel_mask.sum()

        loss = loss + self.durs_loss_scale * durs_loss + self.mel_loss_scale * mel_loss

        # aligner loss
        bin_loss, ctc_loss = None, None
        ctc_loss = self.forward_sum_loss(
            attn_logprob=attn_logprob,
            in_lens=true_text_len if not self.use_only_nlp_features_for_encoder else nlp_len,
            out_lens=true_spect_len
        )
        loss = loss + ctc_loss
        if self.add_bin_loss:
            bin_loss = self.bin_loss(hard_attention=attn_hard, soft_attention=attn_soft)
            loss = loss + self.bin_loss_scale * bin_loss
        true_avg_pitch = average_pitch(true_pitch.unsqueeze(1), attn_hard_dur).squeeze(1)

        # pitch loss
        pitch_loss = F.mse_loss(pred_pitch, true_avg_pitch, reduction='none')  # noqa
        pitch_loss = (pitch_loss * text_mask).sum() / text_mask.sum()

        loss = loss + self.pitch_loss_scale * pitch_loss

        return loss, durs_loss, acc, acc_dist_1, acc_dist_3, pitch_loss, mel_loss, ctc_loss, bin_loss

    def forward(self, text, text_len, pitch=None, spect=None, spect_len=None, attn_prior=None, nlp_tokens=None,
                nlp_attn_prior=None):
        if self.training:
            assert pitch is not None

        text_mask = get_mask_from_lengths(text_len).unsqueeze(2)

        if self.encoder_type == "transformer":
            enc_out, enc_mask = self.encoder(input=text, conditioning=0)
        elif self.encoder_type == "mixer":
            if self.conditioning_on_nlp_model_text_encoder:
                nlp_mask = (nlp_tokens != self.nlp_padding_value).unsqueeze(2)
                nlp_emb = self.nlp_model_text_proj(nlp_tokens) * nlp_mask
                nlp_emb = self.nlp_emb(nlp_emb.transpose(1, 2)).transpose(1, 2) * nlp_mask
                enc_out, enc_mask = self.encoder(text, text_mask, conditioning=nlp_emb)
            elif self.use_only_nlp_features_for_encoder:
                nlp_mask = (nlp_tokens != self.nlp_padding_value).unsqueeze(2)
                nlp_emb = self.nlp_model_text_proj(nlp_tokens) * nlp_mask
                nlp_emb = self.nlp_emb(nlp_emb.transpose(1, 2)).transpose(1, 2) * nlp_mask
                enc_out, enc_mask = self.encoder(nlp_emb, nlp_mask)
            else:
                enc_out, enc_mask = self.encoder(text, text_mask)
        else:
            raise NotImplementedError

        # aligner
        nlp_len = None
        if self.use_only_nlp_features_for_encoder:
            attn_soft, attn_logprob = self.aligner(
                spect, nlp_emb.permute(0, 2, 1), mask=nlp_mask == 0, attn_prior=nlp_attn_prior,
            )
            # it is assumed that zeros in nlp_mask is consecutive and only at the end
            nlp_len = (nlp_mask.squeeze(2) != 0).sum(dim=-1)
            attn_hard = binarize_attention_parallel(attn_soft, nlp_len, spect_len)
        else:
            text_emb = self.symbol_emb(text)
            attn_soft, attn_logprob = self.aligner(
                spect, text_emb.permute(0, 2, 1), mask=text_mask == 0, attn_prior=attn_prior,
            )
            attn_hard = binarize_attention_parallel(attn_soft, text_len, spect_len)

        attn_hard_dur = attn_hard.sum(2)[:, 0, :]
        assert torch.all(torch.eq(attn_hard_dur.sum(dim=1), spect_len))

        # extract text nlp features via nlp aligner
        if self.extract_nlp_features_via_nlp_aligner:
            nlp_proj = self.nlp_model_text_proj(nlp_tokens)
            nlp_features = self.nlp_aligner(
                enc_out, nlp_proj, nlp_proj, q_mask=enc_mask.squeeze(2), kv_mask=nlp_tokens != self.nlp_padding_value
            )

        # duration predictor
        log_durs_predicted = self.duration_predictor(enc_out, enc_mask)
        durs_predicted = torch.clamp(log_durs_predicted.exp() - 1, 0)

        # pitch predictor
        pitch_predicted = self.pitch_predictor(enc_out, enc_mask)

        # avg pitch, add pitch_emb
        if not self.training:
            if pitch is not None:
                pitch = average_pitch(pitch.unsqueeze(1), attn_hard_dur).squeeze(1)
                pitch_emb = self.pitch_emb(pitch.unsqueeze(1))
            else:
                pitch_emb = self.pitch_emb(pitch_predicted.unsqueeze(1))
        else:
            pitch = average_pitch(pitch.unsqueeze(1), attn_hard_dur).squeeze(1)
            pitch_emb = self.pitch_emb(pitch.unsqueeze(1))

        enc_out = enc_out + pitch_emb.transpose(1, 2)

        # conditioning on text through nlp model, add nlp_emb
        if self.conditioning_on_nlp_model_text_decoder:
            nlp_mask = (nlp_tokens != self.nlp_padding_value).unsqueeze(2)
            nlp_emb = self.nlp_model_text_proj(nlp_tokens) * nlp_mask
            nlp_emb = self.nlp_emb(nlp_emb.transpose(1, 2)).transpose(1, 2) * nlp_mask
            enc_out = enc_out + nlp_emb

        # add text nlp features from nlp aligner
        if self.extract_nlp_features_via_nlp_aligner:
            enc_out = enc_out + nlp_features

        # regulate length
        len_regulated_enc_out, dec_lens = regulate_len(attn_hard_dur, enc_out)

        if self.decoder_type == "transformer":
            dec_out, _ = self.decoder(input=len_regulated_enc_out, seq_lens=dec_lens)
        elif self.decoder_type == "mixer":
            dec_out, dec_lens = self.decoder(len_regulated_enc_out, get_mask_from_lengths(dec_lens).unsqueeze(2))
        else:
            raise NotImplementedError

        pred_spect = self.proj(dec_out)

        return (
            pred_spect,
            durs_predicted,
            log_durs_predicted,
            pitch_predicted,
            attn_soft,
            attn_logprob,
            attn_hard,
            attn_hard_dur,
            nlp_len
        )

    # TODO(Oktai): change according to use_only_nlp_features_for_encoder
    def infer(
        self,
        text,
        text_len,
        spect=None,
        spect_len=None,
        attn_prior=None,
        use_gt_durs=False,
        nlp_tokens=None,
        pitch=None,
    ):
        text_mask = get_mask_from_lengths(text_len).unsqueeze(2)

        if self.encoder_type == "transformer":
            enc_out, enc_mask = self.encoder(input=text, conditioning=0)
        elif self.encoder_type == "mixer":
            if self.conditioning_on_nlp_model_text_encoder:
                nlp_mask = (nlp_tokens != self.nlp_padding_value).unsqueeze(2)
                nlp_emb = self.nlp_model_text_proj(nlp_tokens) * nlp_mask
                nlp_emb = self.nlp_emb(nlp_emb.transpose(1, 2)).transpose(1, 2) * nlp_mask
                enc_out, enc_mask = self.encoder(text, text_mask, conditioning=nlp_emb)
            else:
                enc_out, enc_mask = self.encoder(text, text_mask)
        else:
            raise NotImplementedError

        # aligner
        attn_hard_dur = None
        if use_gt_durs and spect is not None and spect_len is not None and attn_prior is not None:
            if attn_prior.shape[1] != spect.shape[2]:
                # TODO(otatanov): need to delete it
                logging.warning(f"bad attn prior is detected: atth_prior shape: {attn_prior}, spect shape: {spect}")
                spect = spect[:, :, : attn_prior.shape[1]]
                spect_len[...] = attn_prior.shape[1]

            text_emb = self.symbol_emb(text)
            attn_soft, _ = self.aligner(spect, text_emb.permute(0, 2, 1), mask=text_mask == 0, attn_prior=attn_prior,)
            attn_hard = binarize_attention_parallel(attn_soft, text_len, spect_len)
            attn_hard_dur = attn_hard.sum(2)[:, 0, :]
            assert torch.all(torch.eq(attn_hard_dur.sum(dim=1), spect_len))

        # extract text nlp features via nlp aligner
        if self.extract_nlp_features_via_nlp_aligner:
            nlp_proj = self.nlp_model_text_proj(nlp_tokens)
            nlp_features = self.nlp_aligner(
                enc_out, nlp_proj, nlp_proj, q_mask=enc_mask.squeeze(2), kv_mask=nlp_tokens != self.nlp_padding_value
            )

        # duration predictor
        log_durs_predicted = self.duration_predictor(enc_out, enc_mask)
        durs_predicted = torch.clamp(log_durs_predicted.exp() - 1, 0)

        # avg pitch, pitch predictor
        if use_gt_durs and pitch is not None:
            pitch = average_pitch(pitch.unsqueeze(1), attn_hard_dur).squeeze(1)
            pitch_emb = self.pitch_emb(pitch.unsqueeze(1))
        else:
            pitch_predicted = self.pitch_predictor(enc_out, enc_mask)
            pitch_emb = self.pitch_emb(pitch_predicted.unsqueeze(1))

        # add pitch emb
        enc_out = enc_out + pitch_emb.transpose(1, 2)

        # conditioning on nlp model decoder
        if self.conditioning_on_nlp_model_text_decoder:
            nlp_mask = (nlp_tokens != self.nlp_padding_value).unsqueeze(2)
            nlp_emb = self.nlp_model_text_proj(nlp_tokens) * nlp_mask
            nlp_emb = self.nlp_emb(nlp_emb.transpose(1, 2)).transpose(1, 2) * nlp_mask
            enc_out = enc_out + nlp_emb

        # add text nlp features from nlp aligner
        if self.extract_nlp_features_via_nlp_aligner:
            enc_out = enc_out + nlp_features

        if use_gt_durs:
            if attn_hard_dur is not None:
                len_regulated_enc_out, dec_lens = regulate_len(attn_hard_dur, enc_out)
            else:
                raise NotImplementedError
        else:
            len_regulated_enc_out, dec_lens = regulate_len(durs_predicted, enc_out)

        if self.decoder_type == "transformer":
            dec_out, _ = self.decoder(input=len_regulated_enc_out, seq_lens=dec_lens)
        elif self.decoder_type == "mixer":
            dec_out, _ = self.decoder(len_regulated_enc_out, get_mask_from_lengths(dec_lens).unsqueeze(2))
        else:
            raise NotImplementedError

        pred_spect = self.proj(dec_out)

        return pred_spect

    def on_train_epoch_start(self):
        bin_loss_start_epoch = np.ceil(self.bin_loss_start_ratio * self._trainer.max_epochs)

        # Add bin loss when current_epoch >= bin_start_epoch
        if not self.add_bin_loss and self.current_epoch >= bin_loss_start_epoch:
            logging.info(f"Using hard attentions after epoch: {self.current_epoch}")
            self.add_bin_loss = True

        if self.add_bin_loss:
            self.bin_loss_scale = min((self.current_epoch - bin_loss_start_epoch) / self.bin_loss_warmup_epochs, 1.0)

    def training_step(self, batch, batch_idx):
        attn_prior, nlp_tokens, nlp_attn_prior = None, None, None
        if (
            self.conditioning_on_nlp_model_text_encoder
            or self.conditioning_on_nlp_model_text_decoder
            or self.extract_nlp_features_via_nlp_aligner
        ):
            audio, audio_len, text, text_len, attn_prior, pitch, _, nlp_tokens = batch
        elif self.use_only_nlp_features_for_encoder:
            audio, audio_len, text, text_len, attn_prior, pitch, _, nlp_tokens, nlp_attn_prior = batch
        else:
            audio, audio_len, text, text_len, attn_prior, pitch, _ = batch

        spect, spect_len = self.preprocessor(input_signal=audio, length=audio_len)

        # pitch normalization
        zero_pitch_idx = pitch == 0
        pitch = (pitch - self.pitch_mean) / self.pitch_std
        pitch[zero_pitch_idx] = 0.0

        (pred_spect, _, pred_log_durs, pred_pitch, attn_soft, attn_logprob, attn_hard, attn_hard_dur, nlp_len, ) = self(
            text=text,
            text_len=text_len,
            pitch=pitch,
            spect=spect,
            spect_len=spect_len,
            attn_prior=attn_prior,
            nlp_tokens=nlp_tokens,
            nlp_attn_prior=nlp_attn_prior
        )

        (loss, durs_loss, acc, acc_dist_1, acc_dist_3, pitch_loss, mel_loss, ctc_loss, bin_loss,) = self._metrics(
            pred_durs=pred_log_durs,
            pred_pitch=pred_pitch,
            true_durs=attn_hard_dur,
            true_text_len=text_len,
            true_pitch=pitch,
            true_spect=spect,
            pred_spect=pred_spect,
            true_spect_len=spect_len,
            attn_logprob=attn_logprob,
            attn_soft=attn_soft,
            attn_hard=attn_hard,
            attn_hard_dur=attn_hard_dur,
            nlp_len=nlp_len
        )

        train_log = {
            'train_loss': loss,
            'train_durs_loss': durs_loss,
            'train_pitch_loss': torch.tensor(1.0).to(durs_loss.device) if pitch_loss is None else pitch_loss,
            'train_mel_loss': mel_loss,
            'train_durs_acc': acc,
            'train_durs_acc_dist_3': acc_dist_3,
            'train_ctc_loss': torch.tensor(1.0).to(durs_loss.device) if ctc_loss is None else ctc_loss,
            'train_bin_loss': torch.tensor(1.0).to(durs_loss.device) if bin_loss is None else bin_loss,
        }

        return {'loss': loss, 'progress_bar': train_log, 'log': train_log}

    def validation_step(self, batch, batch_idx):
        attn_prior, nlp_tokens, nlp_attn_prior = None, None, None
        if (
            self.conditioning_on_nlp_model_text_encoder
            or self.conditioning_on_nlp_model_text_decoder
            or self.extract_nlp_features_via_nlp_aligner
        ):
            audio, audio_len, text, text_len, attn_prior, pitch, _, nlp_tokens = batch
        elif self.use_only_nlp_features_for_encoder:
            audio, audio_len, text, text_len, attn_prior, pitch, _, nlp_tokens, nlp_attn_prior = batch
        else:
            audio, audio_len, text, text_len, attn_prior, pitch, _ = batch

        spect, spect_len = self.preprocessor(input_signal=audio, length=audio_len)

        # pitch normalization
        zero_pitch_idx = pitch == 0
        pitch = (pitch - self.pitch_mean) / self.pitch_std
        pitch[zero_pitch_idx] = 0.0

        (pred_spect, _, pred_log_durs, pred_pitch, attn_soft, attn_logprob, attn_hard, attn_hard_dur, nlp_len, ) = self(
            text=text,
            text_len=text_len,
            pitch=pitch,
            spect=spect,
            spect_len=spect_len,
            attn_prior=attn_prior,
            nlp_tokens=nlp_tokens,
            nlp_attn_prior=nlp_attn_prior
        )

        (loss, durs_loss, acc, acc_dist_1, acc_dist_3, pitch_loss, mel_loss, ctc_loss, bin_loss,) = self._metrics(
            pred_durs=pred_log_durs,
            pred_pitch=pred_pitch,
            true_durs=attn_hard_dur,
            true_text_len=text_len,
            true_pitch=pitch,
            true_spect=spect,
            pred_spect=pred_spect,
            true_spect_len=spect_len,
            attn_logprob=attn_logprob,
            attn_soft=attn_soft,
            attn_hard=attn_hard,
            attn_hard_dur=attn_hard_dur,
            nlp_len=nlp_len
        )

        # without ground truth internal features except for durations
        pred_spect, _, pred_log_durs, pred_pitch, attn_soft, attn_logprob, attn_hard, attn_hard_dur, nlp_len = self(
            text=text,
            text_len=text_len,
            pitch=None,
            spect=spect,
            spect_len=spect_len,
            attn_prior=attn_prior,
            nlp_tokens=nlp_tokens,
            nlp_attn_prior=nlp_attn_prior,
        )

        *_, with_pred_features_mel_loss, _, _ = self._metrics(
            pred_durs=pred_log_durs,
            pred_pitch=pred_pitch,
            true_durs=attn_hard_dur,
            true_text_len=text_len,
            true_pitch=pitch,
            true_spect=spect,
            pred_spect=pred_spect,
            true_spect_len=spect_len,
            attn_logprob=attn_logprob,
            attn_soft=attn_soft,
            attn_hard=attn_hard,
            attn_hard_dur=attn_hard_dur,
            nlp_len=nlp_len
        )

        val_log = {
            'val_loss': loss,
            'val_durs_loss': durs_loss,
            'val_pitch_loss': torch.tensor(1.0).to(durs_loss.device) if pitch_loss is None else pitch_loss,
            'val_mel_loss': mel_loss,
            'val_with_pred_features_mel_loss': with_pred_features_mel_loss,
            'val_durs_acc': acc,
            'val_durs_acc_dist_3': acc_dist_3,
            'val_ctc_loss': torch.tensor(1.0).to(durs_loss.device) if ctc_loss is None else ctc_loss,
            'val_bin_loss': torch.tensor(1.0).to(durs_loss.device) if bin_loss is None else bin_loss,
        }
        self.log_dict(val_log, prog_bar=False, on_epoch=True, logger=True, sync_dist=True)

        if batch_idx == 0 and self.current_epoch % 5 == 0 and isinstance(self.logger, WandbLogger):
            specs = []
            pitches = []
            for i in range(min(3, spect.shape[0])):
                specs += [
                    wandb.Image(
                        plot_spectrogram_to_numpy(spect[i, :, : spect_len[i]].data.cpu().numpy()),
                        caption=f"gt mel {i}",
                    ),
                    wandb.Image(
                        plot_spectrogram_to_numpy(pred_spect.transpose(1, 2)[i, :, : spect_len[i]].data.cpu().numpy()),
                        caption=f"pred mel {i}",
                    ),
                ]

                pitches += [
                    wandb.Image(
                        plot_pitch_to_numpy(
                            average_pitch(pitch.unsqueeze(1), attn_hard_dur)
                            .squeeze(1)[i, : text_len[i]]
                            .data.cpu()
                            .numpy(),
                            ylim_range=[-2.5, 2.5],
                        ),
                        caption=f"gt pitch {i}",
                    ),
                ]

                pitches += [
                    wandb.Image(
                        plot_pitch_to_numpy(pred_pitch[i, : text_len[i]].data.cpu().numpy(), ylim_range=[-2.5, 2.5]),
                        caption=f"pred pitch {i}",
                    ),
                ]

            self.logger.experiment.log({"specs": specs, "pitches": pitches})

    def nlp_model_tokenizer(self):
        if getattr(self, "_nlp_model_tokenizer", None) is not None:
            return self._nlp_model_tokenizer

        if self._train_dl is not None and self._train_dl.dataset is not None:
            self._nlp_model_tokenizer = self._train_dl.dataset.nlp_model_tokenizer

        # TODO(otatanov): rewrite it
        self._nlp_model_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

        return self._nlp_model_tokenizer

    def generate_spectrogram(
        self,
        tokens: Optional[torch.Tensor] = None,
        tokens_len: Optional[torch.Tensor] = None,
        nlp_tokens: Optional[torch.Tensor] = None,
        raw_texts: Optional[List[str]] = None,
        **kwargs,
    ):
        if tokens is not None:
            if tokens_len is None:
                # it is assumed that padding is consecutive and only at the end
                tokens_len = (tokens != self.tokenizer.pad).sum(dim=-1)
        else:
            if raw_texts is None:
                logging.error("raw_texts must be specified if tokens is None")

            t_seqs = [self.tokenizer(t) for t in raw_texts]
            tokens = torch.nn.utils.rnn.pad_sequence(
                sequences=[torch.tensor(t, dtype=torch.long, device=self.device) for t in t_seqs],
                batch_first=True,
                padding_value=self.tokenizer.pad,
            )
            tokens_len = torch.tensor([len(t) for t in t_seqs], dtype=torch.long, device=tokens.device)

        if self.is_cond_on_nlp and nlp_tokens is None:
            if raw_texts is None:
                logging.error("raw_texts must be specified if nlp_tokens is None")

            nlp_model_tokenizer = self.nlp_model_tokenizer()
            nlp_padding_value = nlp_model_tokenizer._convert_token_to_id('<pad>')
            nlp_space_value = nlp_model_tokenizer._convert_token_to_id('▁')

            assert isinstance(self.tokenizer, EnglishCharsTokenizer) or isinstance(self.tokenizer, EnglishPhonemesTokenizer)

            preprocess_texts_as_tts_input = [
                self.tokenizer.g2p.text_preprocessing_func(t)
                if isinstance(self.tokenizer, EnglishPhonemesTokenizer)
                else self.tokenizer.text_preprocessing_func(t)
                for t in raw_texts
            ]
            nlp_tokens_as_ids_list = [
                nlp_model_tokenizer.encode(t, add_special_tokens=False) for t in preprocess_texts_as_tts_input
            ]

            if self.tokenizer.pad_with_space:
                nlp_tokens_as_ids_list = [
                    [nlp_space_value] + t + [nlp_space_value] for t in nlp_tokens_as_ids_list
                ]

            nlp_tokens = torch.full(
                (len(nlp_tokens_as_ids_list), max([len(t) for t in nlp_tokens_as_ids_list])),
                fill_value=nlp_padding_value,
                device=tokens.device,
            )
            for i, nlp_tokens_i in enumerate(nlp_tokens_as_ids_list):
                nlp_tokens[i, : len(nlp_tokens_i)] = torch.tensor(nlp_tokens_i, device=tokens.device)

        pred_spect = self.infer(tokens, tokens_len, nlp_tokens=nlp_tokens)
        return pred_spect

    def parse(self, text: str, **kwargs) -> torch.Tensor:
        return torch.tensor(self.tokenizer.encode(text)).long().unsqueeze(0).to(self.device)

    @staticmethod
    def _loader(cfg):
        try:
            _ = cfg.dataset.manifest_filepath
        except omegaconf.errors.MissingMandatoryValue:
            logging.warning("manifest_filepath was skipped. No dataset for this model.")
            return None

        dataset = instantiate(cfg.dataset)
        return torch.utils.data.DataLoader(  # noqa
            dataset=dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params,
        )

    def setup_training_data(self, cfg):
        self._train_dl = self._loader(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self._loader(cfg)

    def setup_test_data(self, cfg):
        """Omitted."""
        pass

    @classmethod
    def list_available_models(cls):
        """Empty."""
        pass
