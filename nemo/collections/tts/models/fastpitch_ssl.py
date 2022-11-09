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
import os
import random
from typing import Optional

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger

from nemo.collections.asr.parts.preprocessing import features
from nemo.collections.tts.helpers.helpers import plot_multipitch_to_numpy, plot_spectrogram_to_numpy
from nemo.collections.tts.losses.fastpitchloss import DurationLoss, MelLoss, PitchLoss
from nemo.collections.tts.models import ssl_tts
from nemo.collections.tts.modules.fastpitch import FastPitchModule, average_pitch
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import MelSpectrogramType
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging, model_utils


def mask_from_lens(lens, max_len: Optional[int] = None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask


class FastPitchModel_SSL(ModelPT):
    """FastPitch model (https://arxiv.org/abs/2006.06873) that is used to generate mel spectrogram from text."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None, vocoder=None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        self.learn_alignment = False

        self._parser = None
        self._tb_logger = None
        super().__init__(cfg=cfg, trainer=trainer)

        self.bin_loss_warmup_epochs = cfg.get("bin_loss_warmup_epochs", 100)
        self.log_train_images = False

        loss_scale = 0.1 if self.learn_alignment else 1.0
        dur_loss_scale = loss_scale
        pitch_loss_scale = loss_scale
        if "dur_loss_scale" in cfg:
            dur_loss_scale = cfg.dur_loss_scale
        if "pitch_loss_scale" in cfg:
            pitch_loss_scale = cfg.pitch_loss_scale

        self.mel_loss = MelLoss()
        self.pitch_loss = PitchLoss(loss_scale=pitch_loss_scale)
        self.duration_loss = DurationLoss(loss_scale=dur_loss_scale)

        self.aligner = None

        # can probably drop the preprocessor
        self.preprocessor = instantiate(self._cfg.preprocessor)
        input_fft = None
        self.use_encoder = use_encoder = cfg.get("use_encoder", False)
        if use_encoder:
            self.encoder = instantiate(self._cfg.encoder)

        output_fft = instantiate(self._cfg.output_fft)

        duration_predictor = None
        self.use_duration_predictor = cfg.get("use_duration_predictor")
        if self.use_duration_predictor:
            assert self.encoder is not None, "use_encoder must be True if use_duration_predictor is True"
            # this means we are using unique tokens
            duration_predictor = instantiate(self._cfg.duration_predictor)

        self.pitch_conditioning = pitch_conditioning = cfg.get("pitch_conditioning", True)
        if pitch_conditioning:
            pitch_predictor = instantiate(self._cfg.pitch_predictor)
        else:
            pitch_predictor = None

        self.content_projection_layer = torch.nn.Linear(self._cfg.content_emb_indim, self._cfg.content_emb_outdim)
        self.speaker_projection_layer = torch.nn.Linear(self._cfg.speaker_emb_indim, self._cfg.speaker_emb_outdim)

        self.num_datasets = cfg.get("n_datasets", 1)
        if self.num_datasets > 1:
            self.dataset_embedding_layer = torch.nn.Embedding(self.num_datasets, self._cfg.symbols_embedding_dim)

        self.fastpitch = FastPitchModule(
            input_fft,
            output_fft,
            duration_predictor,
            pitch_predictor,
            self.aligner,
            cfg.n_speakers,
            cfg.symbols_embedding_dim,
            cfg.pitch_embedding_kernel_size,
            cfg.n_mel_channels,
        )

        self.non_trainable_models = {}
        use_speaker_loss = self._cfg.get("speaker_loss", False)
        self.use_speaker_loss = use_speaker_loss
        if use_speaker_loss:
            if os.path.exists(self._cfg.ssl_model_ckpt_path):
                print("Loading SSL model from ", self._cfg.ssl_model_ckpt_path)
                # print("Device: ", self.device, self.fastpitch.device)
                ssl_model = ssl_tts.SSLDisentangler.load_from_checkpoint(
                    self._cfg.ssl_model_ckpt_path, strict=False
                ).cpu()
                ssl_model.eval()
                self.non_trainable_models['ssl_model'] = ssl_model
            else:
                print("SSL model ckpt does not exist")

        self.non_trainable_models['vocoder'] = vocoder

    def vocode_spectrogram(self, spectrogram):
        # spectrogram [C, T] numpy
        with torch.no_grad():
            _spec = torch.from_numpy(spectrogram).unsqueeze(0).to(torch.float32).to(self.device)
            wav_generated = self.non_trainable_models['vocoder'].generator(x=_spec)[0]
            return wav_generated.cpu().numpy()

    @property
    def tb_logger(self):
        if self._tb_logger is None:
            if self.logger is None and self.logger.experiment is None:
                return None
            tb_logger = self.logger.experiment
            if isinstance(self.logger, LoggerCollection):
                for logger in self.logger:
                    if isinstance(logger, TensorBoardLogger):
                        tb_logger = logger.experiment
                        break
            self._tb_logger = tb_logger
        return self._tb_logger

    def forward(
        self,
        *,
        text,
        durs=None,
        pitch=None,
        speaker=None,
        pace=1.0,
        spec=None,
        attn_prior=None,
        mel_lens=None,
        input_lens=None,
        enc_out=None,
        enc_mask=None,
    ):
        return self.fastpitch(
            text=text,
            durs=durs,
            pitch=pitch,
            speaker=speaker,
            pace=pace,
            spec=spec,
            attn_prior=attn_prior,
            mel_lens=mel_lens,
            input_lens=input_lens,
            enc_out=enc_out,
            enc_mask=enc_mask,
        )

    @typecheck(output_types={"spect": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType())})
    def generate_spectrogram(
        self, tokens: 'torch.tensor', speaker: Optional[int] = None, pace: float = 1.0
    ) -> torch.tensor:
        if self.training:
            logging.warning("generate_spectrogram() is meant to be called in eval mode.")
        if isinstance(speaker, int):
            speaker = torch.tensor([speaker]).to(self.device)
        spect, *_ = self(text=tokens, durs=None, pitch=None, speaker=speaker, pace=pace)
        return spect

    def compute_encoding(self, content_embedding, speaker_embedding, dataset_id=None):
        # content embedding is (B, C, T)
        # speaker embedding is (B, C)
        # pitch_contour is (B, T)
        content_embedding = content_embedding.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
        content_embedding_projected = self.content_projection_layer(content_embedding)
        content_embedding_projected = content_embedding_projected.permute(0, 2, 1)
        speaker_embedding_projected = self.speaker_projection_layer(speaker_embedding)
        speaker_embedding_repeated = speaker_embedding_projected[:, :, None].repeat(
            1, 1, content_embedding_projected.shape[2]
        )

        encoded = torch.cat([content_embedding_projected, speaker_embedding_repeated], dim=1)

        encoded = encoded.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)

        if self.num_datasets > 1:
            dataset_embedding = self.dataset_embedding_layer(dataset_id)  # (B, C)
            dataset_embedding_repeated = dataset_embedding[:, None, :].repeat(1, encoded.shape[1], 1)
            encoded = encoded + dataset_embedding_repeated

        return encoded

    def compute_speaker_loss(self, generated_mel, gt_speaker_embedding):
        ssl_model = self.non_trainable_models['ssl_model']
        ssl_model.to(self.device)

        generated_mel_sliced = generated_mel[:, :, :172]
        generated_mel_len_batch = (
            torch.tensor([generated_mel_sliced.shape[-1] for _idx in range(generated_mel_sliced.shape[0])])
            .long()
            .to(self.device)
        )

        normalized_mel = features.normalize_batch(generated_mel_sliced, generated_mel_len_batch, "per_feature")

        encoded, _ = ssl_model.encoder(audio_signal=normalized_mel, length=generated_mel_len_batch)  # b,c,t
        pred_speaker_embedding = ssl_model.downstream_nets['speaker_verification'](encoded[:, :, 0])
        l2_norm = torch.norm(pred_speaker_embedding, p=2, dim=-1, keepdim=True)
        speaker_embedding_normalized = pred_speaker_embedding / l2_norm

        cosine_similarity = torch.nn.functional.cosine_similarity(
            speaker_embedding_normalized, gt_speaker_embedding, dim=-1
        ).mean()

        speaker_loss = 1 - cosine_similarity
        return speaker_loss

    def training_step(self, batch, batch_idx):
        content_embedding = batch["content_embedding"]
        encoded_len = batch["encoded_len"]
        speaker_embedding = batch["speaker_embedding"]
        mels = batch["mel_spectrogram"]
        spec_len = batch["mel_len"]
        pitch = batch["pitch_contour"]
        dataset_id = batch["dataset_id"]
        durs = batch["duration"]

        enc_out = self.compute_encoding(content_embedding, speaker_embedding, dataset_id)
        if self.use_encoder:
            enc_out, _ = self.encoder(input=enc_out, seq_lens=encoded_len)

        enc_mask = mask_from_lens(encoded_len)
        enc_mask = enc_mask[:, :, None]

        mels_pred, _, _, log_durs_pred, pitch_pred, attn_soft, attn_logprob, attn_hard, attn_hard_dur, pitch = self(
            text=None,
            durs=durs,
            pitch=pitch,
            speaker=None,
            pace=1.0,
            spec=None,
            attn_prior=None,
            mel_lens=spec_len,
            input_lens=None,
            enc_out=enc_out,
            enc_mask=enc_mask,
        )

        loss = 0
        mel_loss = self.mel_loss(spect_predicted=mels_pred, spect_tgt=mels)
        loss += mel_loss
        if self.use_duration_predictor:
            dur_loss = self.duration_loss(log_durs_predicted=log_durs_pred, durs_tgt=durs, len=encoded_len)
            self.log("t_dur_loss", dur_loss)
            loss += dur_loss

        if self.pitch_conditioning:
            pitch_loss = self.pitch_loss(pitch_predicted=pitch_pred, pitch_tgt=pitch, len=encoded_len)
            loss += pitch_loss
            self.log("t_pitch_loss", pitch_loss)

        if self.use_speaker_loss:
            speaker_loss = self.compute_speaker_loss(mels_pred, speaker_embedding)
            loss += speaker_loss
            self.log("t_speaker_loss", speaker_loss)

        self.log("t_loss", loss)
        self.log("t_mel_loss", mel_loss)

        # Log images to tensorboard
        if self.log_train_images and isinstance(self.logger, TensorBoardLogger):
            self.log_train_images = False

            self.tb_logger.add_image(
                "train_mel_target",
                plot_spectrogram_to_numpy(mels[0].data.cpu().float().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            spec_predict = mels_pred[0].data.cpu().float().numpy()
            self.tb_logger.add_image(
                "train_mel_predicted", plot_spectrogram_to_numpy(spec_predict), self.global_step, dataformats="HWC",
            )

        return loss

    def validation_step(self, batch, batch_idx):
        content_embedding = batch["content_embedding"]
        encoded_len = batch["encoded_len"]
        speaker_embedding = batch["speaker_embedding"]
        mels = batch["mel_spectrogram"]
        spec_len = batch["mel_len"]
        pitch = batch["pitch_contour"]
        dataset_id = batch["dataset_id"]
        durs = batch["duration"]

        enc_out = self.compute_encoding(content_embedding, speaker_embedding, dataset_id)
        if self.use_encoder:
            enc_out, _ = self.encoder(input=enc_out, seq_lens=encoded_len)

        enc_mask = mask_from_lens(encoded_len)
        enc_mask = enc_mask[:, :, None]

        mels_pred, _, _, log_durs_pred, pitch_pred, _, _, _, attn_hard_dur, pitch = self(
            text=None,
            durs=durs,
            pitch=pitch,
            speaker=None,
            pace=1.0,
            spec=None,
            attn_prior=None,
            mel_lens=spec_len,
            input_lens=None,
            enc_out=enc_out,
            enc_mask=enc_mask,
        )

        mel_loss = self.mel_loss(spect_predicted=mels_pred, spect_tgt=mels)

        val_out = {
            "val_loss": mel_loss,
            "mel_loss": mel_loss,
            "mel_target": mels if batch_idx == 0 else None,
            "mel_pred": mels_pred if batch_idx == 0 else None,
            "spec_len": spec_len if batch_idx == 0 else None,
            "pitch_target": pitch if batch_idx == 0 else None,
            "pitch_pred": pitch_pred if batch_idx == 0 else None,
        }

        if self.use_duration_predictor:
            dur_loss = self.duration_loss(log_durs_predicted=log_durs_pred, durs_tgt=durs, len=encoded_len)
            val_out["dur_loss"] = dur_loss

        if self.pitch_conditioning:
            pitch_loss = self.pitch_loss(pitch_predicted=pitch_pred, pitch_tgt=pitch, len=encoded_len)
            val_out["pitch_loss"] = pitch_loss
            val_out["val_loss"] = mel_loss + pitch_loss

        return val_out

    def validation_epoch_end(self, outputs):
        collect = lambda key: torch.stack([x[key] for x in outputs]).mean()
        val_loss = collect("val_loss")
        mel_loss = collect("mel_loss")

        self.log("v_loss", val_loss)
        self.log("v_mel_loss", mel_loss)

        if self.pitch_conditioning:
            pitch_loss = collect("pitch_loss")
            self.log("v_pitch_loss", pitch_loss)

        single_output = outputs[0]
        spec_target = single_output['mel_target']
        spec_predict = single_output['mel_pred']
        spec_len = single_output['spec_len']
        pitch_target = single_output['pitch_target']
        pitch_pred = single_output['pitch_pred']

        if isinstance(self.logger, TensorBoardLogger):
            _rand_idx = random.randint(0, spec_target.shape[0] - 1)
            self.tb_logger.add_image(
                "val_mel_target",
                plot_spectrogram_to_numpy(spec_target[_rand_idx].data.cpu().float().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            spec_predict = spec_predict[_rand_idx].data.cpu().float().numpy()
            self.tb_logger.add_image(
                "val_mel_predicted", plot_spectrogram_to_numpy(spec_predict), self.global_step, dataformats="HWC",
            )

            if self.pitch_conditioning:
                _pitch_pred = pitch_pred[_rand_idx].data.cpu().numpy()
                _pitch_target = pitch_target[_rand_idx].data.cpu().numpy()

                self.tb_logger.add_image(
                    "val_pitch",
                    plot_multipitch_to_numpy(_pitch_target, _pitch_pred),
                    self.global_step,
                    dataformats="HWC",
                )

            _spec_len = spec_len[_rand_idx].data.cpu().item()
            wav_vocoded = self.vocode_spectrogram(spec_target[_rand_idx].data.cpu().float().numpy()[:, :_spec_len])
            self.tb_logger.add_audio("Real audio", wav_vocoded, self.global_step, 22050)

            wav_vocoded = self.vocode_spectrogram(spec_predict[:, :_spec_len])
            self.tb_logger.add_audio("Generated Audio", wav_vocoded, self.global_step, 22050)
            self.log_train_images = True

    def synthesize_wav(
        self,
        content_embedding,
        speaker_embedding,
        encoded_len=None,
        pitch_contour=None,
        compute_pitch=False,
        compute_duration=False,
        durs_gt=None,
        dataset_id=0,
    ):
        """
        content_embedding : (B, C, T)
        speaker_embedding : (B, C)
        pitch_contour : (B, T*4) (mel pitch) or None
        compute_pitch: if true, predict pitch contour from content and speaker embedding.
        """
        _bs, _, _n_time = content_embedding.size()
        if encoded_len is None:
            encoded_len = (torch.ones(_bs) * _n_time).long().to(self.device)

        dataset_id = (torch.ones(_bs) * dataset_id).long().to(self.device)
        enc_out = self.compute_encoding(content_embedding, speaker_embedding, dataset_id=dataset_id)
        if self.use_encoder:
            enc_out, _ = self.encoder(input=enc_out, seq_lens=encoded_len)

        enc_mask = mask_from_lens(encoded_len)

        if compute_duration:
            durs = None
        elif durs_gt is not None:
            durs = durs_gt
        else:
            durs = torch.ones_like(enc_mask) * 4.0

        enc_mask = enc_mask[:, :, None]
        if pitch_contour is not None and compute_pitch == False:
            if durs_gt is not None:
                pitch = average_pitch(pitch_contour.unsqueeze(1), durs_gt).squeeze(1)
            elif durs is not None:
                pitch = average_pitch(pitch_contour.unsqueeze(1), durs).squeeze(1)
            else:
                raise ValueError("durs or durs_gt must be provided")

        else:
            pitch = None

        mels_pred, _, _, log_durs_pred, pitch_pred, _, _, _, attn_hard_dur, pitch = self(
            text=None,
            durs=durs,
            pitch=pitch,
            speaker=None,
            pace=1.0,
            spec=None,
            attn_prior=None,
            mel_lens=None,
            input_lens=None,
            enc_out=enc_out,
            enc_mask=enc_mask,
        )

        wavs = []
        for idx in range(_bs):
            mel_pred = mels_pred[idx].data.cpu().float().numpy()
            # _mel_len = int(encoded_len[idx].item() * 4)
            # mel_pred = mel_pred[:, :_mel_len]
            wav = self.vocode_spectrogram(mel_pred)
            wavs.append(wav)

        return wavs

    def __setup_dataloader_from_config(self, cfg):
        dataset = instantiate(cfg.dataset)

        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.pad_collate_fn, **cfg.dataloader_params)

    def setup_training_data(self, cfg):
        self._train_dl = self.__setup_dataloader_from_config(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self.__setup_dataloader_from_config(cfg)

    def setup_test_data(self, cfg):
        """Omitted."""
        pass

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        return []
