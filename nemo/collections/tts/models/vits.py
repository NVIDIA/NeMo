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


import contextlib
import omegaconf
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.cuda.amp import autocast
from torch.nn import functional as F

from nemo.collections.tts.helpers.helpers import (
    slice_segments, 
    clip_grad_value_, 
    plot_spectrogram_to_numpy, 
)
from nemo.collections.tts.losses.vits_losses import (
    KlLoss, 
    FeatureMatchingLoss, 
    DiscriminatorLoss, 
    GeneratorLoss
)
from nemo.collections.tts.models.base import TextToWaveform
from nemo.collections.tts.modules.vits_modules import MultiPeriodDiscriminator
from nemo.collections.tts.torch.data import DistributedBucketSampler
from nemo.collections.tts.torch.tts_data_types import SpeakerID
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.optim.lr_scheduler import CosineAnnealing
from nemo.utils import logging, model_utils



HAVE_WANDB = True
try:
    import wandb
except ModuleNotFoundError:
    HAVE_WANDB = False

class VitsModel(TextToWaveform):
    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        # Convert to Hydra 1.0 compatible DictConfig

        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        # setup normalizer
        self.normalizer = None
        self.text_normalizer_call = None
        self.text_normalizer_call_kwargs = {}
        self._setup_normalizer(cfg)

        # setup tokenizer
        self.tokenizer = None
        self._setup_tokenizer(cfg)
        assert self.tokenizer is not None

        num_tokens = len(self.tokenizer.tokens)
        self.tokenizer_pad = self.tokenizer.pad

        super().__init__(cfg=cfg, trainer=trainer)

        self.audio_to_melspec_processor = instantiate(cfg.preprocessor, highfreq=cfg.train_ds.dataset.highfreq)

        self.feat_matching_loss = FeatureMatchingLoss()
        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.kl_loss = KlLoss()

        self.net_g = instantiate(cfg.synthesizer, 
            n_vocab=num_tokens,
            spec_channels=cfg.n_fft // 2 + 1,
            segment_size=cfg.segment_size // cfg.n_window_stride,
            padding_idx=self.tokenizer_pad,)
        
        self.net_d = MultiPeriodDiscriminator(cfg.use_spectral_norm)
        
        self.automatic_optimization = False

    def _setup_normalizer(self, cfg):
        if "text_normalizer" in cfg:
            normalizer_kwargs = {}

            if "whitelist" in cfg.text_normalizer:
                normalizer_kwargs["whitelist"] = self.register_artifact(
                    'text_normalizer.whitelist', cfg.text_normalizer.whitelist
                )

            self.normalizer = instantiate(cfg.text_normalizer, **normalizer_kwargs)
            self.text_normalizer_call = self.normalizer.normalize
            if "text_normalizer_call_kwargs" in cfg:
                self.text_normalizer_call_kwargs = cfg.text_normalizer_call_kwargs

    def _setup_tokenizer(self, cfg):
        text_tokenizer_kwargs = {}
        if "g2p" in cfg.text_tokenizer and cfg.text_tokenizer.g2p is not None:
            g2p_kwargs = {}

            if "phoneme_dict" in cfg.text_tokenizer.g2p:
                g2p_kwargs["phoneme_dict"] = self.register_artifact(
                    'text_tokenizer.g2p.phoneme_dict', cfg.text_tokenizer.g2p.phoneme_dict,
                )

            if "heteronyms" in cfg.text_tokenizer.g2p:
                g2p_kwargs["heteronyms"] = self.register_artifact(
                    'text_tokenizer.g2p.heteronyms', cfg.text_tokenizer.g2p.heteronyms,
                )

            text_tokenizer_kwargs["g2p"] = instantiate(cfg.text_tokenizer.g2p, **g2p_kwargs)

        self.tokenizer = instantiate(cfg.text_tokenizer, **text_tokenizer_kwargs)

    def parse(self, text: str, normalize=True) -> torch.tensor:
        if self.training:
            logging.warning("parse() is meant to be called in eval mode.")
        if normalize and self.text_normalizer_call is not None:
            text = self.text_normalizer_call(text, **self.text_normalizer_call_kwargs)

        eval_phon_mode = contextlib.nullcontext()
        if hasattr(self.tokenizer, "set_phone_prob"):
            eval_phon_mode = self.tokenizer.set_phone_prob(prob=1.0)

        with eval_phon_mode:
            tokens = self.tokenizer.encode(text)

        return torch.tensor(tokens).long().unsqueeze(0).to(self.device)

    def configure_optimizers(self):
        optim_config = self._cfg.optim.copy()
        OmegaConf.set_struct(optim_config, False)
        sched_config = optim_config.pop("sched", None)
        OmegaConf.set_struct(optim_config, True)

        optim_g = instantiate(optim_config, params=self.net_g.parameters(),)
        optim_d = instantiate(optim_config, params=self.net_d.parameters(),)
        
        if sched_config is not None:
            if sched_config.name == 'ExponentialLR':
                scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=sched_config.lr_decay)
                scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=sched_config.lr_decay)
            elif sched_config.name == 'CosineAnnealing':
                scheduler_g = CosineAnnealing(optimizer=optim_g, max_steps=sched_config.max_steps, min_lr=sched_config.min_lr,)
                scheduler_d = CosineAnnealing(optimizer=optim_d, max_steps=sched_config.max_steps, min_lr=sched_config.min_lr,)

            scheduler_g_dict = {'scheduler': scheduler_g, 'interval': 'step'}
            scheduler_d_dict = {'scheduler': scheduler_d, 'interval': 'step'}
            return [optim_g, optim_d], [scheduler_g_dict, scheduler_d_dict]
        else:
            return [optim_g, optim_d]

    # for inference
    def forward(self, tokens, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=1000):
        x_lengths = tokens.size(-1)
        y_hat = self.net_g.infer(tokens, x_lengths, sid=sid, noise_scale=noise_scale,
            length_scale=length_scale, noise_scale_w=noise_scale_w, max_len=max_len)[0]

        return y_hat

    def training_step(self, batch, batch_idx):
        speakers = None
        if SpeakerID in self._train_dl.dataset.sup_data_types_set:
            (y, y_lengths, x, x_lengths, speakers) = batch
        else:
            (y, y_lengths, x, x_lengths) = batch

        spec, spec_lengths = self.audio_to_melspec_processor(y, y_lengths, linear_spec=True)
        
        with autocast(enabled=True):
            y_hat, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = self.net_g(
                x, x_lengths, spec, spec_lengths, speakers
            )

        # y_mel = slice_segments(mel, ids_slice, self._cfg.segment_size // self.cfg.n_window_stride)
        y_hat = y_hat.float()

        y_hat_mel, _ = self.audio_to_melspec_processor(y_hat.squeeze(1), y_lengths, linear_spec=False)        
        
        y = slice_segments(y.unsqueeze(1), ids_slice * self.cfg.n_window_stride, self._cfg.segment_size)
        y_mel, _ = self.audio_to_melspec_processor(y.squeeze(1), y_lengths, linear_spec=False)
        
        with autocast(enabled=True):
            y_d_hat_r, y_d_hat_g, _, _ = self.net_d(y, y_hat.detach())

        with autocast(enabled=False):
            loss_disc, losses_disc_r, losses_disc_g = self.disc_loss(disc_real_outputs=y_d_hat_r, 
            disc_generated_outputs=y_d_hat_g)
            loss_disc_all = loss_disc

        # get optimizers
        optim_g, optim_d = self.optimizers()
        
        # train discriminator
        optim_d.zero_grad()
        self.manual_backward(loss_disc_all)
        norm_d = clip_grad_value_(self.net_d.parameters(), None)
        optim_d.step()
        
        with autocast(enabled=True):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, y_hat)
        # Generator
        with autocast(enabled=False):
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * self._cfg.c_mel
            loss_kl = self.kl_loss(z_p=z_p, logs_q=logs_q, m_p=m_p, logs_p=logs_p, z_mask=z_mask) * self._cfg.c_kl
            loss_fm = self.feat_matching_loss(fmap_r=fmap_r, fmap_g=fmap_g)
            loss_gen, losses_gen = self.gen_loss(disc_outputs=y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

        
        # train generator
        optim_g.zero_grad()
        self.manual_backward(loss_gen_all)
        norm_g = clip_grad_value_(self.net_g.parameters(), None)
        optim_g.step()

        schedulers = self.lr_schedulers()
        if schedulers is not None:
            sch1, sch2 = schedulers
            if self.trainer.is_last_batch and isinstance(sch1, 'torch.optim.lr_scheduler.ExponentialLR') \
            or isinstance(sch1, 'CosineAnnealing'):
                sch1.step()
                sch2.step()

        metrics = {
            "loss_gen": loss_gen,
            "loss_fm": loss_fm,
            "loss_mel * c_mel": loss_mel,
            "loss_dur": loss_dur,
            "loss_kl * c_kl": loss_kl,
            "loss_gen_all": loss_gen_all,
            "loss_disc_all": loss_disc_all,
            "grad_gen": norm_g,
            "grad_disc": norm_d,
        }

        for i, v in enumerate(losses_gen):
            metrics[f"loss_gen_i_{i}"] = v

        for i, v in enumerate(losses_disc_r):
            metrics[f"loss_disc_r_{i}"] = v

        for i, v in enumerate(losses_disc_g):
            metrics[f"loss_disc_g_{i}"] = v

        self.log_dict(metrics, on_step=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        speakers = None
        if self.cfg.n_speakers > 1:
            (y, y_lengths, x, x_lengths, speakers) = batch
        else:
            (y, y_lengths, x, x_lengths) = batch

        y_hat, attn, mask, *_ = self.net_g.infer(x, x_lengths, speakers, max_len=1000)

        y_hat = y_hat.squeeze()
        y_hat_lengths = mask.sum([1, 2]).long() * self._cfg.validation_ds.dataset.hop_length

        mel, mel_lengths = self.audio_to_melspec_processor(y, y_lengths)
        y_hat_mel, y_hat_mel_lengths = self.audio_to_melspec_processor(y_hat, y_hat_lengths)

        # plot audio once per epoch
        if batch_idx == 0 and isinstance(self.logger, WandbLogger) and HAVE_WANDB:
            logger = self.logger.experiment
            specs = []
            audios = []

            specs += [
                wandb.Image(
                    plot_spectrogram_to_numpy(mel[0, :, : mel_lengths[0]].data.cpu().numpy()), caption=f"val_mel_target",
                ),
                wandb.Image(
                    plot_spectrogram_to_numpy(y_hat_mel[0, :, : y_hat_mel_lengths[0]].data.cpu().numpy()),
                    caption=f"val_mel_predicted",
                ),
            ]

            audios += [
                wandb.Audio(
                    y[0, : y_lengths[0]].data.cpu().to(torch.float).numpy(),
                    caption=f"val_wav_target",
                    sample_rate=self._cfg.sample_rate,
                ),
                wandb.Audio(
                    y_hat[0, : y_hat_lengths[0]].data.cpu().to(torch.float).numpy(),
                    caption=f"val_wav_predicted",
                    sample_rate=self._cfg.sample_rate,
                ),
            ]

            logger.log({"specs": specs, "audios": audios})

    def _loader(self, cfg):
        try:
            _ = cfg['dataset']['manifest_filepath']
        except omegaconf.errors.MissingMandatoryValue:
            logging.warning("manifest_filepath was skipped. No dataset for this model.")
            return None

        dataset = instantiate(
            cfg.dataset,
            text_normalizer=self.normalizer,
            text_normalizer_call_kwargs=self.text_normalizer_call_kwargs,
            text_tokenizer=self.tokenizer,
        )
        return torch.utils.data.DataLoader(  # noqa
            dataset=dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params,
        )

    def train_dataloader(self):
    # default used by the Trainer
        dataset = instantiate(
            self.cfg.train_ds.dataset,
            text_normalizer=self.normalizer,
            text_normalizer_call_kwargs=self.text_normalizer_call_kwargs,
            text_tokenizer=self.tokenizer,
        )

        train_sampler = DistributedBucketSampler(
            dataset,
            **self.cfg.train_ds.batch_sampler)

        dataloader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, batch_sampler=train_sampler, 
        **self.cfg.train_ds.dataloader_params,)
        return dataloader
    
    def setup_training_data(self, cfg):
        self._train_dl = self._loader(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self._loader(cfg)

    def setup_test_data(self, cfg):
        """Omitted."""
        pass

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        list_of_models = []
        # TODO: List available models??
        return list_of_models

    def convert_text_to_waveform(self, *, tokens):
        return self(tokens).squeeze(1)
