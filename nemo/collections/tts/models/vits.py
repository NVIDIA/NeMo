import omegaconf
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.cuda.amp import autocast
from torch.nn import functional as F
import wandb

from nemo.collections.tts.helpers.helpers import plot_spectrogram_to_numpy
from nemo.collections.tts.losses.vits_losses import DiscriminatorLoss, FeatureLoss, GeneratorLoss, KlLoss
from nemo.collections.tts.models.base import TextToWaveform
from nemo.collections.tts.modules.vits_modules import SynthesizerTrn, MultiPeriodDiscriminator, audio_to_mel_torch, spec_to_mel_torch, slice_segments, clip_grad_value_
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging, model_utils

# TODO: remove if not needed
# to call optimizer_step
# def closure():
#     return


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
        self.tokenizer_unk = self.tokenizer.oov
        
        super().__init__(cfg=cfg, trainer=trainer)
        
        self.audio_to_melspec_precessor = instantiate(cfg.preprocessor, highfreq=cfg.train_ds.dataset.highfreq)

        # TODO: how model knows padding idx? num tokens?
        self.generator = instantiate(cfg.generator)
        self.multiperioddisc = MultiPeriodDiscriminator()
        self.feat_matching_loss = FeatureLoss()
        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.kl_loss = KlLoss()

        self.log_train_images = False
        self.logged_real_samples = False
        self._tb_logger = None
        self.hann_window = None
        self.sample_rate = cfg.sample_rate
        self.hop_size = cfg.n_window_stride
        self.n_fft = cfg.train_ds.dataset.n_fft
        self.win_length = cfg.train_ds.dataset.win_length

        self.net_g = SynthesizerTrn(
            n_vocab = cfg.symbols_embedding_dim,
            spec_channels = cfg.train_ds.dataset.n_fft // 2 + 1,
            segment_size = cfg.segment_size // cfg.train_ds.dataset.hop_length,
            inter_channels = cfg.inter_channels,
            hidden_channels = cfg.hidden_channels,
            filter_channels = cfg.filter_channels,
            n_heads = cfg.n_heads,
            n_layers = cfg.n_layers,
            kernel_size = cfg.pitch_embedding_kernel_size,
            p_dropout = cfg.p_dropout,
            resblock = cfg.generator.resblock,
            resblock_kernel_sizes = cfg.generator.resblock_kernel_sizes,
            resblock_dilation_sizes = cfg.generator.resblock_dilation_sizes,
            upsample_rates = cfg.generator.upsample_rates,
            upsample_initial_channel = cfg.generator.upsample_initial_channel,
            upsample_kernel_sizes = cfg.generator.upsample_kernel_sizes,
            )
        self.net_d = MultiPeriodDiscriminator(cfg.use_spectral_norm)
        self.automatic_optimization = False

        window_fn = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'none': None,
        }.get(self.hann_window, None)

        self.stft = lambda x: torch.stft(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=window_fn(self.win_length, periodic=False).to(torch.float) if window_fn else None,
        )

        # TODO: remove if not needed
        # self.precision_plugin = self.trainer.accelerator.precision_plugin # to call optimizer_step

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
        if "g2p" in cfg.text_tokenizer:
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

    def parse(self, str_input: str) -> torch.tensor:
        # TODO: Implement
        pass

    def configure_optimizers(self):
        optim_g = torch.optim.AdamW(
            self.net_g.parameters(),
            self._cfg.lr,
            betas=self._cfg.betas,
            eps=self._cfg.eps)
        optim_d = torch.optim.AdamW(
            self.net_d.parameters(),
            self._cfg.lr,
            betas=self._cfg.betas,
            eps=self._cfg.eps)

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optim_g, gamma=self._cfg.lr_decay)
        scheduler_g_dict = {
            'scheduler': scheduler_g,
            'interval': 'step',
        }
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=self._cfg.lr_decay)
        scheduler_d_dict = {
            'scheduler': scheduler_d,
            'interval': 'step'
        }
        return [optim_g, optim_d], [scheduler_g_dict, scheduler_d_dict]

    def forward(self, batch, batch_idx):
        # TODO: Check if this is correct
        with torch.no_grad():
            (x, x_lengths, spec, spec_lengths, y, y_lengths) = batch

            # remove else
            x = x[:1]
            x_lengths = x_lengths[:1]

            y_hat, attn, mask, *_ = self.net_g.module.infer(x, x_lengths, max_len=1000)
            y_hat_lengths = mask.sum([1, 2]).long() * self._cfg.hop_size
        
        return y_hat, y_hat_lengths

    def get_spec(self, audio):
        with torch.cuda.amp.autocast(enabled=False):
            spec = self.stft(audio)
            if spec.dtype in [torch.cfloat, torch.cdouble]:
                spec = torch.view_as_real(spec)
            spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
        return spec

    def training_step(self, batch, batch_idx):
        # TODO: support accum gradient or don't allow to use accum gradient in init

        (y, y_lengths, x, x_lengths) = batch

        spec = self.get_spec(y)
        spec_lengths = self.audio_to_melspec_precessor.get_seq_len(y_lengths)

        with autocast(enabled=True):
            y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q) = self.net_g(x, x_lengths, spec, spec_lengths)
        
            mel = spec_to_mel_torch(
                spec,
                self._cfg.filter_length,
                self._cfg.n_mel_channels,
                self._cfg.sample_rate,
                self._cfg.mel_fmin,
                self._cfg.mel_fmax,
            )
            y_mel = slice_segments(mel, ids_slice, self._cfg.segment_size // self.cfg.n_window_stride)
        
        y_hat = y_hat.float()
        y_hat_mel = audio_to_mel_torch(
            y_hat.squeeze(1),
            self._cfg.filter_length,
            self._cfg.n_mel_channels,
            self._cfg.sample_rate,
            self.cfg.n_window_stride,
            self._cfg.preprocessor.n_window_size,
            self._cfg.mel_fmin,
            self._cfg.mel_fmax,
        )

        y = torch.unsqueeze(y, 1)
        y = slice_segments(y, ids_slice * self.cfg.n_window_stride, self._cfg.segment_size)
        
        y_d_hat_r, y_d_hat_g, _, _ = self.net_d(y, y_hat.detach())
        with autocast(enabled=False):
            loss_disc, losses_disc_r, losses_disc_g = self.disc_loss(y_d_hat_r, y_d_hat_g)
            loss_disc_all = loss_disc

        # get optimizers
        optimizers, _ = self.optimizers()
        optim_g, optim_d = optimizers

        # train discriminator
        optim_d.zero_grad()
        self.manual_backward(loss_disc_all)
        optim_d.step()
        # TODO: maybe change it to PTL-based function
        norm_d = clip_grad_value_(self.net_d.parameters(), None)

        with autocast(enabled=True):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, y_hat)
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
        optim_g.step()
        norm_g = clip_grad_value_(self.net_g.parameters(), None)

        schedulers = self.lr_schedulers()
        if schedulers is not None:
            sch1, sch2 = schedulers
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
            "grad_gen" : norm_g,
            "grad_disc" : norm_d,
        }

        for i, v in enumerate(losses_gen):
            metrics[f"loss_gen_i_{i}"] = v

        for i, v in enumerate(losses_disc_r):
            metrics[f"loss_disc_r_{i}"] = v

        for i, v in enumerate(losses_disc_g):
            metrics[f"loss_disc_g_{i}"] = v

        self.log_dict(metrics, on_step=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        (y, y_lengths, x, x_lengths) = batch

        # TODO: fix hardcode
        y_hat, attn, mask, *_ = self.net_g.infer(x, x_lengths, max_len=1000)
        y_hat = y_hat.squeeze()
        y_hat_lengths = mask.sum([1, 2]).long() * self._cfg.train_ds.dataset.hop_length

        mel, mel_lengths = self.audio_to_melspec_precessor(y, y_lengths)
        y_hat_mel, y_hat_mel_lengths = self.audio_to_melspec_precessor(y_hat, y_hat_lengths)

        # plot audio once per epoch
        if batch_idx == 0 and self.logger is not None and isinstance(self.logger, WandbLogger):
            specs = []
            audios = []

            specs += [
                wandb.Image(
                    plot_spectrogram_to_numpy(mel[0, :, : mel_lengths[0]].cpu().numpy()),
                    caption=f"val_mel_target",
                ),
                wandb.Image(
                    plot_spectrogram_to_numpy(y_hat_mel[0, :, : y_hat_mel_lengths[0]].cpu().numpy()),
                    caption=f"val_mel_predicted",
                ),
            ]

            audios += [
                wandb.Audio(
                    y[0, : y_lengths[0]].data.cpu().numpy(),
                    caption=f"val_wav_target",
                    sample_rate=self.sample_rate,
                ),
                wandb.Audio(
                    y_hat[0, : y_hat_lengths[0]].data.cpu().numpy(),
                    caption=f"val_wav_predicted",
                    sample_rate=self.sample_rate,
                ),
            ]

            self.logger.experiment.log({"specs": specs, "audios": audios})

    def _loader(self, cfg):
        try:
            # _ = cfg.model.train_ds.manifest_filepath
            _ = cfg['dataset']['manifest_filepath']
        except omegaconf.errors.MissingMandatoryValue:
            logging.warning("manifest_filepath was skipped. No dataset for this model.")
            return None

        dataset = instantiate(
            cfg.dataset,
            text_normalizer=self.normalizer,
            text_normalizer_call_kwargs=self.text_normalizer_call_kwargs,
            text_tokenizer=self.tokenizer
            )
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
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        list_of_models = []
        # TODO: List available models??
        return list_of_models

    def convert_text_to_waveform(self, *, tokens):
        #  TODO: Convert text to waveforms
        pass

