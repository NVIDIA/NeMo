from itertools import chain

import librosa
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger
from torch import nn

from nemo.collections.asr.data.audio_to_text import AudioToCharWithDursF0Dataset
from nemo.collections.tts.helpers.helpers import get_mask_from_lengths, plot_spectrogram_to_numpy
from nemo.collections.tts.losses.hifigan_losses import DiscriminatorLoss, FeatureMatchingLoss, GeneratorLoss
from nemo.collections.tts.losses.tacotron2loss import L1MelLoss
from nemo.collections.tts.modules.fastspeech2 import Encoder, VarianceAdaptor
from nemo.collections.tts.modules.fastspeech2_submodules import LengthRegulator2, VariancePredictor
from nemo.collections.tts.modules.hifigan_modules import MultiPeriodDiscriminator, MultiScaleDiscriminator
from nemo.collections.tts.modules.talknet import GaussianEmbedding
from nemo.core.classes import ModelPT
from nemo.core.classes.common import typecheck
from nemo.core.optim.lr_scheduler import NoamAnnealing
from nemo.utils import logging


class DurationLoss(torch.nn.Module):
    def forward(self, duration_pred, duration_target, offset=1):
        log_duration_target = torch.log(duration_target + offset)
        return torch.nn.functional.mse_loss(duration_pred, log_duration_target)


class FastSpeech2SModel(ModelPT):
    """FastSpeech 2 model used to convert between text (phonemes) and mel-spectrograms."""

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        torch.backends.cudnn.benchmark = True
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        super().__init__(cfg=cfg, trainer=trainer)

        self.vocab = None
        if cfg.train_ds.dataset._target_ == "nemo.collections.asr.data.audio_to_text.AudioToCharWithDursF0Dataset":
            self.vocab = AudioToCharWithDursF0Dataset.make_vocab(**cfg.train_ds.dataset.vocab)
            self.phone_embedding = nn.Embedding(len(self.vocab.labels), 256, padding_idx=self.vocab.pad)
        else:
            self.phone_embedding = nn.Embedding(84, 256, padding_idx=83)
        self.energy = cfg.add_energy_predictor
        self.pitch = cfg.add_pitch_predictor

        self.mel_loss_coeff = 1.0
        if "mel_loss_coeff" in self._cfg:
            self.mel_loss_coeff = self._cfg.mel_loss_coeff

        self.audio_to_melspec_precessor = instantiate(self._cfg.preprocessor)
        self.encoder = Encoder(embed_input=False)

        # self.duration_predictor = VariancePredictor(d_model=256, d_inner=256, kernel_size=3, dropout=0.2)
        self.variance_adapter = VarianceAdaptor(pitch=self.pitch, energy=self.energy, vocab=self.vocab)

        self.generator = instantiate(self._cfg.generator)
        self.multiperioddisc = MultiPeriodDiscriminator()
        self.multiscaledisc = MultiScaleDiscriminator()

        self.mel_val_loss = L1MelLoss()
        self.durationloss = DurationLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.mseloss = torch.nn.MSELoss()

        self.use_energy_pred = False
        self.use_pitch_pred = False
        self.use_duration_pred = False
        self.log_train_images = False
        self.logged_real_samples = False
        self._tb_logger = None
        self.max = 0
        self.mel_basis = None
        self.hann_window = None
        self.splice_length = self._cfg.splice_length
        typecheck.set_typecheck_enabled(enabled=False)

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

    def configure_optimizers(self):
        gen_params = chain(
            self.phone_embedding.parameters(),
            self.encoder.parameters(),
            self.generator.parameters(),
            # self.duration_predictor.parameters(),
            self.variance_adapter.parameters(),
            # self.length_regulator.parameters(),
        )
        disc_params = chain(self.multiscaledisc.parameters(), self.multiperioddisc.parameters())
        opt1 = torch.optim.AdamW(disc_params, lr=self._cfg.lr)
        opt2 = torch.optim.AdamW(gen_params, lr=self._cfg.lr)
        num_procs = self._trainer.num_gpus * self._trainer.num_nodes
        num_samples = len(self._train_dl.dataset)
        batch_size = self._train_dl.batch_size
        iter_per_epoch = np.ceil(num_samples / (num_procs * batch_size))
        max_steps = iter_per_epoch * self._trainer.max_epochs
        logging.info(f"MAX STEPS: {max_steps}")
        sch1 = NoamAnnealing(opt1, d_model=256, warmup_steps=3000, max_steps=max_steps, min_lr=1e-5)
        sch1_dict = {
            'scheduler': sch1,
            'interval': 'step',
        }
        sch2 = NoamAnnealing(opt2, d_model=256, warmup_steps=3000, max_steps=max_steps, min_lr=1e-5)
        sch2_dict = {
            'scheduler': sch2,
            'interval': 'step',
        }
        return [opt1, opt2], [sch1_dict, sch2_dict]

    def forward(self, *, spec, spec_len, text, text_length, splice=True, durations=None, pitch=None, energies=None):
        embedded_tokens = self.phone_embedding(text)
        encoded_text, encoded_text_mask = self.encoder(text=embedded_tokens, text_lengths=text_length)

        # log_duration_prediction = None
        # log_duration_prediction = self.duration_predictor(encoded_text)
        # log_duration_prediction.masked_fill_(~get_mask_from_lengths(text_length).squeeze(), 0)
        # duration_rounded = torch.clamp_min(torch.exp(log_duration_prediction) - 1, 0).long()

        # if durations is None:
        #     # if splice:
        #     #     if torch.le(torch.sum(duration_rounded, dim=1), self.splice_length).bool().any():
        #     #         logging.error("Duration prediction failed on this batch. Increasing size")
        #     #         for duration in duration_rounded:
        #     #             if torch.sum(duration) < self.splice_length:
        #     #                 duration += torch.ceil(
        #     #                     (self.splice_length - torch.sum(duration)) / duration.size(0)
        #     #                 ).long()
        #     context = self.length_regulator(encoded_text, duration_rounded)
        #     spec_len = torch.sum(duration_rounded, dim=1)
        # else:
        #     context = self.length_regulator(encoded_text, durations)

        context, log_dur_preds, pitch_preds, energy_preds, spec_len = self.variance_adapter(
            x=encoded_text,
            x_len=text_length,
            dur_target=durations,
            pitch_target=pitch,
            energy_target=energies,
            spec_len=spec_len,
        )

        gen_in = context
        splices = None
        if splice:
            output = []
            splices = []
            for i, sample in enumerate(context):
                # start = torch.randint(low=0, high=int(sample.size(0) - self.splice_length), size=(1,))
                start = np.random.randint(low=0, high=min(int(sample.size(0)), int(spec_len[i])) - self.splice_length)
                # Splice generated spec
                output.append(sample[start : start + self.splice_length, :])
                splices.append(start)
            gen_in = torch.stack(output)

        output = self.generator(gen_in.transpose(1, 2))

        return output, splices, log_dur_preds, pitch_preds, energy_preds, encoded_text_mask

    def training_step(self, batch, batch_idx, optimizer_idx):
        if self.vocab is None:
            f, fl, t, tl, durations, pitch, energies = batch
        else:
            pitch, energies = None, None
            f, fl, t, tl, durations, _, _ = batch
        spec, spec_len = self.audio_to_melspec_precessor(f, fl)

        # train discriminator
        if optimizer_idx == 0:
            with torch.no_grad():
                audio_pred, splices, _, _, _, _ = self(
                    spec=spec,
                    spec_len=spec_len,
                    text=t,
                    text_length=tl,
                    durations=durations if not self.use_duration_pred else None,
                    pitch=pitch if not self.use_pitch_pred else None,
                    energies=energies if not self.use_energy_pred else None,
                )
                real_audio = []
                for i, splice in enumerate(splices):
                    real_audio.append(f[i, splice * 256 : (splice + self.splice_length) * 256])
                real_audio = torch.stack(real_audio).unsqueeze(1)

            real_score_mp, gen_score_mp, _, _ = self.multiperioddisc(real_audio, audio_pred)
            real_score_ms, gen_score_ms, _, _ = self.multiscaledisc(real_audio, audio_pred)

            loss_mp, loss_mp_real, _ = self.disc_loss(real_score_mp, gen_score_mp)
            loss_ms, loss_ms_real, _ = self.disc_loss(real_score_ms, gen_score_ms)
            loss_mp /= len(loss_mp_real)
            loss_ms /= len(loss_ms_real)
            loss_disc = loss_mp + loss_ms

            self.log("loss_discriminator", loss_disc, prog_bar=True)
            self.log("loss_discriminator_ms", loss_ms)
            self.log("loss_discriminator_mp", loss_mp)
            return loss_disc

        # train generator
        elif optimizer_idx == 1:
            audio_pred, splices, log_dur_preds, pitch_preds, energy_preds, encoded_text_mask = self(
                spec=spec,
                spec_len=spec_len,
                text=t,
                text_length=tl,
                durations=durations if not self.use_duration_pred else None,
                pitch=pitch if not self.use_pitch_pred else None,
                energies=energies if not self.use_energy_pred else None,
            )
            real_audio = []
            for i, splice in enumerate(splices):
                real_audio.append(f[i, splice * 256 : (splice + self.splice_length) * 256])
            real_audio = torch.stack(real_audio).unsqueeze(1)

            # Do HiFiGAN generator loss
            real_spliced_spec = self.mel_spectrogram(real_audio)
            pred_spliced_spec = self.mel_spectrogram(audio_pred)
            loss_mel = torch.nn.functional.l1_loss(real_spliced_spec, pred_spliced_spec)
            loss_mel *= self.mel_loss_coeff
            _, gen_score_mp, real_feat_mp, gen_feat_mp = self.multiperioddisc(real_audio, audio_pred)
            _, gen_score_ms, real_feat_ms, gen_feat_ms = self.multiscaledisc(real_audio, audio_pred)
            loss_gen_mp, list_loss_gen_mp = self.gen_loss(gen_score_mp)
            loss_gen_ms, list_loss_gen_ms = self.gen_loss(gen_score_ms)
            loss_gen_mp /= len(list_loss_gen_mp)
            loss_gen_ms /= len(list_loss_gen_ms)
            total_loss = loss_gen_mp + loss_gen_ms + loss_mel
            loss_feat_mp = self.feat_matching_loss(real_feat_mp, gen_feat_mp)
            loss_feat_ms = self.feat_matching_loss(real_feat_ms, gen_feat_ms)
            total_loss += loss_feat_mp + loss_feat_ms
            self.log(name="loss_gen_disc_feat", value=loss_feat_mp + loss_feat_ms)
            self.log(name="loss_gen_disc_feat_ms", value=loss_feat_ms)
            self.log(name="loss_gen_disc_feat_mp", value=loss_feat_mp)

            self.log(name="loss_gen_mel", value=loss_mel)
            self.log(name="loss_gen_disc", value=loss_gen_mp + loss_gen_ms)
            self.log(name="loss_gen_disc_mp", value=loss_gen_mp)
            self.log(name="loss_gen_disc_ms", value=loss_gen_ms)

            dur_loss = self.durationloss(log_dur_preds, durations.float())
            self.log(name="loss_gen_duration", value=dur_loss)
            total_loss += dur_loss
            if self.pitch:
                # pitch_loss = self.durationloss(log_pitch_preds, pitch.float())
                pitch_loss = self.mseloss(pitch_preds, pitch.float())
                total_loss += pitch_loss
                self.log(name="loss_gen_pitch", value=pitch_loss)
            if self.energy:
                energy_loss = self.mseloss(energy_preds, energies)
                total_loss += energy_loss
                self.log(name="loss_gen_energy", value=energy_loss)

            # Log images to tensorboard
            if self.log_train_images:
                self.log_train_images = False

                if self.logger is not None and self.logger.experiment is not None:
                    self.tb_logger.add_image(
                        "train_mel_target",
                        plot_spectrogram_to_numpy(real_spliced_spec[0].data.cpu().numpy()),
                        self.global_step,
                        dataformats="HWC",
                    )
                    spec_predict = pred_spliced_spec[0].data.cpu().numpy()
                    self.tb_logger.add_image(
                        "train_mel_predicted",
                        plot_spectrogram_to_numpy(spec_predict),
                        self.global_step,
                        dataformats="HWC",
                    )
            self.log(name="loss_gen", prog_bar=True, value=total_loss)
            return total_loss

    def validation_step(self, batch, batch_idx):
        if self.vocab is None:
            f, fl, t, tl, _ = batch
        else:
            f, fl, t, tl, _, _, _ = batch
        spec, spec_len = self.audio_to_melspec_precessor(f, fl)
        audio_pred, _, _, _, _, _ = self(spec=spec, spec_len=spec_len, text=t, text_length=tl, splice=False)
        pred_spec = self.mel_spectrogram(audio_pred)
        loss = self.mel_val_loss(
            spec_pred=pred_spec, spec_target=spec, spec_target_len=spec_len, pad_value=-11.52, transpose=False
        )

        return {
            "val_loss": loss,
            "audio_target": f.squeeze(),
            "audio_pred": audio_pred.squeeze(),
        }

    def on_train_epoch_start(self):
        if self.vocab is None:
            # Switch to using energy predictions after 50% of training
            if not self.use_energy_pred and self.current_epoch >= np.ceil(0.5 * self._trainer.max_epochs):
                logging.info(f"Using energy predictions after epoch: {self.current_epoch}")
                self.use_energy_pred = True

            # Switch to using pitch predictions after 62.5% of training
            if not self.use_pitch_pred and self.current_epoch >= np.ceil(0.625 * self._trainer.max_epochs):
                logging.info(f"Using pitch predictions after epoch: {self.current_epoch}")
                self.use_pitch_pred = True

            # # Switch to using duration predictions after 75% of training
            # if not self.use_duration_pred and self.current_epoch >= np.ceil(0.75 * self._trainer.max_epochs):
            #     logging.info(f"Using duration predictions after epoch: {self.current_epoch}")
            #     self.use_duration_pred = True

    def validation_epoch_end(self, outputs):
        if self.tb_logger is not None:
            _, audio_target, audio_predict = outputs[0].values()
            if not self.logged_real_samples:
                self.tb_logger.add_audio("val_target", audio_target[0].data.cpu(), self.global_step, 22050)
                self.logged_real_samples = True
            audio_predict = audio_predict[0].data.cpu()
            self.tb_logger.add_audio("val_pred", audio_predict, self.global_step, 22050)
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()  # This reduces across batches, not workers!
        self.log('val_loss', avg_loss, sync_dist=True)

        self.log_train_images = True

    def __setup_dataloader_from_config(self, cfg, shuffle_should_be: bool = True, name: str = "train"):
        if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
            raise ValueError(f"No dataset for {name}")
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloder_params for {name}")
        if shuffle_should_be:
            if 'shuffle' not in cfg.dataloader_params:
                logging.warning(
                    f"Shuffle should be set to True for {self}'s {name} dataloader but was not found in its "
                    "config. Manually setting to True"
                )
                with open_dict(cfg.dataloader_params):
                    cfg.dataloader_params.shuffle = True
            elif not cfg.dataloader_params.shuffle:
                logging.error(f"The {name} dataloader for {self} has shuffle set to False!!!")
        elif not shuffle_should_be and cfg.dataloader_params.shuffle:
            logging.error(f"The {name} dataloader for {self} has shuffle set to True!!!")

        dataset = instantiate(cfg.dataset)
        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def setup_training_data(self, cfg):
        self._train_dl = self.__setup_dataloader_from_config(cfg)

    def list_available_models(self):
        pass

    def setup_validation_data(self, cfg):
        pass
        self._validation_dl = self.__setup_dataloader_from_config(cfg, shuffle_should_be=False, name="validation")

    def mel_spectrogram(
        self,
        y,
        n_fft=1024,
        num_mels=80,
        sampling_rate=22050,
        hop_size=256,
        win_size=1024,
        fmin=0.0,
        fmax=None,
        center=True,
    ):
        if self.mel_basis is None:
            mel = librosa.filters.mel(sampling_rate, n_fft, num_mels, fmin, fmax)
            self.mel_basis = torch.from_numpy(mel).float().to(y.device)
            self.hann_window = torch.hann_window(win_size).to(y.device)

        spec = torch.stft(
            y.squeeze(), n_fft, hop_length=hop_size, win_length=win_size, window=self.hann_window, center=center,
        )

        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
        spec = torch.matmul(self.mel_basis, spec)
        spec = torch.log(torch.clamp(spec, min=1e-5))

        return spec
