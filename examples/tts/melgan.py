from pathlib import Path

import librosa
import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from melgan_models import Discriminator, Generator, MultiResolutionSTFTLoss
from omegaconf import DictConfig, OmegaConf, open_dict
from pqmf import PQMF
from pwg_melgan_models import MelGANGenerator, MelGANMultiScaleDiscriminator, MultiRWDDiscriminator
from pytorch_lightning.utilities import rank_zero_only

from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.helpers.helpers import plot_spectrogram_to_numpy
from nemo.core.classes import ModelPT
from nemo.core.config import hydra_runner
from nemo.core.optim.lr_scheduler import CosineAnnealing
from nemo.utils import logging
from nemo.utils.decorators import experimental
from nemo.utils.exp_manager import exp_manager
from nemo.collections.tts.helpers.helpers import get_mask_from_lengths


@experimental
class MBMelGanModel(ModelPT):
    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        super().__init__(cfg=cfg, trainer=trainer)

        self.audio_to_melspec_precessor = instantiate(self._cfg.preprocessor)
        self.generator = MelGANGenerator(**cfg.gen)
        if "input_lengths" in cfg.disc:
            self.discriminator = MultiRWDDiscriminator(**cfg.disc)
        else:
            self.discriminator = MelGANMultiScaleDiscriminator(**cfg.disc)
        self.pqmf = None
        self.subband_loss = None
        if cfg.pqmf:
            self.pqmf = PQMF()
            self.subband_loss = MultiResolutionSTFTLoss(
                fft_sizes=[384, 683, 171], hop_sizes=[30, 60, 10], win_lengths=[150, 300, 60]
            )
        loss_cfg = {}
        if "loss" in cfg:
            loss_cfg = cfg.loss
        self.loss = MultiResolutionSTFTLoss(**loss_cfg)
        self.train_disc = False
        self.mse_loss = torch.nn.MSELoss()
        self.adv_coeff = self._cfg.init_adv_lambda
        self.increase_coeff = self._cfg.increase_lambda
        self.add_gan_loss = self._cfg.get("add_gan_loss", True)

    @property
    def input_types(self):
        pass

    @property
    def output_types(self):
        pass

    def configure_optimizers(self):
        opt1 = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3, eps=1e-07, amsgrad=True)
        opt2 = torch.optim.Adam(self.generator.parameters(), lr=1e-3, eps=1e-07, amsgrad=True)
        num_procs = self._trainer.num_gpus * self._trainer.num_nodes
        num_samples = len(self._train_dl.dataset)
        batch_size = self._train_dl.batch_size
        iter_per_epoch = np.ceil(num_samples / (num_procs * batch_size))
        max_steps = iter_per_epoch * self._trainer.max_epochs
        logging.info(f"MAX STEPS: {max_steps}")
        # sch1 = torch.optim.lr_scheduler.MultiStepLR(opt1, milestones=[400, 800, 1200, 1600, 2000, 2400], gamma=0.5)
        # sch2 = torch.optim.lr_scheduler.MultiStepLR(opt2, milestones=[400, 800, 1200, 1600, 2000, 2400], gamma=0.5)
        sch1 = CosineAnnealing(
            opt1, max_steps=max_steps, min_lr=1e-5, warmup_steps=np.ceil(0.2 * max_steps)
        )  # Use warmup to delay start
        sch1_dict = {
            'scheduler': sch1,
            'interval': 'step',
        }
        sch2 = CosineAnnealing(opt2, max_steps=max_steps, min_lr=1e-5)
        sch2_dict = {
            'scheduler': sch2,
            'interval': 'step',
        }
        return [opt1, opt2], [sch1_dict, sch2_dict]

    def forward(self):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        # TODO, manual logging to tensorboard because lightning isn't working for some reason
        audio, audio_len = batch
        spec, _ = self.audio_to_melspec_precessor(audio, audio_len)
        mb_audio_pred = self.generator(spec)
        log_on_step = self.global_step % 1000 == 0 or self.global_step % 1000 == 1

        # train discriminator
        if optimizer_idx == 0 and self.train_disc:
            audio_pred = mb_audio_pred
            if self.pqmf is not None:
                audio_pred = self.pqmf.synthesis(mb_audio_pred)
            fake_score = self.discriminator(audio_pred.detach())
            real_score = self.discriminator(audio.unsqueeze(1))

            # loss_disc = 0
            # for scale in fake_score:
            #     loss_disc += F.relu(1 + scale[-1]).mean()

            # for scale in real_score:
            #     loss_disc += F.relu(1 - scale[-1]).mean()

            loss_disc_real = [0] * len(fake_score)
            loss_disc_fake = [0] * len(fake_score)
            for i in range(len(fake_score)):
                loss_disc_real[i] += self.mse_loss(
                    real_score[i][-1], real_score[i][-1].new_ones(real_score[i][-1].size())
                )
                loss_disc_fake[i] += torch.mean(fake_score[i][-1] ** 2)
            sum_loss_dis = sum(loss_disc_real) + sum(loss_disc_fake)

            self.log(f"loss_discriminator", sum_loss_dis, prog_bar=True)
            for i in range(len(fake_score)):
                self.log(f"loss_discriminator_real_{i}", loss_disc_real[i])
                self.log(f"loss_discriminator_fake_{i}", loss_disc_fake[i])
            return {'loss': sum_loss_dis}
        # train generator
        elif optimizer_idx == 1:
            loss = 0
            audio_pred = mb_audio_pred
            if self.pqmf is not None:
                audio_pred = self.pqmf.synthesis(mb_audio_pred)

            # full-band loss
            sc_loss, mag_loss = self.loss(audio_pred.squeeze(1), audio)
            loss_feat = sum(sc_loss) + sum(mag_loss)

            # MB loss
            if self.pqmf is not None:
                loss_feat *= 0.5
                audio_mb = self.pqmf.analysis(audio.unsqueeze(1))
                mb_audio_pred = mb_audio_pred.view(-1, mb_audio_pred.size(2))
                audio_mb = audio_mb.view(-1, audio_mb.size(2))
                sub_sc_loss, sub_mag_loss = self.subband_loss(mb_audio_pred, audio_mb)
                loss_feat += 0.5 * (sum(sub_sc_loss) + sum(sub_mag_loss))

            loss += loss_feat

            if self.train_disc:
                fake_score = self.discriminator(audio_pred)
                # real_score = self.discriminator(audio.unsqueeze(1))

                # loss_gen = 0
                # for scale in fake_score:
                #     loss_gen += -scale[-1].mean()

                loss_gan = [0] * len(fake_score)
                for i, scale in enumerate(fake_score):
                    loss_gan[i] += self.mse_loss(scale[-1], scale[-1].new_ones(scale[-1].size()))

                sum_loss_gan = self.adv_coeff * sum(loss_gan) / len(fake_score)

                loss += sum_loss_gan

            self.log("loss_generator", loss, prog_bar=True)
            if self.train_disc:
                self.log(f"loss_generator_gan_loss", sum_loss_gan)
                for i in range(len(fake_score)):
                    self.log(
                        f"loss_generator_gan_loss_{i}", self.adv_coeff * loss_gan[i] / len(fake_score),
                    )
            self.log("loss_generator_feat_loss", loss_feat)
            factor = 0.5 if self.pqmf else 1.0
            self.log("loss_generator_feat_loss_fb_sc", sum(sc_loss) * factor)
            self.log("loss_generator_feat_loss_fb_mag", sum(mag_loss) * factor)
            for i in range(len(sc_loss)):
                self.log(f"loss_generator_feat_loss_fb_sc_{i}", sc_loss[i] * factor)
                self.log(f"loss_generator_feat_loss_fb_mag_{i}", mag_loss[i] * factor)
            if self.pqmf is not None:
                self.log("loss_generator_mb_sc", sum(sub_sc_loss) * factor)
                self.log("loss_generator_mb_mag", sum(sub_mag_loss) * factor)
                for i in range(len(sc_loss)):
                    self.log(f"loss_generator_feat_loss_mb_sc_{i}", sc_loss[i] * factor)
                    self.log(
                        f"loss_generator_feat_loss_mb_mag_{i}", mag_loss[i] * factor,
                    )
            # self.logger.experiment.add_scalar("loss_generator", loss, self.global_step)
            # if self.train_disc:
            #     self.logger.experiment.add_scalar(f"loss_generator_gan_loss", sum_loss_gan, self.global_step)
            #     for i in len(fake_score):
            #         self.logger.experiment.add_scalar(
            #             f"loss_generator_gan_loss_{i}",
            #             self.adv_coeff * loss_gan[i] / len(fake_score),
            #             self.global_step,
            #         )
            # self.logger.experiment.add_scalar("loss_generator_feat_loss", loss_feat, self.global_step)
            # factor = 0.5 if self.pqmf else 1.0
            # self.logger.experiment.add_scalar(
            #     "loss_generator_feat_loss_fb_sc", sum(sc_loss) * factor, self.global_step
            # )
            # self.logger.experiment.add_scalar(
            #     "loss_generator_feat_loss_fb_mag", sum(mag_loss) * factor, self.global_step
            # )
            # for i in len(sc_loss):
            #     self.logger.experiment.add_scalar(
            #         f"loss_generator_feat_loss_fb_sc_{i}", sc_loss[i] * factor, self.global_step
            #     )
            #     self.logger.experiment.add_scalar(
            #         f"loss_generator_feat_loss_fb_mag_{i}", mag_loss[i] * factor, self.global_step
            #     )
            # if self.pqmf is not None:
            #     self.logger.experiment.add_scalar(
            #         "loss_generator_mb_sc", sum(sub_sc_loss) * factor, self.global_step
            #     )
            #     self.logger.experiment.add_scalar(
            #         "loss_generator_mb_mag", sum(sub_mag_loss) * factor, self.global_step
            #     )
            #     for i in len(sc_loss):
            #         self.logger.experiment.add_scalar(
            #             f"loss_generator_feat_loss_mb_sc_{i}", sc_loss[i] * factor, self.global_step
            #         )
            #         self.logger.experiment.add_scalar(
            #             f"loss_generator_feat_loss_mb_mag_{i}", mag_loss[i] * factor, self.global_step
            #         )

            return {'loss': loss}
        return None

    def validation_step(self, batch, batch_idx):
        audio, audio_len = batch
        with torch.no_grad():
            spec, _ = self.audio_to_melspec_precessor(audio, audio_len)
            mb_audio_pred = self.generator(spec)

            loss = 0
            loss_dict = {}
            audio_pred = mb_audio_pred
            if self.pqmf is not None:
                audio_pred = self.pqmf.synthesis(mb_audio_pred)
            spec_pred, _ = self.audio_to_melspec_precessor(audio_pred.squeeze(1), audio_len)

            # Ensure that audio len is consistent between audio_pred and audio
            # For SC Norm loss, we can just zero out
            # For Mag L1 loss, we need to mask
            if audio_pred.shape[-1] < audio.shape[-1]:
                # prediction audio is less than audio, pad predicted audio to real audio
                pad_amount = audio.shape[-1] - audio_pred.shape[-1]
                audio_pred = torch.nn.functional.pad(audio_pred, (0, pad_amount), value=0.0)
            else:
                # prediction audio is larger than audio, slice predicted audio to real audio
                audio_pred = audio_pred[:, :, : audio.shape[1]]

            mask = ~get_mask_from_lengths(audio_len, max_len=torch.max(audio_len))
            mask = mask.unsqueeze(1)
            audio_pred.data.masked_fill_(mask, 0.0)

            # full-band loss
            sc_loss, mag_loss = self.loss(audio_pred.squeeze(1), audio, audio_len)
            loss_feat = sum(sc_loss) + sum(mag_loss)
            loss_dict["sc_loss"] = sc_loss
            loss_dict["mag_loss"] = mag_loss

            # MB loss
            if self.pqmf is not None:
                loss_feat *= 0.5
                audio_mb = self.pqmf.analysis(audio.unsqueeze(1))
                mb_audio_pred = mb_audio_pred.view(-1, mb_audio_pred.size(2))
                audio_mb = audio_mb.view(-1, audio_mb.size(2))
                # TODO: This needs to be debugged
                sub_sc_loss, sub_mag_loss = self.subband_loss(mb_audio_pred, audio_mb, audio_len)
                loss_feat += 0.5 * (sum(sub_sc_loss) + sum(sub_mag_loss))
                loss_dict["sub_sc_loss"] = sub_sc_loss
                loss_dict["sub_mag_loss"] = sub_mag_loss

            loss += loss_feat
            loss_dict["loss_feat"] = loss_feat

            if self.train_disc:
                fake_score = self.discriminator(audio_pred)

                loss_gen = [0 * len(fake_score)]
                for i, scale in enumerate(fake_score):
                    loss_gen[i] += self.mse_loss(scale[-1], scale[-1].new_ones(scale[-1].size()))

                loss_dict["gan_loss"] = loss_gen
                loss += self.adv_coeff * loss_gen / len(fake_score)

        loss_dict["spec"] = spec
        loss_dict["spec_pred"] = spec_pred
        loss_dict["loss"] = loss
        return loss_dict

    def validation_epoch_end(self, outputs):
        if self.logger is not None and self.logger.experiment is not None:
            self.logger.experiment.add_image(
                "val_mel_target",
                plot_spectrogram_to_numpy(outputs[0]["spec"][0].data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            self.logger.experiment.add_image(
                "val_mel_predicted",
                plot_spectrogram_to_numpy(outputs[0]["spec_pred"][0].data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )

        def get_stack(list_of_dict, key):
            return_list = [[]] * len(list_of_dict[0][key])
            for y in list_of_dict:
                list_of_losses = y[key]
                for i, loss in enumerate(list_of_losses):
                    return_list[i].append(loss)
            for i, loss in enumerate(return_list):
                return_list[i] = torch.mean(torch.stack(loss))
            return return_list

        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("val_loss", loss, sync_dist=True)

        if self.train_disc:
            gan_loss = get_stack(outputs, "gan_loss")
            self.log(f"val_loss_gan_loss", self.adv_coeff / len(gan_loss) * sum(gan_loss), sync_dist=True)
            for i in len(gan_loss):
                self.log(
                    f"val_loss_gan_loss_{i}", self.adv_coeff / len(gan_loss) * gan_loss[i], sync_dist=True,
                )

        loss_feat = 0
        factor = 1
        if self.pqmf is not None:
            sub_sc_loss = get_stack(outputs, "sub_sc_loss")
            sub_mag_loss = get_stack(outputs, "sub_mag_loss")
            loss_feat += sum(sub_mag_loss) + sum(sub_sc_loss)
            factor = 0.5
        loss_feat *= factor
        sc_loss = get_stack(outputs, "sc_loss")
        mag_loss = get_stack(outputs, "mag_loss")
        loss_feat += (sum(mag_loss) + sum(sc_loss)) * factor
        self.log("val_loss_feat_loss", loss_feat, sync_dist=True)
        self.log("val_loss_feat_loss_fb_sc", sum(sc_loss) * factor, sync_dist=True)
        self.log("val_loss_feat_loss_fb_mag", sum(mag_loss) * factor, sync_dist=True)
        for i in range(len(sc_loss)):
            self.log(f"val_loss_feat_loss_fb_sc_{i}", sc_loss[i] * factor, sync_dist=True)
            self.log(f"val_loss_feat_loss_fb_mag_{i}", mag_loss[i] * factor, sync_dist=True)
        if self.pqmf is not None:
            self.log("val_loss_mb_sc", sum(sub_sc_loss) * factor, sync_dist=True)
            self.log("val_loss_mb_mag", sum(sub_mag_loss) * factor, sync_dist=True)
            for i in range(len(sub_sc_loss)):
                self.log(f"val_loss_feat_loss_mb_sc_{i}", sc_loss[i] * factor, sync_dist=True)
                self.log(f"val_loss_feat_loss_mb_mag_{i}", mag_loss[i] * factor, sync_dist=True)

    def __setup_dataloader_from_config(self, cfg, shuffle_should_be: bool = True, name: str = "train"):
        if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
            raise ValueError(f"No dataset for {name}")  # TODO
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloder_params for {name}")  # TODO
        if shuffle_should_be:
            if 'shuffle' not in cfg.dataloader_params:
                logging.warning(
                    f"Shuffle should be set to True for {self}'s {name} dataloader but was not found in its "
                    "config. Manually setting to True"
                )
                with open_dict(cfg["dataloader_params"]):
                    cfg.dataloader_params.shuffle = True
            elif not cfg.dataloader_params.shuffle:
                logging.error(f"The {name} dataloader for {self} has shuffle set to False!!!")
        elif not shuffle_should_be and cfg.dataloader_params.shuffle:
            logging.error(f"The {name} dataloader for {self} has shuffle set to True!!!")

        dataset = instantiate(cfg.dataset)
        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def setup_training_data(self, cfg):
        self._train_dl = self.__setup_dataloader_from_config(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self.__setup_dataloader_from_config(cfg, shuffle_should_be=False, name="validation")

    def training_epoch_end(self, outputs):
        if self.current_epoch % 10 == 0:
            lrs = []
            for scheduler in self._trainer.lr_schedulers:
                param_groups = scheduler['scheduler'].optimizer.param_groups
                lrs.append(param_groups[0]['lr'])
            self.logger.experiment.add_scalar("lr-Adam", lrs[0], self.global_step)
            self.logger.experiment.add_scalar("lr-Adam-1", lrs[1], self.global_step)
            self.logger.experiment.add_scalar("epoch", self.current_epoch, self.global_step)

        if self.current_epoch >= np.ceil(0.2 * self._trainer.max_epochs) and self.add_gan_loss:
            self.train_disc = True

        # Add staircase increase for adv_coeff
        if self.increase_coeff:
            if self.current_epoch >= np.ceil(11.0 / 15.0 * self._trainer.max_epochs):
                self.adv_coeff = 10
            elif self.current_epoch >= np.ceil(9.0 / 15.0 * self._trainer.max_epochs):
                self.adv_coeff = 7.5
            elif self.current_epoch >= np.ceil(7.0 / 15.0 * self._trainer.max_epochs):
                self.adv_coeff = 5
            elif self.current_epoch >= np.ceil(5.0 / 15.0 * self._trainer.max_epochs):
                self.adv_coeff = 2.5
        return super().training_epoch_end(outputs)

    @classmethod
    def list_available_models(cls) -> 'Optional[Dict[str, str]]':
        pass


def train_pwg(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = MBMelGanModel(cfg=cfg.model, trainer=trainer)
    # lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    # trainer.callbacks.extend([lr_logger, epoch_time_logger])
    trainer.callbacks.extend([epoch_time_logger])
    trainer.fit(model)


def infer_pwg(cfg):
    model = MBMelGanModel.load_from_checkpoint(
        "/home/jasoli/nemo/NeMo/examples/tts/experiments/1501438_MelGAN_MSE_BS64_E3000/MB_MelGan--last.ckpt"
    )
    # "/home/jasoli/nemo/NeMo/examples/tts/nemo_experiments/MB_MelGan/2020-09-24_11-29-43/checkpoints/MB_MelGan--last.ckpt"
    model.setup_validation_data(cfg.model.validation_ds)
    model.cuda()
    model.eval()
    audio, audio_len = next(iter(model._validation_dl))
    audio = audio.to("cuda")
    audio_len = audio_len.to("cuda")
    with torch.no_grad():
        spec, _ = model.audio_to_melspec_precessor(audio, audio_len)
        mb_audio_pred = model.generator(spec[0].unsqueeze(0))
        audio_pred = model.pqmf.synthesis(mb_audio_pred)
    spec_out, _ = model.audio_to_melspec_precessor(audio_pred.squeeze(0), audio_len[0].unsqueeze(0))
    librosa.output.write_wav("MB_MelGAN_0.wav", audio_pred[0].cpu().numpy().squeeze(), sr=cfg.sample_rate)

    from matplotlib import pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.imshow(spec[0].cpu().numpy().squeeze(), origin="lower")
    ax2.imshow(spec_out.cpu().numpy().squeeze(), origin="lower")
    plt.savefig("MB_MelGAN_0")
    print(audio_pred[0])
    print(audio[0])
    print(spec[0].cpu().numpy().shape)
    print(spec_out.cpu().numpy().shape)


def infer_pwg_batch(cfg):
    checkpoint = list(Path(cfg.checkpoint_dir).glob("*nemo"))[0]
    # model = MBMelGanModel.load_from_checkpoint(str(checkpoint))
    model = MBMelGanModel.restore_from(str(checkpoint))
    model.setup_validation_data(cfg.model.validation_ds)
    model.cuda()
    model.eval()
    for batch in model._validation_dl:
        audio, audio_len = batch
        audio = audio.to("cuda")
        audio_len = audio_len.to("cuda")
        with torch.no_grad():
            spec, _ = model.audio_to_melspec_precessor(audio, audio_len)
            print(spec.shape)
            mb_audio_pred = model.generator(spec)
            audio_pred = mb_audio_pred
            if model.pqmf is not None:
                audio_pred = model.pqmf.synthesis(mb_audio_pred)
        for i, single_audio in enumerate(audio_pred):
            print(single_audio.cpu().numpy().squeeze())
            sf.write(
                f"{cfg.name}_{i}.wav", single_audio.cpu().numpy().squeeze()[: audio_len[i]], samplerate=cfg.sample_rate
            )
        # for i, single_audio in enumerate(audio):
        #     print(single_audio.cpu().numpy().squeeze())
        #     sf.write(
        #         f"Sally_real_{i}.wav", single_audio.cpu().numpy().squeeze()[: audio_len[i]], samplerate=cfg.sample_rate
        #     )
        break


def infer_pwg_batch_2(cfg):
    checkpoint = list(Path(cfg.checkpoint_dir).glob("*nemo"))[0]
    model = MBMelGanModel.restore_from(str(checkpoint))
    model.cuda()
    model.eval()
    for i, mel in enumerate(Path("/home/jasoli/waveglow/mel_spectrograms/mel_spectrograms").glob("*.pt")):
        with torch.no_grad():
            spec = torch.load(mel)
            spec = torch.unsqueeze(spec, 0).cuda()
            print(spec.shape)
            mb_audio_pred = model.generator(spec)
            audio_pred = mb_audio_pred
            if model.pqmf is not None:
                audio_pred = model.pqmf.synthesis(mb_audio_pred)
            print(audio_pred.cpu().numpy().squeeze())
            sf.write(f"{cfg.name}_{i}.wav", audio_pred.cpu().numpy().squeeze(), samplerate=cfg.sample_rate)


# @experimental
# class MelGanModel(ModelPT):
#     def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
#         if isinstance(cfg, dict):
#             cfg = OmegaConf.create(cfg)
#         super().__init__(cfg=cfg, trainer=trainer)

#         self.audio_to_melspec_precessor = instantiate(self._cfg.preprocessor)
#         self.generator = Generator(**cfg.gen)
#         self.discriminator = Discriminator(**cfg.disc)
#         self.loss = MultiResolutionSTFTLoss()
#         # self.discriminator = None
#         # if cfg.training:
#         #     self.discriminator = Discriminator(**cfg.disc)

#     @property
#     def input_types(self):
#         pass

#     @property
#     def output_types(self):
#         pass

#     def configure_optimizers(self):
#         opt1 = torch.optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
#         opt2 = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))
#         return [opt1, opt2]

#     def forward(self):
#         pass

#     def training_step(self, batch, batch_idx, optimizer_idx):
#         # TODO, manual logging to tensorboard because lightning isn't working for some reason
#         audio, audio_len = batch
#         spec, _ = self.audio_to_melspec_precessor(audio, audio_len)
#         audio_pred = self.generator(spec)
#         audio = audio.unsqueeze(1)

#         # with torch.no_grad():
#         #     spec_pred, _ = self.audio_to_melspec_precessor(audio_pred.detach(), audio_len)
#         #     spec_error = F.l1_loss(spec, spec_pred).item()

#         # train generator
#         if optimizer_idx == 0:
#             fake_score = self.discriminator(audio_pred)
#             real_score = self.discriminator(audio)

#             loss_gen = 0
#             for scale in fake_score:
#                 loss_gen += -scale[-1].mean()

#             sc_loss, mag_loss = self.loss(audio_pred.squeeze(), audio.squeeze())
#             loss_feat = sc_loss + mag_loss

#             # loss_feat = 0
#             # feat_weights = 4.0 / (self._cfg.disc.n_layers + 1)
#             # D_weights = 1.0 / self._cfg.disc.num_D
#             # wt = D_weights * feat_weights
#             # for i in range(self._cfg.disc.num_D):
#             #     for j in range(len(fake_score[i]) - 1):
#             #         loss_feat += wt * F.l1_loss(fake_score[i][j], real_score[i][j].detach())

#             loss = loss_gen + self._cfg.lambda_feat * loss_feat
#             # result = pl.TrainResult(loss)
#             # result.log("loss_generator", loss, prog_bar=True)
#             return {
#                 'loss': loss,
#                 'progress_bar': {'loss_gen': loss},
#                 'log': {'loss_generator': loss},
#             }
#         # train discriminator
#         if optimizer_idx == 1:
#             fake_score = self.discriminator(audio_pred.detach())
#             real_score = self.discriminator(audio)

#             loss_disc = 0
#             for scale in fake_score:
#                 loss_disc += F.relu(1 + scale[-1]).mean()

#             for scale in real_score:
#                 loss_disc += F.relu(1 - scale[-1]).mean()
#             # result = pl.TrainResult(loss_disc)
#             # result.log("loss_discriminator", loss_disc, prog_bar=True)
#             return {
#                 'loss': loss_disc,
#                 'progress_bar': {'loss_disc': loss_disc},
#                 'log': {'loss_discriminator': loss_disc},
#             }

#     def validation_step(self, batch, batch_idx):
#         audio, audio_len = batch
#         spec, _ = self.audio_to_melspec_precessor(audio, audio_len)
#         audio_pred = self.generator(spec)

#         audio = audio.unsqueeze(1)
#         real_audio_len = audio.shape[-1]
#         fake_audio_len = audio_pred.shape[-1]
#         if fake_audio_len > real_audio_len:
#             # pad real
#             diff = fake_audio_len - real_audio_len
#             audio = torch.nn.functional.pad(audio, (0, diff), mode="constant", value=0)
#         else:
#             # pad fake
#             diff = real_audio_len - fake_audio_len
#             audio_pred = torch.nn.functional.pad(audio_pred, (0, diff), mode="constant", value=0)
#         fake_score = self.discriminator(audio_pred)
#         real_score = self.discriminator(audio)

#         loss_gen = 0
#         for scale in fake_score:
#             loss_gen += -scale[-1].mean()

#         loss_feat = 0
#         feat_weights = 4.0 / (self._cfg.disc.n_layers + 1)
#         D_weights = 1.0 / self._cfg.disc.num_D
#         wt = D_weights * feat_weights
#         for i in range(self._cfg.disc.num_D):
#             for j in range(len(fake_score[i]) - 1):
#                 loss_feat += wt * F.l1_loss(fake_score[i][j], real_score[i][j].detach())

#         loss = loss_gen + self._cfg.lambda_feat * loss_feat

#         # result = pl.EvalResult(checkpoint_on=loss)
#         # result.loss = loss

#         # if batch_idx == 0:
#         #     result.spec = spec
#         #     result.spec_pred = spec_pred

#         # return result
#         spec_pred, _ = self.audio_to_melspec_precessor(audio_pred.squeeze(), audio_len)
#         return {
#             "loss": loss,
#             "spec": spec,
#             "spec_pred": spec_pred,
#         }

#     def validation_epoch_end(self, outputs):
#         if self.logger is not None and self.logger.experiment is not None:
#             self.logger.experiment.add_image(
#                 "val_mel_target",
#                 plot_spectrogram_to_numpy(outputs[0]["spec"][0].data.cpu().numpy()),
#                 self.global_step,
#                 dataformats="HWC",
#             )
#             self.logger.experiment.add_image(
#                 "val_mel_predicted",
#                 plot_spectrogram_to_numpy(outputs[0]["spec_pred"][0].data.cpu().numpy()),
#                 self.global_step,
#                 dataformats="HWC",
#             )
#         loss = torch.stack([x['loss'] for x in outputs]).mean()
#         tensorboard_logs = {'loss_val': loss}
#         return {'loss_val': loss, 'log': tensorboard_logs}

#     def __setup_dataloader_from_config(self, cfg, shuffle_should_be: bool = True, name: str = "train"):
#         if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
#             raise ValueError(f"No dataset for {name}")  # TODO
#         if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
#             raise ValueError(f"No dataloder_params for {name}")  # TODO
#         if shuffle_should_be:
#             if 'shuffle' not in cfg.dataloader_params:
#                 logging.warning(
#                     f"Shuffle should be set to True for {self}'s {name} dataloader but was not found in its "
#                     "config. Manually setting to True"
#                 )
#                 with open_dict(cfg["dataloader_params"]):
#                     cfg.dataloader_params.shuffle = True
#             elif not cfg.dataloader_params.shuffle:
#                 logging.error(f"The {name} dataloader for {self} has shuffle set to False!!!")
#         elif not shuffle_should_be and cfg.dataloader_params.shuffle:
#             logging.error(f"The {name} dataloader for {self} has shuffle set to True!!!")

#         dataset = instantiate(cfg.dataset)
#         return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

#     def setup_training_data(self, cfg):
#         self._train_dl = self.__setup_dataloader_from_config(cfg)

#     def setup_validation_data(self, cfg):
#         self._validation_dl = self.__setup_dataloader_from_config(cfg, shuffle_should_be=False, name="validation")

#     @classmethod
#     def list_available_models(cls) -> 'Optional[Dict[str, str]]':
#         pass

# from matplotlib import pyplot as plt

# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.imshow(spec[0].cpu().numpy().squeeze(), origin="lower")
# ax2.imshow(spec_out.cpu().numpy().squeeze(), origin="lower")
# plt.savefig("MB_MelGAN_0")
# print(audio_pred[0])
# print(audio[0])
# print(spec[0].cpu().numpy().shape)
# print(spec_out.cpu().numpy().shape)


# @hydra_runner(config_path="conf", config_name="melgan")
# def infer(cfg):
#     # model = MelGanModel(cfg=cfg.model)
#     # FB_MelGAN
#     model = MelGanModel.load_from_checkpoint(
#         "/home/jasoli/nemo/NeMo/examples/tts/nemo_experiments/FB_MelGan/2020-09-15_14-26-54/checkpoints/MelGan--last.ckpt"
#     )

#     # MelGAN
#     # model = MelGanModel.load_from_checkpoint(
#     #     "/home/jasoli/nemo/NeMo/examples/tts/nemo_experiments/MelGan/2020-08-31_14-17-42/checkpoints/MelGan--last.ckpt"
#     # )
#     model.setup_validation_data(cfg.model.validation_ds)
#     model.cuda()
#     audio, audio_len = next(iter(model._validation_dl))
#     audio = audio.to("cuda")
#     audio_len = audio_len.to("cuda")
#     with torch.no_grad():
#         spec, _ = model.audio_to_melspec_precessor(audio, audio_len)
#         audio_pred = model.generator(spec[0].unsqueeze(0))
#     spec_out, _ = model.audio_to_melspec_precessor(audio_pred.squeeze(0), audio_len[0].unsqueeze(0))
#     librosa.output.write_wav("FB_MelGAN_0.wav", audio_pred[0].cpu().numpy().squeeze(), sr=22050)
#     # librosa.output.write_wav("MelGAN_0.wav", audio_pred[0].cpu().numpy().squeeze(), sr=22050)

#     from matplotlib import pyplot as plt

#     fig, (ax1, ax2) = plt.subplots(2, 1)
#     ax1.imshow(spec[0].cpu().numpy().squeeze(), origin="lower")
#     ax2.imshow(spec_out.cpu().numpy().squeeze(), origin="lower")
#     plt.savefig("FB_MelGAN_0")
#     # plt.savefig("MelGAN_0")
#     print(audio_pred[0])
#     print(audio[0])
#     print(spec[0].cpu().numpy().shape)
#     print(spec_out.cpu().numpy().shape)

# @hydra_runner(config_path="conf", config_name="melgan")
# def main(cfg):
#     trainer = pl.Trainer(**cfg.trainer)
#     exp_manager(trainer, cfg.get("exp_manager", None))
#     model = MelGanModel(cfg=cfg.model, trainer=trainer)
#     lr_logger = pl.callbacks.LearningRateLogger()
#     epoch_time_logger = LogEpochTimeCallback()
#     trainer.callbacks.extend([lr_logger, epoch_time_logger])
#     trainer.fit(model)


@hydra_runner(config_path="conf", config_name="multiband_melgan")
def main(cfg):
    if "infer" in cfg:
        infer_pwg_batch(cfg)
    else:
        train_pwg(cfg)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter

"""
Currently at 25 steps per epoch
Runs pretraining for 15000 steps, which is 600 epochs

refactor to run pretraining for 600/3000 epochs: first 20%
"""
