import librosa
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from melgan_models import Discriminator, Generator, MultiResolutionSTFTLoss
from pwg_melgan_models import MelGANGenerator, MelGANMultiScaleDiscriminator
from pqmf import PQMF
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.utilities import rank_zero_only

from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.helpers.helpers import plot_spectrogram_to_numpy
from nemo.core.classes import ModelPT
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.decorators import experimental
from nemo.utils.exp_manager import exp_manager
from nemo.core.optim.lr_scheduler import CosineAnnealing


@experimental
class MBMelGanModel(ModelPT):
    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        super().__init__(cfg=cfg, trainer=trainer)

        self.audio_to_melspec_precessor = instantiate(self._cfg.preprocessor)
        self.generator = MelGANGenerator(**cfg.gen)
        self.discriminator = MelGANMultiScaleDiscriminator(**cfg.disc)
        self.pqmf = PQMF()
        self.loss = MultiResolutionSTFTLoss()
        self.subband_loss = MultiResolutionSTFTLoss(
            fft_sizes=[384, 683, 171], hop_sizes=[30, 60, 10], win_lengths=[150, 300, 60]
        )
        self.train_disc = False
        self.mse_loss = torch.nn.MSELoss()
        self.adv_coeff = self._cfg.init_adv_lambda
        self.increase_coeff = self._cfg.increase_lambda

    @property
    def input_types(self):
        pass

    @property
    def output_types(self):
        pass

    def configure_optimizers(self):
        opt1 = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3, eps=1e-07, amsgrad=True)
        opt2 = torch.optim.Adam(self.generator.parameters(), lr=1e-3, eps=1e-07, amsgrad=True)
        iters_per_batch = self._trainer.max_epochs / float(
            self._trainer.num_gpus * self._trainer.num_nodes * self._trainer.accumulate_grad_batches
        )
        num_samples = len(self._train_dl.dataset)
        batch_size = self._train_dl.batch_size
        max_steps = round(num_samples * iters_per_batch / float(batch_size))
        logging.info(f"MAX STEPS: {max_steps}")
        # sch1 = torch.optim.lr_scheduler.MultiStepLR(opt1, milestones=[400, 800, 1200, 1600, 2000, 2400], gamma=0.5)
        # sch2 = torch.optim.lr_scheduler.MultiStepLR(opt2, milestones=[400, 800, 1200, 1600, 2000, 2400], gamma=0.5)
        sch1 = CosineAnnealing(opt1, max_steps=max_steps, min_lr=1e-5, warmup_steps=15000)  # Use warmup to delay start
        sch2 = CosineAnnealing(opt2, max_steps=max_steps, min_lr=1e-5)
        return [opt1, opt2], [sch1, sch2]

    def forward(self):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        # TODO, manual logging to tensorboard because lightning isn't working for some reason
        audio, audio_len = batch
        spec, _ = self.audio_to_melspec_precessor(audio, audio_len)
        mb_audio_pred = self.generator(spec)

        # train discriminator
        if optimizer_idx == 0 and self.train_disc:
            audio_pred = self.pqmf.synthesis(mb_audio_pred)
            fake_score = self.discriminator(audio_pred.detach())
            real_score = self.discriminator(audio.unsqueeze(1))

            loss_disc = 0
            # for scale in fake_score:
            #     loss_disc += F.relu(1 + scale[-1]).mean()

            # for scale in real_score:
            #     loss_disc += F.relu(1 - scale[-1]).mean()

            loss_disc = 0.0
            for i in range(len(fake_score)):
                loss_disc += self.mse_loss(real_score[i][-1], real_score[i][-1].new_ones(real_score[i][-1].size()))
                loss_disc += self.mse_loss(fake_score[i][-1], fake_score[i][-1].new_zeros(fake_score[i][-1].size()))
            loss_disc /= len(fake_score)

            if self.global_step % 1000:
                self.logger.experiment.add_scalar("loss_discriminator", loss_disc, self.global_step)
            return {
                'loss': loss_disc,
                'progress_bar': {'loss_disc': loss_disc},
                # 'log': {'loss_discriminator': loss_disc},
            }
        # train generator
        elif optimizer_idx == 1:
            loss = 0
            audio_pred = self.pqmf.synthesis(mb_audio_pred)

            # full-band loss
            sc_loss, mag_loss = self.loss(audio_pred.squeeze(1), audio)
            loss_feat = sc_loss + mag_loss
            loss += 0.5 * loss_feat

            # MB loss
            audio_mb = self.pqmf.analysis(audio.unsqueeze(1))
            mb_audio_pred = mb_audio_pred.view(-1, mb_audio_pred.size(2))
            audio_mb = audio_mb.view(-1, audio_mb.size(2))
            sub_sc_loss, sub_mag_loss = self.subband_loss(mb_audio_pred, audio_mb)
            loss += 0.5 * (sub_sc_loss + sub_mag_loss)

            if self.train_disc:
                fake_score = self.discriminator(audio_pred)
                # real_score = self.discriminator(audio.unsqueeze(1))

                # loss_gen = 0
                # for scale in fake_score:
                #     loss_gen += -scale[-1].mean()

                loss_gen = 0
                for scale in fake_score:
                    loss_gen += self.mse_loss(scale[-1], scale[-1].new_ones(scale[-1].size()))
                loss_gen /= len(fake_score)

                loss += self.adv_coeff * loss_gen
            if self.global_step % 1000:
                self.logger.experiment.add_scalar("loss_generator", loss, self.global_step)
            return {
                'loss': loss,
                'progress_bar': {'loss_gen': loss},
                # 'log': {'loss_generator': loss},
            }
        return None

    def validation_step(self, batch, batch_idx):
        audio, audio_len = batch
        spec, _ = self.audio_to_melspec_precessor(audio, audio_len)
        audio_pred_mb = self.generator(spec)

        # return result
        audio_pred = self.pqmf.synthesis(audio_pred_mb)
        spec_pred, _ = self.audio_to_melspec_precessor(audio_pred.squeeze(1), audio_len)
        return {
            # "loss": loss,
            "spec": spec,
            "spec_pred": spec_pred,
        }

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
        return {}

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
        if self.global_step >= 15000:
            self.train_disc = True

        # Add staircase increase for adv_coeff
        if self.increase_coeff:
            if self.global_step >= 55000:
                self.adv_coeff = 10
            elif self.global_step >= 45000:
                self.adv_coeff = 7.5
            elif self.global_step >= 35000:
                self.adv_coeff = 5
            elif self.global_step >= 25000:
                self.adv_coeff = 2.5
        return super().training_epoch_end(outputs)

    @classmethod
    def list_available_models(cls) -> 'Optional[Dict[str, str]]':
        pass


@hydra_runner(config_path="conf", config_name="multiband_melgan")
def train_pwg(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = MBMelGanModel(cfg=cfg.model, trainer=trainer)
    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])
    trainer.fit(model)


@hydra_runner(config_path="conf", config_name="multiband_melgan")
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
    librosa.output.write_wav("MB_MelGAN_0.wav", audio_pred[0].cpu().numpy().squeeze(), sr=22050)

    from matplotlib import pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.imshow(spec[0].cpu().numpy().squeeze(), origin="lower")
    ax2.imshow(spec_out.cpu().numpy().squeeze(), origin="lower")
    plt.savefig("MB_MelGAN_0")
    print(audio_pred[0])
    print(audio[0])
    print(spec[0].cpu().numpy().shape)
    print(spec_out.cpu().numpy().shape)


@hydra_runner(config_path="conf", config_name="multiband_melgan")
def infer_pwg_batch(cfg):
    model = MBMelGanModel.load_from_checkpoint(
        # "/home/jasoli/nemo/NeMo/examples/tts/experiments/1501438_MelGAN_MSE_BS64_E3000/MB_MelGan--last.ckpt"
        # "/mnt/hdd/experiment_results/MelGAN/1528354_MelGAN_LSE_GenPre/MB_MelGan/2020-10-16_22-15-23/checkpoints/MB_MelGan--end.ckpt"
        "/mnt/hdd/experiment_results/MelGAN/1528357_LinGAN_LSE_GenPre/MB_MelGan/2020-10-16_22-15-25/checkpoints/MB_MelGan--end.ckpt"
    )
    # "/home/jasoli/nemo/NeMo/examples/tts/nemo_experiments/MB_MelGan/2020-09-24_11-29-43/checkpoints/MB_MelGan--last.ckpt"
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
            audio_pred = model.pqmf.synthesis(mb_audio_pred)
        for i, single_audio in enumerate(audio_pred):
            print(single_audio.cpu().numpy().squeeze())
            librosa.output.write_wav(
                f"MB_LinGAN_{i}.wav", single_audio.cpu().numpy().squeeze()[: audio_len[i]], sr=22050
            )
        break


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

if __name__ == '__main__':
    # main()  # noqa pylint: disable=no-value-for-parameter
    # infer()
    train_pwg()
    # infer_pwg()
    # infer_pwg_batch()
    pass

"""
Currently at 25 steps per epoch
Runs pretraining for 15000 steps, which is 600 epochs
"""
