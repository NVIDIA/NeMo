# Copyright (c) 2019 NVIDIA Corporation
import librosa
import numpy as np
import torch

from nemo.backends.pytorch.nm import TrainableNM, LossNM
from nemo.core.neural_types import *
from .parts.waveglow import WaveGlow


class WaveGlowNM(TrainableNM):
    """ TODO: Docstring for Tacotron2Encdoer
    """
    @staticmethod
    def create_ports():
        input_ports = {
            "mel_spectrogram": NeuralType(
                {0: AxisType(BatchTag),
                 1: AxisType(MelSpectrogramSignalTag),
                 2: AxisType(TimeTag)}),
            "audio": NeuralType({0: AxisType(BatchTag),
                                 1: AxisType(TimeTag)})
        }

        output_ports = {
            "audio": NeuralType({0: AxisType(BatchTag),
                                 1: AxisType(TimeTag)}),
            "log_s_list": NeuralType(),
            "log_det_W_list": NeuralType(),
        }
        return input_ports, output_ports

    def __init__(
            self,
            n_mel_channels=80,
            n_flows=12,
            n_group=8,
            n_early_every=4,
            n_early_size=2,
            n_wn_layers=8,
            n_wn_channels=512,
            wn_kernel_size=3,
            **kwargs):
        super().__init__(**kwargs)
        wavenet_config = {
            "n_layers": n_wn_layers,
            "n_channels": n_wn_channels,
            "kernel_size": wn_kernel_size
        }
        self.waveglow = WaveGlow(
            n_mel_channels=n_mel_channels,
            n_flows=n_flows,
            n_group=n_group,
            n_early_every=n_early_every,
            n_early_size=n_early_size,
            WN_config=wavenet_config)
        self.to(self._device)

    def forward(self, mel_spectrogram, audio):
        if self.training:
            audio, log_s_list, log_det_W_list = self.waveglow(
                (mel_spectrogram, audio))
        else:
            audio = self.waveglow.infer(
                mel_spectrogram)
            log_s_list = log_det_W_list = []
        return audio, log_s_list, log_det_W_list


class WaveGlowInferNM(WaveGlowNM):
    """ TODO: Docstring for Tacotron2Encdoer
    """
    @staticmethod
    def create_ports():
        input_ports = {
            "mel_spectrogram": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(MelSpectrogramSignalTag),
                2: AxisType(TimeTag)})
        }

        output_ports = {
            "audio": NeuralType({0: AxisType(BatchTag),
                                 1: AxisType(TimeTag)})
        }
        return input_ports, output_ports

    def __str__(self):
        return "WaveGlowNM"

    def __init__(
            self,
            n_mel_channels=80,
            n_flows=12,
            n_group=8,
            n_early_every=4,
            n_early_size=2,
            n_wn_layers=8,
            n_wn_channels=512,
            wn_kernel_size=3,
            sigma=0.6,
            **kwargs):
        self._sigma = sigma
        super().__init__(
            n_mel_channels=n_mel_channels,
            n_flows=n_flows,
            n_group=n_group,
            n_early_every=n_early_every,
            n_early_size=n_early_size,
            n_wn_layers=n_wn_layers,
            n_wn_channels=n_wn_channels,
            wn_kernel_size=wn_kernel_size,
            **kwargs)
        self._removed_weight_norm = False

    def setup_denoiser(self):
        with torch.no_grad():
            mel_input = torch.zeros((1, 80, 88), device=self._device)
            bias_audio = self.waveglow.infer(mel_input, sigma=0.0).float()
            bias_audio = bias_audio.squeeze().cpu().numpy()
            bias_spec, _ = librosa.core.magphase(
                librosa.core.stft(bias_audio, n_fft=1024))
            self.bias_spec = np.expand_dims(bias_spec[:, 0], axis=-1)

    def denoise(self, audio, strength=0.1):
        audio_spec, audio_angles = librosa.core.magphase(
            librosa.core.stft(audio, n_fft=1024))
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = np.clip(
            audio_spec_denoised, a_min=0.0, a_max=None)
        audio_denoised = librosa.core.istft(audio_spec_denoised * audio_angles)
        return audio_denoised, audio_spec_denoised

    def forward(self, mel_spectrogram):
        if not self._removed_weight_norm:
            print("remove WN")
            self.waveglow = self.waveglow.remove_weightnorm(self.waveglow)
            self._removed_weight_norm = True
        if self.training:
            raise ValueError("You are using the WaveGlow Infer Neural Module "
                             "in training mode.")
        with torch.no_grad():
            audio = self.waveglow.infer(mel_spectrogram, sigma=self._sigma)
        return audio


class WaveGlowLoss(LossNM):
    """TODO
    """
    @staticmethod
    def create_ports():
        input_ports = {
            "audio_pred": NeuralType({0: AxisType(BatchTag),
                                      1: AxisType(TimeTag)}),
            "log_s_list": NeuralType(),
            "log_det_W_list": NeuralType(),
        }

        output_ports = {"loss": NeuralType(None)}
        return input_ports, output_ports

    def __init__(self, sigma=1.0, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma

    def _loss_function(self, **kwargs):
        return self._loss(*(kwargs.values()))

    def _loss(self, audio_pred, log_s_list, log_det_W_list):
        z = audio_pred
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s)
                log_det_W_total = log_det_W_list[i]
            else:
                log_s_total = log_s_total + torch.sum(log_s)
                log_det_W_total += log_det_W_list[i]

        loss = (torch.sum(z * z) / (2 * self.sigma * self.sigma)
                - log_s_total - log_det_W_total)
        return loss / (z.size(0) * z.size(1) * z.size(2))
