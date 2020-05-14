# Copyright (c) 2019 NVIDIA Corporation
import librosa
import numpy as np
import torch

from nemo import logging
from nemo.backends.pytorch.nm import LossNM, TrainableNM
from nemo.collections.tts.parts.waveglow import WaveGlow, remove_weightnorm
from nemo.core.neural_types import *
from nemo.utils.decorators import add_port_docs

__all__ = ["WaveGlowNM", "WaveGlowInferNM", "WaveGlowLoss"]


class WaveGlowNM(TrainableNM):
    """
    WaveGlowNM implements the Waveglow model in whole. This NM is meant to
    be used during training

    Args:
        n_mel_channels (int): Size of input mel spectrogram
            Defaults to 80.
        n_flows (int): Number of normalizing flows/layers of waveglow.
            Defaults to 12
        n_group (int): Each audio/spec pair is split in n_group number of
            groups. It must be divisible by 2 as halves are split this way.
            Defaults to 8
        n_early_every (int): After n_early_every layers, n_early_size number of
            groups are skipped to the output of the Neural Module.
            Defaults to 4
        n_early_size (int): The number of groups to skip to the output at every
            n_early_every layers.
            Defaults to 2
        n_wn_layers (int): The number of layers of the wavenet submodule.
            Defaults to 8
        n_wn_channels (int): The number of channels of the wavenet submodule.
            Defaults to 512
        wn_kernel_size (int): The kernel size of the wavenet submodule.
            Defaults to 3
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # "mel_spectrogram": NeuralType(
            #     {0: AxisType(BatchTag), 1: AxisType(MelSpectrogramSignalTag), 2: AxisType(TimeTag),}
            # ),
            # "audio": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "mel_spectrogram": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "audio": NeuralType(('B', 'T'), AudioSignal(self.sample_rate)),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        # TODO @blisc: please take a look at those definitions
        return {
            # "audio": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # "log_s_list": NeuralType(),
            # "log_det_W_list": NeuralType(),
            "audio": NeuralType(('B', 'T'), AudioSignal(self.sample_rate)),
            "log_s_list": NeuralType(elements_type=ChannelType()),
            "log_det_W_list": NeuralType(elements_type=ChannelType()),
        }

    def __init__(
        self,
        sample_rate: int,
        n_mel_channels: int = 80,
        n_flows: int = 12,
        n_group: int = 8,
        n_early_every: int = 4,
        n_early_size: int = 2,
        n_wn_layers: int = 8,
        n_wn_channels: int = 512,
        wn_kernel_size: int = 3,
    ):
        self.sample_rate = sample_rate
        super().__init__()
        wavenet_config = {
            "n_layers": n_wn_layers,
            "n_channels": n_wn_channels,
            "kernel_size": wn_kernel_size,
        }
        self.waveglow = WaveGlow(
            n_mel_channels=n_mel_channels,
            n_flows=n_flows,
            n_group=n_group,
            n_early_every=n_early_every,
            n_early_size=n_early_size,
            WN_config=wavenet_config,
        )
        self.to(self._device)

    def forward(self, mel_spectrogram, audio):
        # This function should probably be split
        # If training, it returns the predicted normal distribution
        # Else it returns the predicted audio
        if self.training:
            audio, log_s_list, log_det_W_list = self.waveglow((mel_spectrogram, audio))
        else:
            audio = self.waveglow.infer(mel_spectrogram)
            log_s_list = log_det_W_list = []
        return audio, log_s_list, log_det_W_list


class WaveGlowInferNM(WaveGlowNM):
    """
    WaveGlowInferNM is the inference Neural Module for WaveGlowNM. This NM is
    meant to be used during inference. Keep in mind, the inference module
    runs in the reverse order of the training module.

    Args:
        n_mel_channels (int): Size of input mel spectrogram
            Defaults to 80.
        n_flows (int): Number of normalizing flows/layers of waveglow.
            Defaults to 12
        n_group (int): Each audio/spec pair is split in n_group number of
            groups. It must be divisible by 2 as halves are split this way.
            Defaults to 8
        n_early_every (int): After n_early_every layers, n_early_size number of
            groups are added as input to the current layer.
            Defaults to 4
        n_early_size (int): The number of groups to sample at every
            n_early_every layers. The sampled values are then passed through
            the remaining layer.
            Defaults to 2
        n_wn_layers (int): The number of layers of the wavenet submodule.
            Defaults to 8
        n_wn_channels (int): The number of channels of the wavenet submodule.
            Defaults to 512
        wn_kernel_size (int): The kernel size of the wavenet submodule.
            Defaults to 3
        sigma (float): Standard deviation of the normal distribution from which
            we sample z. Defaults to 0.6.
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # "mel_spectrogram": NeuralType(
            #     {0: AxisType(BatchTag), 1: AxisType(MelSpectrogramSignalTag), 2: AxisType(TimeTag),}
            # )
            "mel_spectrogram": NeuralType(('B', 'D', 'T'), MelSpectrogramType())
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        # return {"audio": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)})}
        return {"audio": NeuralType(('B', 'T'), AudioSignal(freq=self.sample_rate))}

    def __str__(self):
        return "WaveGlowNM"

    def __init__(
        self,
        *,
        sample_rate: int,
        n_mel_channels: int = 80,
        n_flows: int = 12,
        n_group: int = 8,
        n_early_every: int = 4,
        n_early_size: int = 2,
        n_wn_layers: int = 8,
        n_wn_channels: int = 512,
        wn_kernel_size: int = 3,
        sigma: float = 0.6,
    ):
        self._sigma = sigma
        # self.sample_rate = sample_rate  # Done in parent class
        super().__init__(
            sample_rate=sample_rate,
            n_mel_channels=n_mel_channels,
            n_flows=n_flows,
            n_group=n_group,
            n_early_every=n_early_every,
            n_early_size=n_early_size,
            n_wn_layers=n_wn_layers,
            n_wn_channels=n_wn_channels,
            wn_kernel_size=wn_kernel_size,
        )
        self._removed_weight_norm = False

    def setup_denoiser(self):
        with torch.no_grad():
            mel_input = torch.zeros((1, 80, 88), device=self._device)
            bias_audio = self.waveglow.infer(mel_input, sigma=0.0).float()
            bias_audio = bias_audio.squeeze().cpu().numpy()
            bias_spec, _ = librosa.core.magphase(librosa.core.stft(bias_audio, n_fft=1024))
            self.bias_spec = np.expand_dims(bias_spec[:, 0], axis=-1)

    def denoise(self, audio, strength=0.1):
        audio_spec, audio_angles = librosa.core.magphase(librosa.core.stft(audio, n_fft=1024))
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = np.clip(audio_spec_denoised, a_min=0.0, a_max=None)
        audio_denoised = librosa.core.istft(audio_spec_denoised * audio_angles)
        return audio_denoised, audio_spec_denoised

    def forward(self, mel_spectrogram):
        if not self._removed_weight_norm:
            logging.info("remove WN")
            self.waveglow = remove_weightnorm(self.waveglow)
            self._removed_weight_norm = True
        if self.training:
            raise ValueError("You are using the WaveGlow Infer Neural Module in training mode.")
        with torch.no_grad():
            audio = self.waveglow.infer(mel_spectrogram, sigma=self._sigma)
        return audio


class WaveGlowLoss(LossNM):
    """
    WaveGlowLoss implements the waveglow loss which aims to maximize the
    log-likelihood of the audio given the mel spectrogram. This loss is
    expressed as the log-likelihood of a standard normal distribution and the
    sum of the log of the determinant of the Jacobians of the mapping from
    x, audio, to z, the normal distribution. The second term can be further
    split in the contribution by the affine coupling layer, log_s, and the 1x1
    invertible convolution layer, log_det_W.

    Args:
        sigma (float): Standard deviation of the normal distribution that we
            are aiming to model.
            Defaults to 1.
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        # TODO @blisc: please take a look at those definitions
        return {
            # "z": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # "log_s_list": NeuralType(),
            # "log_det_W_list": NeuralType(),
            "z": NeuralType(('B', 'T'), AudioSignal(freq=self.sample_rate)),
            "log_s_list": NeuralType(elements_type=ChannelType()),
            "log_det_W_list": NeuralType(elements_type=ChannelType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, sample_rate: int, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma
        self.sample_rate = sample_rate

    def _loss_function(self, **kwargs):
        return self._loss(*(kwargs.values()))

    def _loss(self, z, log_s_list, log_det_W_list):
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s)
                log_det_W_total = log_det_W_list[i]
            else:
                log_s_total = log_s_total + torch.sum(log_s)
                log_det_W_total += log_det_W_list[i]

        loss = torch.sum(z * z) / (2 * self.sigma * self.sigma) - log_s_total - log_det_W_total
        return loss / (z.size(0) * z.size(1) * z.size(2))
