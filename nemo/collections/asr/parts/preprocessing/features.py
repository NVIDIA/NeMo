# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
#
# Copyright (c) 2018 Ryan Leary
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# This file contains code artifacts adapted from https://github.com/ryanleary/patter

import math

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.util import tiny
from packaging import version
from torch.autograd import Variable
from torch_stft import STFT

from nemo.collections.asr.parts.preprocessing.perturb import AudioAugmentor
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.common.parts.patch_utils import stft_patch
from nemo.utils import logging

try:
    import nvidia.dali as dali
    from nvidia.dali.plugin.pytorch import feed_ndarray as feed_ndarray

    DALI_VERSION = version.parse(dali.__version__)
    DALI_VERSION_MIN = version.parse('1.0.0')

    HAVE_DALI = True
except ModuleNotFoundError:
    HAVE_DALI = False


class DALIContext:
    """
        Context used keep the DALI pipeline and related parameters
    """

    def __init__(self):
        self.pipe = None
        self.batch_size = None
        self.device = None
        self.device_id = None


CONSTANT = 1e-5


def normalize_batch(x, seq_len, normalize_type):
    if normalize_type == "per_feature":
        x_mean = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        x_std = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            if x[i, :, : seq_len[i]].shape[1] == 1:
                raise ValueError(
                    "normalize_batch with `per_feature` normalize_type received a tensor of length 1. This will result "
                    "in torch.std() returning nan"
                )
            x_mean[i, :] = x[i, :, : seq_len[i]].mean(dim=1)
            x_std[i, :] = x[i, :, : seq_len[i]].std(dim=1)
        # make sure x_std is not zero
        x_std += CONSTANT
        return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)
    elif normalize_type == "all_features":
        x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            x_mean[i] = x[i, :, : seq_len[i].item()].mean()
            x_std[i] = x[i, :, : seq_len[i].item()].std()
        # make sure x_std is not zero
        x_std += CONSTANT
        return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1)
    elif "fixed_mean" in normalize_type and "fixed_std" in normalize_type:
        x_mean = torch.tensor(normalize_type["fixed_mean"], device=x.device)
        x_std = torch.tensor(normalize_type["fixed_std"], device=x.device)
        return (x - x_mean.view(x.shape[0], x.shape[1]).unsqueeze(2)) / x_std.view(x.shape[0], x.shape[1]).unsqueeze(2)
    else:
        return x


def splice_frames(x, frame_splicing):
    """ Stacks frames together across feature dim

    input is batch_size, feature_dim, num_frames
    output is batch_size, feature_dim*frame_splicing, num_frames

    """
    seq = [x]
    for n in range(1, frame_splicing):
        seq.append(torch.cat([x[:, :, :n], x[:, :, n:]], dim=2))
    return torch.cat(seq, dim=1)


class WaveformFeaturizer(object):
    def __init__(self, sample_rate=16000, int_values=False, augmentor=None):
        self.augmentor = augmentor if augmentor is not None else AudioAugmentor()
        self.sample_rate = sample_rate
        self.int_values = int_values

    def max_augmentation_length(self, length):
        return self.augmentor.max_augmentation_length(length)

    def process(self, file_path, offset=0, duration=0, trim=False, orig_sr=None):
        audio = AudioSegment.from_file(
            file_path,
            target_sr=self.sample_rate,
            int_values=self.int_values,
            offset=offset,
            duration=duration,
            trim=trim,
            orig_sr=orig_sr,
        )
        return self.process_segment(audio)

    def process_segment(self, audio_segment):
        self.augmentor.perturb(audio_segment)
        return torch.tensor(audio_segment.samples, dtype=torch.float)

    @classmethod
    def from_config(cls, input_config, perturbation_configs=None):
        if perturbation_configs is not None:
            aa = AudioAugmentor.from_config(perturbation_configs)
        else:
            aa = None

        sample_rate = input_config.get("sample_rate", 16000)
        int_values = input_config.get("int_values", False)

        return cls(sample_rate=sample_rate, int_values=int_values, augmentor=aa)


class FeaturizerFactory(object):
    def __init__(self):
        pass

    @classmethod
    def from_config(cls, input_cfg, perturbation_configs=None):
        return WaveformFeaturizer.from_config(input_cfg, perturbation_configs=perturbation_configs)


# Create helper class to patch forward func for use with AMP
class STFTPatch(STFT):
    def forward(self, input_data):
        return super().transform(input_data)[0]


# Create helper class for STFT that yields num_frames = num_samples // hop_length
class STFTExactPad(STFTPatch):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(self, *params, **kw_params):
        super().__init__(*params, **kw_params)
        self.pad_amount = (self.filter_length - self.hop_length) // 2

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        )

        if self.window is not None:
            window_sum = librosa.filters.window_sumsquare(
                self.window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.filter_length,
                dtype=np.float32,
            )
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(torch.from_numpy(window_sum), requires_grad=False).to(
                magnitude.device
            )
            inverse_transform[..., approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= self.filter_length / self.hop_length

        inverse_transform = inverse_transform[..., self.pad_amount :]
        inverse_transform = inverse_transform[..., : -self.pad_amount :]
        inverse_transform = inverse_transform.squeeze(1)

        return inverse_transform


class FilterbankFeatures(nn.Module):
    """Featurizer that converts wavs to Mel Spectrograms.
    See AudioToMelSpectrogramPreprocessor for args.
    """

    def __init__(
        self,
        sample_rate=16000,
        n_window_size=320,
        n_window_stride=160,
        window="hann",
        normalize="per_feature",
        n_fft=None,
        preemph=0.97,
        nfilt=64,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2 ** -24,
        dither=CONSTANT,
        pad_to=16,
        max_duration=16.7,
        frame_splicing=1,
        exact_pad=False,
        stft_exact_pad=False,  # TODO: Remove this in 1.1.0
        stft_conv=False,  # TODO: Remove this in 1.1.0
        pad_value=0,
        mag_power=2.0,
        use_grads=False,
        use_dali=False,
    ):
        super().__init__()
        self.use_dali = use_dali
        if self.use_dali:
            if not HAVE_DALI:
                raise ModuleNotFoundError(
                    "AudioToMelSpectrogramPreprocessor: use_dali was set to True, but NVIDIA DALI is not installed."
                    "To install, please follow https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html#nvidia-dali"
                )
            else:
                logging.warning(
                    f'Using NVIDIA DALI for feature extraction is experimental, not ready for production and is not fully supported. Use at your own risk.'
                )
        self.dali_ctx = DALIContext()

        if stft_conv or stft_exact_pad:
            logging.warning(
                "Using torch_stft is deprecated and will be removed in 1.1.0. Please set stft_conv and stft_exact_pad "
                "to False for FilterbankFeatures and AudioToMelSpectrogramPreprocessor. Please set exact_pad to True "
                "as needed."
            )
        if (exact_pad or stft_exact_pad) and n_window_stride % 2 == 1:
            raise NotImplementedError(
                f"{self} received exact_pad == True, but hop_size was odd. If audio_length % hop_size == 0. Then the "
                "returned spectrogram would not be of length audio_length // hop_size. Please use an even hop_size."
            )
        self.log_zero_guard_value = log_zero_guard_value
        if (
            n_window_size is None
            or n_window_stride is None
            or not isinstance(n_window_size, int)
            or not isinstance(n_window_stride, int)
            or n_window_size <= 0
            or n_window_stride <= 0
        ):
            raise ValueError(
                f"{self} got an invalid value for either n_window_size or "
                f"n_window_stride. Both must be positive ints."
            )
        logging.info(f"PADDING: {pad_to}")

        self.sample_rate = sample_rate
        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.stft_pad_amount = (self.n_fft - self.hop_length) // 2 if exact_pad else None
        self.stft_exact_pad = stft_exact_pad
        self.stft_conv = stft_conv

        if use_dali:
            logging.info("STFT using NVIDIA DALI")
        if stft_conv:
            logging.info("STFT using conv")
            if stft_exact_pad:
                logging.info("STFT using exact pad")
                self.stft = STFTExactPad(self.n_fft, self.hop_length, self.win_length, window)
            else:
                self.stft = STFTPatch(self.n_fft, self.hop_length, self.win_length, window)
        else:
            logging.info("STFT using torch")
            if exact_pad:
                logging.info("STFT using exact pad")
            torch_windows = {
                'hann': torch.hann_window,
                'hamming': torch.hamming_window,
                'blackman': torch.blackman_window,
                'bartlett': torch.bartlett_window,
                'none': None,
            }
            window_fn = torch_windows.get(window, None)
            window_tensor = window_fn(self.win_length, periodic=False) if window_fn else None
            self.window_tensor_lst = window_tensor.numpy().tolist() if window_tensor is not None else None
            self.register_buffer("window", window_tensor)
            self.stft = lambda x: stft_patch(
                x,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                center=False if exact_pad else True,
                window=self.window.to(dtype=torch.float),
                return_complex=False,
            )

        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.nfilt = nfilt
        self.preemph = preemph
        self.pad_to = pad_to
        self.lowfreq = lowfreq
        self.highfreq = highfreq = highfreq or sample_rate / 2

        filterbanks = torch.tensor(
            librosa.filters.mel(sample_rate, self.n_fft, n_mels=nfilt, fmin=lowfreq, fmax=highfreq), dtype=torch.float
        ).unsqueeze(0)
        self.register_buffer("fb", filterbanks)

        # Calculate maximum sequence length
        max_length = self.get_seq_len(torch.tensor(max_duration * sample_rate, dtype=torch.float))
        max_pad = pad_to - (max_length % pad_to) if pad_to > 0 else 0
        self.max_length = max_length + max_pad
        self.pad_value = pad_value
        self.mag_power = mag_power

        # We want to avoid taking the log of zero
        # There are two options: either adding or clamping to a small value
        if log_zero_guard_type not in ["add", "clamp"]:
            raise ValueError(
                f"{self} received {log_zero_guard_type} for the "
                f"log_zero_guard_type parameter. It must be either 'add' or "
                f"'clamp'."
            )

        self.use_grads = use_grads
        if not use_grads:
            self.forward = torch.no_grad()(self.forward)

        # log_zero_guard_value is the the small we want to use, we support
        # an actual number, or "tiny", or "eps"
        self.log_zero_guard_type = log_zero_guard_type
        logging.debug(f"sr: {sample_rate}")
        logging.debug(f"n_fft: {self.n_fft}")
        logging.debug(f"win_length: {self.win_length}")
        logging.debug(f"hop_length: {self.hop_length}")
        logging.debug(f"n_mels: {nfilt}")
        logging.debug(f"fmin: {lowfreq}")
        logging.debug(f"fmax: {highfreq}")
        logging.debug(f"using grads: {use_grads}")

    def can_use_dali(self):
        return self.use_dali

    def log_zero_guard_value_fn(self, dtype):
        if isinstance(self.log_zero_guard_value, str):
            if self.log_zero_guard_value == "tiny":
                return torch.finfo(dtype).tiny
            elif self.log_zero_guard_value == "eps":
                return torch.finfo(dtype).eps
            else:
                raise ValueError(
                    f"{self} received {self.log_zero_guard_value} for the "
                    f"log_zero_guard_type parameter. It must be either a "
                    f"number, 'tiny', or 'eps'"
                )
        else:
            return self.log_zero_guard_value

    def get_seq_len(self, seq_len):
        if isinstance(self.stft, STFT):
            pad_amount = self.stft.pad_amount * 2
        else:
            # Assuming that center is True is stft_pad_amount = 0
            pad_amount = self.stft_pad_amount * 2 if self.stft_pad_amount is not None else self.n_fft // 2 * 2
        seq_len = torch.floor((seq_len + pad_amount - self.n_fft) / self.hop_length) + 1
        return seq_len.to(dtype=torch.long)

    @property
    def filter_banks(self):
        return self.fb

    def forward_torch(self, x, seq_len):
        seq_len = self.get_seq_len(seq_len.float())

        if self.stft_pad_amount is not None:
            x = torch.nn.functional.pad(
                x.unsqueeze(1), (self.stft_pad_amount, self.stft_pad_amount), "reflect"
            ).squeeze(1)

        # dither (only in training mode for eval determinism)
        if self.training and self.dither > 0:
            x += self.dither * torch.randn_like(x)

        # do preemphasis
        if self.preemph is not None:
            x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1)

        # disable autocast to get full range of stft values
        with torch.cuda.amp.autocast(enabled=False):
            x = self.stft(x)

        # torch returns real, imag; so convert to magnitude
        if not self.stft_conv:
            # guard is needed for sqrt if grads are passed through
            guard = 0 if not self.use_grads else CONSTANT
            if x.dtype in [torch.cfloat, torch.cdouble]:
                x = torch.view_as_real(x)
            x = torch.sqrt(x.pow(2).sum(-1) + guard)

        # get power spectrum
        if self.mag_power != 1.0:
            x = x.pow(self.mag_power)

        # dot with filterbank energies
        x = torch.matmul(self.fb.to(x.dtype), x)

        # log features if required
        if self.log:
            if self.log_zero_guard_type == "add":
                x = torch.log(x + self.log_zero_guard_value_fn(x.dtype))
            elif self.log_zero_guard_type == "clamp":
                x = torch.log(torch.clamp(x, min=self.log_zero_guard_value_fn(x.dtype)))
            else:
                raise ValueError("log_zero_guard_type was not understood")

        # frame splicing if required
        if self.frame_splicing > 1:
            x = splice_frames(x, self.frame_splicing)

        # normalize if required
        if self.normalize:
            x = normalize_batch(x, seq_len, normalize_type=self.normalize)

        # mask to zero any values beyond seq_len in batch, pad to multiple of `pad_to` (for efficiency)
        max_len = x.size(-1)
        mask = torch.arange(max_len).to(x.device)
        mask = mask.expand(x.size(0), max_len) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).type(torch.bool).to(device=x.device), self.pad_value)
        del mask
        pad_to = self.pad_to
        if pad_to == "max":
            x = nn.functional.pad(x, (0, self.max_length - x.size(-1)), value=self.pad_value)
        elif pad_to > 0:
            pad_amt = x.size(-1) % pad_to
            if pad_amt != 0:
                x = nn.functional.pad(x, (0, pad_to - pad_amt), value=self.pad_value)
        return x, seq_len

    def _daligraph_reflect_pad(self, x, x_len, stft_pad_amount):
        def flip_1d(x):
            # TODO(janton): remove the layout trick when Flip supports arbitrary data layouts
            x = dali.fn.reshape(x, shape=(-1, 1, 1), layout="HWC")
            x = dali.fn.flip(x, vertical=1)
            x = dali.fn.reshape(x, shape=(-1,), layout="t")
            return x

        pad_start = dali.fn.slice(x, start=1, shape=stft_pad_amount, axes=(0,))
        pad_start = flip_1d(pad_start)

        pad_end = dali.fn.slice(x, start=(x_len - stft_pad_amount - 1), shape=stft_pad_amount, axes=(0,))
        pad_end = flip_1d(pad_end)
        x = dali.fn.cat(pad_start, x, pad_end, axis=0)
        return x

    def _daligraph_splice_frames(self, x, nfeatures, x_len, stacking=1, subsampling=1):
        if stacking > 1:
            seq = [x]
            for n in range(1, stacking):
                f = dali.fn.slice(x, start=n, shape=x_len, axes=(1,), out_of_bounds_policy='pad', fill_values=0)
                seq.append(f)
            x = dali.fn.cat(*seq, axis=0)
            nfeatures = nfeatures * stacking
        if subsampling > 1:
            out_len = (x_len + subsampling - 1) // subsampling
            m = dali.fn.transforms.scale(scale=[subsampling, 1], center=[0.5, 0])
            x = dali.fn.reshape(x, rel_shape=[1, 1, -1], layout="HWC")  # Layout required by WarpAffine
            x = dali.fn.warp_affine(
                x, matrix=m, size=dali.fn.cat(nfeatures, out_len), interp_type=dali.types.INTERP_NN
            )
            x = dali.fn.reshape(x, rel_shape=[1, 1], layout="ft")
        return x

    def init_dali_pipeline(self, batch_size, device, device_id, num_threads=4):
        pipe = dali.pipeline.Pipeline(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            prefetch_queue_depth=1,
            exec_async=True,
            exec_pipelined=True,
        )
        with pipe:
            audio = dali.fn.external_source(name="input_signal", device=device)
            audio_len = dali.fn.external_source(name="length", device='cpu')

            if self.stft_pad_amount is not None:
                audio = self._daligraph_reflect_pad(audio, audio_len, self.stft_pad_amount)
                audio_len = audio_len + 2 * self.stft_pad_amount

            spec_len = dali.fn.cast((audio_len // self.hop_length) + 1, dtype=dali.types.INT64)

            # Additive gaussian noise (dither)
            if self.training and self.dither > 0.0:
                gaussian_noise = dali.fn.random.normal(device=device)
                audio = audio + self.dither * gaussian_noise

            # Preemphasis filter
            if self.preemph > 0.0:
                audio = dali.fn.preemphasis_filter(audio, preemph_coeff=self.preemph)

            # Power spectrogram
            spec = dali.fn.spectrogram(
                audio,
                nfft=self.n_fft,
                power=self.mag_power,
                window_fn=self.window_tensor_lst,
                window_length=self.win_length,
                window_step=self.hop_length,
                center_windows=True,
                reflect_padding=True,
            )

            # Spectrogram to Mel Spectrogram
            spec = dali.fn.mel_filter_bank(
                spec, sample_rate=self.sample_rate, nfilter=self.nfilt, freq_low=self.lowfreq, freq_high=self.highfreq,
            )

            # log features if required
            if self.log:
                eps = self.log_zero_guard_value_fn(torch.float32)
                if self.log_zero_guard_type == "add":
                    spec = spec + eps
                elif self.log_zero_guard_type == "clamp":
                    spec = dali.math.max(spec, eps)
                else:
                    raise ValueError("log_zero_guard_type was not understood")
                # Natural Logarithm
                spec = dali.fn.to_decibels(spec, multiplier=math.log(10), reference=1.0, cutoff_db=-120)

            if self.frame_splicing > 1:
                spec = self._daligraph_splice_frames(spec, self.nfilt, spec_len, stacking=self.frame_splicing)

            # Trimming Spectrogram to match the reference implementation
            start = dali.types.Constant(0, shape=[], dtype=dali.types.INT64, device='cpu')
            spec = dali.fn.slice(spec, start, spec_len, axes=(1,))

            # Normalization
            normalization_axes = None
            if self.normalize:
                if self.normalize == "per_feature":
                    normalization_axes = [1]
                elif self.normalize == "all_features":
                    normalization_axes = [0, 1]
                elif "fixed_mean" in self.normalize and "fixed_std" in self.normalize:
                    raise ValueError("Normalization with fixed mean/stddev not yet supported.")
                    # TODO: implement
                else:
                    raise ValueError(f"Unknown normalization type: {self.normalize}")

                # Normalization
                spec = dali.fn.normalize(spec, axes=normalization_axes, epsilon=1e-5 ** 2, ddof=1)

            # Pads temporal dimension to take the length of the longest sample in the batch, padded up to a multiple of ``pad_to``
            pad_align = (self.pad_to,) if self.pad_to > 0 else None
            pad_shape = (self.max_length,) if self.pad_to == 'max' else (-1,)
            spec = dali.fn.pad(spec, fill_value=self.pad_value, axes=(1,), align=pad_align, shape=pad_shape)
            if device == 'gpu':
                spec_len = spec_len.gpu()
        pipe.set_outputs(spec, spec_len)
        # Building DALI pipeline
        pipe.build()
        return pipe

    def forward_dali(self, x, x_len):
        device = x.device
        device_str = str(device)
        device_str_toks = device_str.split(':')
        device_type_str = 'gpu' if device_str_toks[0] == 'cuda' else 'cpu'
        device_id = None
        if device_type_str == 'gpu':
            device_id = int(device_str_toks[1])
            cuda_stream = torch.cuda.current_stream(device=device)
        else:
            device_id = None
            cuda_stream = None
        batch_size = x.shape[0]
        if (
            self.dali_ctx.batch_size != batch_size
            or self.dali_ctx.device != device_type_str
            or self.dali_ctx.device_id != device_id
        ):
            self.dali_ctx.batch_size = batch_size
            self.dali_ctx.device = device_type_str
            self.dali_ctx.device_id = device_id
            self.dali_ctx.pipe = self.init_dali_pipeline(batch_size, device_type_str, device_id)

        self.dali_ctx.pipe.feed_input("input_signal", x)
        self.dali_ctx.pipe.feed_input("length", x_len.cpu())
        out0, out1 = self.dali_ctx.pipe.run()

        out0 = out0.as_tensor()
        out1 = out1.as_tensor()

        processed_signal = torch.zeros(out0.shape(), dtype=torch.float32, device=device)
        feed_ndarray(out0, processed_signal, cuda_stream=cuda_stream)

        processed_length = torch.zeros(out1.shape(), dtype=torch.long, device=device)
        feed_ndarray(out1, processed_length, cuda_stream=cuda_stream)
        processed_length = processed_length.reshape(-1)

        return processed_signal, processed_length

    def forward(self, input_signal, length):
        if self.can_use_dali():
            processed_signal, processed_length = self.forward_dali(input_signal, length)
        else:
            processed_signal, processed_length = self.forward_torch(input_signal, length)
        return processed_signal, processed_length
