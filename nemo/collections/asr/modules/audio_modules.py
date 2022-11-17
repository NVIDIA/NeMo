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

from typing import Dict, Optional, Tuple

import torch

from nemo.collections.asr.parts.preprocessing.features import make_seq_mask_like
from nemo.collections.asr.parts.utils.audio_utils import db2mag
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import AudioSignal, FloatType, LengthsType, NeuralType, SpectrogramType
from nemo.utils import logging
from nemo.utils.decorators import experimental

try:
    import torchaudio

    HAVE_TORCHAUDIO = True
except ModuleNotFoundError:
    HAVE_TORCHAUDIO = False


__all__ = [
    'AudioToSpectrogram',
    'SpectrogramToAudio',
    'MaskEstimatorRNN',
    'MaskReferenceChannel',
    'MaskBasedBeamformer',
]


@experimental
class AudioToSpectrogram(NeuralModule):
    """Transform a batch of input multi-channel signals into a batch of
    STFT-based spectrograms.

    Args:
        fft_length: length of FFT
        hop_length: length of hops/shifts of the sliding window
        power: exponent for magnitude spectrogram. Default `None` will
               return a complex-valued spectrogram
    """

    def __init__(self, fft_length: int, hop_length: int, power: Optional[float] = None):
        if not HAVE_TORCHAUDIO:
            logging.error('Could not import torchaudio. Some features might not work.')

            raise ModuleNotFoundError(
                "torchaudio is not installed but is necessary to instantiate a {self.__class__.__name__}"
            )

        super().__init__()

        # For now, assume FFT length is divisible by two
        if fft_length % 2 != 0:
            raise ValueError(f'fft_length = {fft_length} must be divisible by 2')

        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=fft_length, hop_length=hop_length, power=power, pad_mode='constant'
        )

        # number of subbands
        self.F = fft_length // 2 + 1

    @property
    def num_subbands(self) -> int:
        return self.F

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports.
        """
        return {
            "input": NeuralType(('B', 'T', 'C'), AudioSignal()),
            "input_length": NeuralType(('B',), LengthsType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports.
        """
        return {
            "output": NeuralType(('B', 'D', 'T', 'C'), SpectrogramType()),
            "output_length": NeuralType(('B',), LengthsType()),
        }

    @typecheck()
    def forward(self, input: torch.Tensor, input_length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert a batch of C-channel input signals
        into a batch of complex-valued spectrograms.

        Args:
            input: Time-domain input signal with C channels, shape (B, T, C)
            input_length: Length of valid entries along the time dimension, shape (B,)

        Returns:
            Output spectrogram with F subbands and N time frames, shape (B, F, N, C)
            and output length with shape (B,).
        """
        output_length = self.get_output_length(input_length=input_length)

        B, T = input.shape[0:2]
        input = input.view(B, T, -1)

        # (B, T, C) -> (B, C, T)
        input = input.permute(0, 2, 1)

        # STFT output (B, C, F, N)
        with torch.cuda.amp.autocast(enabled=False):
            output = self.stft(input)

        # Permute channels to back
        output = output.permute(0, 2, 3, 1)

        # Mask padded frames
        length_mask: torch.Tensor = make_seq_mask_like(
            lengths=output_length, like=output, time_dim=-2, valid_ones=False
        )
        output = output.masked_fill(length_mask, 0.0)

        return output, output_length

    def get_output_length(self, input_length: torch.Tensor) -> torch.Tensor:
        """Get length of valid frames for the output.

        Args:
            input_length: number of valid samples, shape (B,)

        Returns:
            Number of valid frames, shape (B,)
        """
        output_length = input_length.div(self.stft.hop_length, rounding_mode='floor').add(1).long()
        return output_length


@experimental
class SpectrogramToAudio(NeuralModule):
    """Transform a batch of input multi-channel spectrograms into a batch of
    time-domain signals.

    Args:
        fft_length: length of FFT
        hop_length: length of hops/shifts of the sliding window
        power: exponent for magnitude spectrogram. Default `None` will
               return a complex-valued spectrogram
    """

    def __init__(self, fft_length: int, hop_length: int):
        if not HAVE_TORCHAUDIO:
            logging.error('Could not import torchaudio. Some features might not work.')

            raise ModuleNotFoundError(
                "torchaudio is not installed but is necessary to instantiate a {self.__class__.__name__}"
            )

        super().__init__()

        # For now, assume FFT length is divisible by two
        if fft_length % 2 != 0:
            raise ValueError(f'fft_length = {fft_length} must be divisible by 2')

        self.istft = torchaudio.transforms.InverseSpectrogram(
            n_fft=fft_length, hop_length=hop_length, pad_mode='constant'
        )

        self.F = fft_length // 2 + 1

    @property
    def num_subbands(self) -> int:
        return self.F

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports.
        """
        return {
            "input": NeuralType(('B', 'D', 'T', 'C'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports.
        """
        return {
            "output": NeuralType(('B', 'T', 'C'), AudioSignal()),
            "output_length": NeuralType(('B',), LengthsType()),
        }

    @typecheck()
    def forward(self, input: torch.Tensor, input_length: torch.Tensor) -> torch.Tensor:
        """Convert input complex-valued spectrogram to a time-domain
        signal. Multi-channel IO is supported.

        Args:
            input: Input spectrogram for C channels, shape (B, F, N, C)
            input_length: Length of valid entries along the time dimension, shape (B,)

        Returns:
            Time-domain signal with T time-domain samples and C channels, (B, T, C)
            and output length with shape (B,).
        """
        output_length = self.get_output_length(input_length=input_length)

        B, F, N = input.shape[0:3]
        assert F == self.F, f'Number of subbands F={F} not matching self.F={self.F}'
        input = input.view(B, F, N, -1)

        # (B, F, N, C) -> (B, C, F, N)
        input = input.permute(0, 3, 1, 2)

        # iSTFT output (B, C, T)
        with torch.cuda.amp.autocast(enabled=False):
            output = self.istft(input.cdouble())

        # (B, C, T) -> (B, T, C)
        output = output.permute(0, 2, 1)

        # Mask padded samples
        length_mask: torch.Tensor = make_seq_mask_like(
            lengths=output_length, like=output, time_dim=-2, valid_ones=False
        )
        output = output.masked_fill(length_mask, 0.0)

        return output, output_length

    def get_output_length(self, input_length: torch.Tensor) -> torch.Tensor:
        """Get length of valid samples for the output.

        Args:
            input_length: number of valid frames, shape (B,)

        Returns:
            Number of valid samples, shape (B,)
        """
        output_length = input_length.sub(1).mul(self.istft.hop_length).long()
        return output_length


@experimental
class SpatialFeatures(NeuralModule):
    """Generate spatial features from a commplex-valued spectrogram.

    Args:
        num_subbands: expected number of subbands in the input signal
        magnitude_reduction: averaging across channels. Default `None`, will calculate
                             magnitude of each channel.
        use_ipd: Use inter-channel phase difference (IPD). Currently not supported.
        mag_normalization: normalization for magnitude features
        ipd_normalization: normalization for IPD features
    """

    def __init__(
        self,
        num_subbands: int,
        mag_reduction: Optional[str] = 'rms',
        use_ipd: bool = False,
        mag_normalization: Optional[str] = None,
        ipd_normalization: Optional[str] = None,
    ):
        super().__init__()
        self.mag_reduction = mag_reduction
        self.use_ipd = use_ipd

        # TODO: normalization
        if mag_normalization is not None:
            raise NotImplementedError(f'Unknown magnitude normalization {mag_normalization}')
        self.mag_normalization = mag_normalization

        if ipd_normalization is not None:
            raise NotImplementedError(f'Unknown ipd normalization {ipd_normalization}')
        self.ipd_normalization = ipd_normalization

        if self.use_ipd:
            self.num_features = 2 * num_subbands
        else:
            self.num_features = num_subbands

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports.
        """
        return {
            "input": NeuralType(('B', 'D', 'T', 'C'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports.
        """
        return {
            "output": NeuralType(('B', 'D', 'T', 'C'), SpectrogramType()),
            "output_length": NeuralType(('B',), LengthsType()),
        }

    @typecheck()
    def forward(self, input: torch.Tensor, input_length: torch.Tensor) -> torch.Tensor:
        """Convert input batch of C-channel spectrograms into
        a batch of time-frequency features with dimension num_feat.
        The output number of channels may be the same as input, or
        reduced to 1, e.g., if averaging over magnitude and not appending individual IPDs.

        Args:
            input: Spectrogram for C channels with F subbands and N time frames, (B, F, N, C)
            input_length: Length of valid entries along the time dimension, shape (B,)

        Returns:
            num_feat features with num_feat_channels, shape (B, num_feat, N, num_feat_channels)
        """
        # Magnitude spectrum
        if self.mag_reduction is None:
            mag = torch.abs(input)
        elif self.mag_reduction == 'abs_mean':
            mag = torch.abs(torch.mean(input, axis=3, keepdim=True))
        elif self.mag_reduction == 'mean_abs':
            mag = torch.mean(torch.abs(input), axis=3, keepdim=True)
        elif self.mag_reduction == 'rms':
            mag = torch.sqrt(torch.mean(torch.abs(input) ** 2, axis=3, keepdim=True))
        else:
            raise ValueError(f'Unexpected magnitude reduction {self.mag_reduction}')

        if self.mag_normalization is not None:
            mag = self.mag_normalization(mag)

        features = mag

        if self.use_ipd:
            # Calculate IPD relative to average spec
            spec_mean = torch.mean(input, axis=3, keepdim=True)

            rtf = input / spec_mean
            ipd = torch.angle(rtf)

            if self.ipd_normalization is not None:
                ipd = self.ipd_normalization(ipd)

            # Concatenate to existing features
            features = torch.cat([features.expand(ipd.shape), ipd], axis=1)

        return features, input_length


@experimental
class MaskEstimatorRNN(NeuralModule):
    """Estimate `num_outputs` masks from the input spectrogram
    using stacked RNNs and projections.

    The module is structured as follows:
        input --> spatial features --> input projection -->
            --> stacked RNNs --> output projection for each output --> sigmoid

    Reference:
        Multi-microphone neural speech separation for far-field multi-talker
        speech recognition (https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8462081)

    Args:
        num_outputs: number of output masks to estiamte
        num_subbands: number of subbands of the input spectrogram
        num_features: number of features after the input projections
        num_layers: number of RNN layers
        num_hidden_features: number of hidden features in RNN layers
        dropout: dropout layer for RNN layers
        bidirectional: If `True`, use bidirectional RNNs
        rnn_type: `lstm` or `gru`, defaults to `lstm`
    """

    def __init__(
        self,
        num_outputs: int,
        num_subbands: int,
        num_features: int = 1024,
        num_layers: int = 3,
        num_hidden_features: Optional[int] = None,
        dropout: float = 0,
        bidirectional=True,
        rnn_type: str = 'lstm',
        mag_reduction: str = 'rms',
        use_ipd: bool = None,
    ):
        super().__init__()
        if num_hidden_features is None:
            num_hidden_features = num_features

        self.spatial_features = SpatialFeatures(
            num_subbands=num_subbands, mag_reduction=mag_reduction, use_ipd=use_ipd
        )

        self.input_projection = torch.nn.Linear(
            in_features=self.spatial_features.num_features, out_features=num_features
        )

        if rnn_type == 'lstm':
            self.rnn = torch.nn.LSTM(
                input_size=num_features,
                hidden_size=num_hidden_features,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        elif rnn_type == 'gru':
            self.rnn = torch.nn.GRU(
                input_size=num_features,
                hidden_size=num_hidden_features,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(f'Unknown rnn_type: {rnn_type}')

        # Each output shares the RNN and has a separate projection
        self.output_projections = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    in_features=2 * num_features if bidirectional else num_features, out_features=num_subbands
                )
                for _ in range(num_outputs)
            ]
        )
        self.output_nonlinearity = torch.nn.Sigmoid()

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports.
        """
        return {
            "input": NeuralType(('B', 'D', 'T', 'C'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports.
        """
        return {
            "output": NeuralType(('B', 'D', 'T', 'C'), FloatType()),
            "output_length": NeuralType(('B',), LengthsType()),
        }

    @typecheck()
    def forward(self, input: torch.Tensor, input_length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate `num_outputs` masks from the input spectrogram.

        Args:
            input: C-channel input, shape (B, F, N, C)
            input_length: Length of valid entries along the time dimension, shape (B,)

        Returns:
            Returns `num_outputs` masks in a tensor, shape (B, F, N, num_outputs),
            and output length with shape (B,)
        """
        input, _ = self.spatial_features(input=input, input_length=input_length)
        B, num_features, N, num_feature_channels = input.shape

        # (B, num_feat, N, num_feat_channels) -> (B, N, num_feat, num_feat_channels)
        input = input.transpose(1, 2)

        # (B, N, num_feat, num_feat_channels) -> (B, N, num_feat * num_feat_channels)
        input = input.view(B, N, -1)

        # Apply projection on num_feat
        input = self.input_projection(input)

        # Apply RNN on the input sequence
        input_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input, input_length.cpu(), batch_first=True, enforce_sorted=False
        ).to(input.device)
        self.rnn.flatten_parameters()
        input_packed, _ = self.rnn(input_packed)
        input, input_length = torch.nn.utils.rnn.pad_packed_sequence(input_packed, batch_first=True)
        input_length = input_length.to(input.device)

        # Create `num_outputs` masks
        output = []
        for output_projection in self.output_projections:
            # Output projection
            mask = output_projection(input)
            mask = self.output_nonlinearity(mask)

            # Back to the original format
            # (B, N, F) -> (B, F, N)
            mask = mask.transpose(2, 1)

            # Append to the output
            output.append(mask)

        # Stack along channel dimension to get (B, F, N, M)
        output = torch.stack(output, axis=-1)

        # Mask frames beyond input length
        length_mask: torch.Tensor = make_seq_mask_like(
            lengths=input_length, like=output, time_dim=-2, valid_ones=False
        )
        output = output.masked_fill(length_mask, 0.0)

        return output, input_length


@experimental
class MaskReferenceChannel(NeuralModule):
    """A simple mask processor which applies mask
    on ref_channel of the input signal.

    Args:
        ref_channel: Index of the reference channel.
        mask_min_db: Threshold mask to a minimal value before applying it, defaults to -200dB
        mask_max_db: Threshold mask to a maximal value before applying it, defaults to 0dB
    """

    def __init__(self, ref_channel: int = 0, mask_min_db: float = -200, mask_max_db: float = 0):
        super().__init__()
        self.ref_channel = ref_channel
        # Mask thresholding
        self.mask_min = db2mag(mask_min_db)
        self.mask_max = db2mag(mask_max_db)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports.
        """
        return {
            "input": NeuralType(('B', 'D', 'T', 'C'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType()),
            "mask": NeuralType(('B', 'D', 'T', 'C'), FloatType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports.
        """
        return {
            "output": NeuralType(('B', 'D', 'T', 'C'), SpectrogramType()),
            "output_length": NeuralType(('B',), LengthsType()),
        }

    @typecheck()
    def forward(
        self, input: torch.Tensor, input_length: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mask on `ref_channel` of the input signal.
        This can be used to generate multi-channel output.
        If `mask` has `M` channels, the output will have `M` channels as well.

        Args:
            input: Input signal complex-valued spectrogram, shape (B, F, N, C)
            input_length: Length of valid entries along the time dimension, shape (B,)
            mask: Mask for M outputs, shape (B, F, N, M)

        Returns:
            M-channel output complex-valed spectrogram with shape (B, F, N, M)
        """
        num_outputs = mask.shape[-1]

        # Apply thresholds
        mask = torch.clamp(mask, min=self.mask_min, max=self.mask_max)

        # Apply each output mask on the ref channel
        output = torch.stack([mask[..., m] * input[..., self.ref_channel] for m in range(num_outputs)], axis=-1)
        return output, input_length


@experimental
class MaskBasedBeamformer(NeuralModule):
    """Multi-channel processor using masks to estimate signal statistics.

    Args:
        filter_type: string denoting the type of the filter. Defaults to `mvdr`
        ref_channel: reference channel for processing
        mask_min_db: Threshold mask to a minimal value before applying it, defaults to -200dB
        mask_max_db: Threshold mask to a maximal value before applying it, defaults to 0dB
    """

    def __init__(
        self,
        filter_type: str = 'mvdr_souden',
        ref_channel: int = 0,
        mask_min_db: float = -200,
        mask_max_db: float = 0,
    ):
        if not HAVE_TORCHAUDIO:
            logging.error('Could not import torchaudio. Some features might not work.')

            raise ModuleNotFoundError(
                "torchaudio is not installed but is necessary to instantiate a {self.__class__.__name__}"
            )

        super().__init__()
        self.ref_channel = ref_channel
        self.filter_type = filter_type
        if self.filter_type == 'mvdr_souden':
            self.psd = torchaudio.transforms.PSD()
            self.filter = torchaudio.transforms.SoudenMVDR()
        else:
            raise ValueError(f'Unknown filter type {filter_type}')
        # Mask thresholding
        self.mask_min = db2mag(mask_min_db)
        self.mask_max = db2mag(mask_max_db)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports.
        """
        return {
            "input": NeuralType(('B', 'D', 'T', 'C'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType()),
            "mask": NeuralType(('B', 'D', 'T', 'C'), FloatType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports.
        """
        return {
            "output": NeuralType(('B', 'D', 'T', 'C'), SpectrogramType()),
            "output_length": NeuralType(('B',), LengthsType()),
        }

    @typecheck()
    def forward(self, input: torch.Tensor, input_length: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply a mask-based beamformer to the input spectrogram.
        This can be used to generate multi-channel output.
        If `mask` has `M` channels, the output will have `M` channels as well.

        Args:
            input: Input signal complex-valued spectrogram, shape (B, F, N, C)
            input_length: Length of valid entries along the time dimension, shape (B,)
            mask: Mask for M output signals, shape (B, F, N, M)
        
        Returns:
            M-channel output signal complex-valued spectrogram, shape (B, F, N, M)
        """
        num_outputs = mask.shape[-1]

        # Permute to TorchAudio format
        # (B, F, N, C) -> (B, C, F, N)
        input = input.permute(0, 3, 1, 2)
        # (B, F, N, M) -> (B, M, F, N)
        mask = mask.permute(0, 3, 1, 2)
        # Apply threshold on the mask
        mask = torch.clamp(mask, min=self.mask_min, max=self.mask_max)
        # Length mask
        length_mask: torch.Tensor = make_seq_mask_like(
            lengths=input_length, like=mask[:, 0, ...], time_dim=-1, valid_ones=False
        )
        # Use each mask to generate an output at ref_channel
        output = []
        for m in range(num_outputs):
            # Prepare mask for the desired and the undesired signal
            mask_desired = mask[:, m, ...].masked_fill(length_mask, 0.0)
            mask_undesired = (1 - mask_desired).masked_fill(length_mask, 0.0)
            # Calculate PSDs
            psd_desired = self.psd(input, mask_desired)
            psd_undesired = self.psd(input, mask_undesired)
            # Apply filter
            output_m = self.filter(input, psd_desired, psd_undesired, reference_channel=self.ref_channel)
            output_m = output_m.masked_fill(length_mask, 0.0)
            # Save the current output (B, F, N)
            output.append(output_m)

        output = torch.stack(output, axis=-1)

        return output, input_length
