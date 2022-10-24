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

from typing import Dict, Optional

import torch

from nemo.collections.asr.parts.utils.audio_utils import db2mag
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import AudioSignal, FloatType, NeuralType, SpectrogramType
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
                "torchaudio is not installed but is necessary for "
                "AudioToSpectrogram module. We recommend you try "
                "building it from source for the PyTorch version you have."
            )

        super().__init__()
        # For now, assume FFT length is divisible by two
        if fft_length % 2 != 0:
            raise ValueError(f'fft_length = {fft_length} must be divisible by 2')
        self.stft = torchaudio.transforms.Spectrogram(n_fft=fft_length, hop_length=hop_length, power=power)
        # number of subbands
        self.F = fft_length // 2 + 1

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports.
        """
        return {
            "input": NeuralType(('B', 'T', 'C'), AudioSignal()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports.
        """
        return {
            "output": NeuralType(('B', 'D', 'T', 'C'), SpectrogramType()),
        }

    @typecheck()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Convert a batch of C-channel input signals
        into a batch of complex-valued spectrograms.

        Args:
            input: Time-domain input signal with C channels, shape (B, T, C)

        Returns:
            Output spectrogram with F subbands and N time frames, shape (B, F, N, C)
        """
        B, T = input.shape[0:2]
        input = input.view(B, T, -1)

        # Permute to (B, C, T)
        input = input.permute(0, 2, 1)
        # STFT output (B, C, F, N)
        output = self.stft(input)
        # Permute channels to back
        output = output.permute(0, 2, 3, 1)

        return output


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
                "torchaudio is not installed but is necessary for "
                "SpectrogramToAudio module. We recommend you try "
                "building it from source for the PyTorch version you have."
            )

        super().__init__()
        # For now, assume FFT length is divisible by two
        if fft_length % 2 != 0:
            raise ValueError(f'fft_length = {fft_length} must be divisible by 2')
        self.istft = torchaudio.transforms.InverseSpectrogram(n_fft=fft_length, hop_length=hop_length)
        self.F = fft_length // 2 + 1

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports.
        """
        return {
            "input": NeuralType(('B', 'D', 'T', 'C'), SpectrogramType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports.
        """
        return {
            "output": NeuralType(('B', 'T', 'C'), AudioSignal()),
        }

    @typecheck()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ TODO

        Args:
            input: Input spectrogram for C channels, shape (B, F, N, C)

        Returns:
            Time-domain signal with T time-domain samples and C channels, (B, T, C)
        """
        B, F, N = input.shape[0:3]
        assert F == self.F, f'Number of subbands F={F} not matching self.F={self.F}'
        input = input.view(B, F, N, -1)

        # Transform to (B, C, F, N)
        input = input.permute(0, 3, 1, 2)
        # STFT output (B, C, T)
        output = self.istft(input)
        # Push channels to back
        output = output.permute(0, 2, 1)

        return output


@experimental
class SpatialFeatures(NeuralModule):
    """Generate spatial features from a commplex-valued spectrogram.

    Args:
        num_subbands: expected number of subbands in the input signal
        magnitude_reduction: averaging across channels. Default `None`, will calculate
            magnitude of each channel.
        use_ipd: Use inter-channel phase difference (IPD). Currently not supported.
        normalize: perform feature normalization.
    """

    def __init__(
        self,
        num_subbands: int,
        magnitude_reduction: Optional[str] = 'rms',
        use_ipd: bool = False,
        normalize: bool = True,
    ):
        super().__init__()
        self.magnitude_reduction = magnitude_reduction
        self.use_ipd = use_ipd
        self.normalize = normalize

        if self.normalize:
            # Mean and variance normalization for magnitude features
            self.magnitude_normalization = torch.nn.InstanceNorm2d(num_features=num_subbands)
            # TODO: mean-only normalization for IPD

        if self.use_ipd:
            raise NotImplementedError(f'Inter-mic phase difference not implemented.')

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports.
        """
        return {
            "input": NeuralType(('B', 'D', 'T', 'C'), SpectrogramType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports.
        """
        return {
            "output": NeuralType(('B', 'D', 'T', 'C'), SpectrogramType()),
        }

    @typecheck()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Convert input batch of C-channel spectrograms into
        a batch of time-frequency features with dimension F_out.
        The output number of channels may be the same as input, or
        reduced to 1, e.g., if averaging over magnitude and not appending IPDs.

        Args:
            input: Spectrogram for C channels with F subbands and N time frames, (B, F, N, C)

        Returns:
            F_out features with M channels, shape (B, F_out, N, M)
        """
        # Magnitude spectrum
        if self.magnitude_reduction is None:
            magnitude = torch.abs(input)
        elif self.magnitude_reduction == 'abs_mean':
            magnitude = torch.abs(torch.mean(input, axis=3, keepdim=True))
        elif self.magnitude_reduction == 'mean_abs':
            magnitude = torch.mean(torch.abs(input), axis=3, keepdim=True)
        elif self.magnitude_reduction == 'rms':
            magnitude = torch.sqrt(torch.mean(torch.abs(input) ** 2, axis=3, keepdim=True))
        else:
            raise ValueError(f'Unexpected magnitude reduction {self.magnitude_reduction}')

        if self.normalize:
            magnitude = self.magnitude_normalization(magnitude)

        return magnitude


@experimental
class MaskEstimatorRNN(NeuralModule):
    """Estimate `num_outputs` mask using stacked RNNs
    from the input spectrogram.

    The module is structured as follows:
        input --> spatial features --> input projection -->
            --> stacked RNNs --> output projection for each output --> sigmoid

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
        num_features: int,
        num_layers: int,
        num_hidden_features: Optional[int] = None,
        dropout: float = 0,
        bidirectional=True,
        rnn_type: str = 'lstm',
    ):
        super().__init__()
        if num_hidden_features is None:
            num_hidden_features = num_features
        # For now, take input RMS over channels
        self.spatial_features = SpatialFeatures(num_subbands=num_subbands, magnitude_reduction='rms')
        self.input_projection = torch.nn.Linear(in_features=num_subbands, out_features=num_features)

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
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports.
        """
        return {
            "output": NeuralType(('B', 'D', 'T', 'C'), FloatType()),
        }

    @typecheck()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Estimate `num_outputs` masks from the input spectrogram.
        TODO: support input_length, e.g., using packed_sequence

        Args:
            input: C-channel input, shape (B, F, N, C)

        Returns:
            `num_outputs` masks in a tensor, shape (B, F, N, num_outputs)
        """
        input = self.spatial_features(input=input)
        input = torch.squeeze(input, dim=3)
        # (B, F, N) -> (B, N, F)
        input = input.transpose(2, 1)
        # Apply projection on F
        input = self.input_projection(input)
        # Apply RNN on the input sequence
        input, _ = self.rnn(input)
        # Calculate masks
        output = []
        for output_projection in self.output_projections:
            mask = output_projection(input)
            mask = self.output_nonlinearity(mask)
            # Back to the original format
            # (B, N, F) -> (B, F, N)
            mask = mask.transpose(2, 1)
            # Append to the output
            output.append(mask)

        # Stack along channel dimension to get (B, F, N, M)
        output = torch.stack(output, axis=-1)

        return output


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
            "mask": NeuralType(('B', 'D', 'T', 'C'), FloatType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports.
        """
        return {
            "output": NeuralType(('B', 'D', 'T', 'C'), SpectrogramType()),
        }

    @typecheck()
    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply mask on `ref_channel` of the input signal.
        This can be used to generate multi-channel output.
        If `mask` has `M` channels, the output will have `M` channels as well.

        Args:
            input: Input signal complex-valued spectrogram, shape (B, F, N, C)
            mask: Mask for M outputs, shape (B, F, N, M)

        Returns:
            M-channel output complex-valed spectrogram with shape (B, F, N, M)
        """
        num_outputs = mask.shape[-1]
        # Apply thresholds
        mask = torch.clamp(mask, min=self.mask_min, max=self.mask_max)
        # Apply each output mask on the ref channel
        output = torch.stack([mask[..., m] * input[..., self.ref_channel] for m in range(num_outputs)], axis=-1)
        return output


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
        self, filter_type: str = 'mvdr', ref_channel: int = 0, mask_min_db: float = -200, mask_max_db: float = 0
    ):
        if not HAVE_TORCHAUDIO:
            logging.error('Could not import torchaudio. Some features might not work.')

            raise ModuleNotFoundError(
                "torchaudio is not installed but is necessary for "
                "MaskBasedBeamformer module. We recommend you try "
                "building it from source for the PyTorch version you have."
            )

        super().__init__()
        self.ref_channel = ref_channel
        self.filter_type = filter_type
        if self.filter_type == 'mvdr':
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
            "mask": NeuralType(('B', 'D', 'T', 'C'), FloatType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports.
        """
        return {
            "output": NeuralType(('B', 'D', 'T', 'C'), SpectrogramType()),
        }

    @typecheck()
    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply a mask-based beamformer to the input spectrogram.
        This can be used to generate multi-channel output.
        If `mask` has `M` channels, the output will have `M` channels as well.

        Args:
            input: Input signal complex-valued spectrogram, shape (B, F, N, C)
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
        # Apply threshold
        mask = torch.clamp(mask, min=self.mask_min, max=self.mask_max)
        # Apply each output mask on the ref channel
        output = []
        for m in range(num_outputs):
            # Prepare mask for the desired and the undesired signal
            mask_desired = mask[:, m, ...]
            mask_undesired = 1 - mask_desired
            # Calculate PSDs
            psd_desired = self.psd(input, mask_desired)
            psd_undesired = self.psd(input, mask_undesired)
            # Apply filter
            output_m = self.filter(input, psd_desired, psd_undesired, reference_channel=self.ref_channel)
            # Save the current output (B, F, N)
            output.append(output_m)

        return torch.stack(output, axis=-1)
