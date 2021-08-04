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

from typing import Optional

import torch
from packaging import version

from nemo.utils import logging

# Library version globals
TORCH_VERSION = None
TORCH_VERSION_MIN = version.Version('1.7')


def stft_patch(
    input: torch.Tensor,
    n_fft: int,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Optional[torch.Tensor] = None,
    center: bool = True,
    pad_mode: str = 'reflect',
    normalized: bool = False,
    onesided: Optional[bool] = None,
    return_complex: Optional[bool] = None,
):
    """
    Patch over torch.stft for PyTorch <= 1.6.
    Arguments are same as torch.stft().

    # TODO: Remove once PyTorch 1.7+ is a requirement.
    """
    global TORCH_VERSION
    if TORCH_VERSION is None:
        TORCH_VERSION = version.parse(torch.__version__)

        logging.warning(
            "torch.stft() signature has been updated for PyTorch 1.7+\n"
            "Please update PyTorch to remain compatible with later versions of NeMo."
        )

    if TORCH_VERSION < TORCH_VERSION_MIN:
        return torch.stft(
            input=input,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            normalized=normalized,
            onesided=True,
        )
    else:
        return torch.stft(
            input=input,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            normalized=normalized,
            onesided=onesided,
            return_complex=return_complex,
        )


def istft_patch(
    input: torch.Tensor,
    n_fft: int,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Optional[torch.Tensor] = None,
    center: bool = True,
    normalized: bool = False,
    onesided: Optional[bool] = None,
    length: int = None,
    return_complex: Optional[bool] = False,
):
    """
    Patch over torch.stft for PyTorch <= 1.6.
    Arguments are same as torch.stft().

    # TODO: Remove once PyTorch 1.7+ is a requirement.
    """
    global TORCH_VERSION
    if TORCH_VERSION is None:
        TORCH_VERSION = version.parse(torch.__version__)

        logging.warning(
            "torch.stft() signature has been updated for PyTorch 1.7+\n"
            "Please update PyTorch to remain compatible with later versions of NeMo."
        )

    if TORCH_VERSION < TORCH_VERSION_MIN:
        return torch.istft(
            input=input,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            length=length,
            normalized=normalized,
            onesided=True,
        )
    else:
        return torch.istft(
            input=input,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            length=length,
            normalized=normalized,
            onesided=onesided,
            return_complex=return_complex,
        )
