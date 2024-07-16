# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from nemo.utils import logging

try:
    import torchaudio

    SQUIM_MOS_MODEL = torchaudio.pipelines.SQUIM_SUBJECTIVE.get_model()
    SQUIM_OBJECTIVE_MODEL = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model()

    HAVE_TORCHAUDIO = True
except ModuleNotFoundError:
    HAVE_TORCHAUDIO = False


def calculate_squim_mos(preds: torch.Tensor, target: torch.Tensor, fs: int) -> torch.Tensor:
    """Calculate Torchaudio Squim-MOS.

    Args:
        preds: tensor with predictions, shape ``(B, T)``
        target: tensor with target signals, shape ``(B, T)``. Target can be a non-matching reference.
        fs: sampling rate of the signals

    Returns:
        Float tensor with shape ``(B,)`` of MOS values per sample
    """
    if not HAVE_TORCHAUDIO:
        raise ModuleNotFoundError("Squim MOS calculation requires that `torchaudio` with SQUIM is installed.")

    if fs != 16000:
        logging.debug('Resampling predictions and targets from %dHz to 16kHz', fs)
        # Resample to 16 kHz using kaiser_best
        preds = torchaudio.functional.resample(
            preds,
            orig_freq=fs,
            new_freq=16000,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method='sinc_interp_kaiser',
            beta=14.769656459379492,
        )

        target = torchaudio.functional.resample(
            target,
            orig_freq=fs,
            new_freq=16000,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method='sinc_interp_kaiser',
            beta=14.769656459379492,
        )
        logging.debug('Resampling done')

    squim_mos_model = SQUIM_MOS_MODEL.to(preds.device)

    # Calculate MOS
    mos_batch = squim_mos_model(preds, target)
    return mos_batch


def calculate_squim_objective(preds: torch.Tensor, fs: int, metric: str) -> torch.Tensor:
    """Calculate Torchaudio Squim scores approximating reference metrics
    without using reference signal.

    Args:
        preds: tensor with predictions, shape ``(B, T)``
        fs: sampling rate of the signals

    Returns:
        Float tensor with shape ``(B,3)`` of estimated (stoi, pesq, si_sdr) values per sample.
    """
    if not HAVE_TORCHAUDIO:
        raise ModuleNotFoundError("Squim calculation requires that `torchaudio` with SQUIM is installed.")

    if metric not in ['stoi', 'pesq', 'si_sdr']:
        raise ValueError(f'Unknown metric {metric}')

    if fs != 16000:
        logging.debug('Resampling predictions from %dHz to 16kHz', fs)
        # Resample to 16 kHz using kaiser_best
        preds = torchaudio.functional.resample(
            preds,
            orig_freq=fs,
            new_freq=16000,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method='sinc_interp_kaiser',
            beta=14.769656459379492,
        )
        logging.debug('Resampling done')

    squim_objective_model = SQUIM_OBJECTIVE_MODEL.to(preds.device)

    # Calculate MOS
    stoi_batch, pesq_batch, si_sdr_batch = squim_objective_model(preds)

    if metric == 'stoi':
        return stoi_batch
    elif metric == 'pesq':
        return pesq_batch
    elif metric == 'si_sdr':
        return si_sdr_batch
    else:
        raise ValueError(f'Unknown metric {metric}')
