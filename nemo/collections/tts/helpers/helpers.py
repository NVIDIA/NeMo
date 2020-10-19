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

from enum import Enum
from typing import Dict, Sequence

import librosa
import matplotlib.pylab as plt
import numpy as np
import torch
from numpy import ndarray
from pesq import pesq
from pystoi import stoi
from pytorch_lightning.utilities import rank_zero_only

from nemo.utils import logging


class OperationMode(Enum):
    """Training or Inference (Evaluation) mode"""

    training = 0
    validation = 1
    infer = 2


def get_mask_from_lengths(lengths, max_len=None):
    if not max_len:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=torch.long)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def griffin_lim(magnitudes, n_iters=50, n_fft=1024):
    """
    Griffin-Lim algorithm to convert magnitude spectrograms to audio signals
    """
    phase = np.exp(2j * np.pi * np.random.rand(*magnitudes.shape))
    complex_spec = magnitudes * phase
    signal = librosa.istft(complex_spec)
    if not np.isfinite(signal).all():
        logging.warning("audio was not finite, skipping audio saving")
        return np.array([0])

    for _ in range(n_iters):
        _, phase = librosa.magphase(librosa.stft(signal, n_fft=n_fft))
        complex_spec = magnitudes * phase
        signal = librosa.istft(complex_spec)
    return signal


@rank_zero_only
def log_audio_to_tb(
    swriter,
    spect,
    name,
    step,
    griffin_lim_mag_scale=1024,
    griffin_lim_power=1.2,
    sr=22050,
    n_fft=1024,
    n_mels=80,
    fmax=8000,
):
    filterbank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=fmax)
    log_mel = spect.data.cpu().numpy().T
    mel = np.exp(log_mel)
    magnitude = np.dot(mel, filterbank) * griffin_lim_mag_scale
    audio = griffin_lim(magnitude.T ** griffin_lim_power)
    swriter.add_audio(name, audio / max(np.abs(audio)), step, sample_rate=sr)


@rank_zero_only
def tacotron2_log_to_tb_func(
    swriter,
    tensors,
    step,
    tag="train",
    log_images=False,
    log_images_freq=1,
    add_audio=True,
    griffin_lim_mag_scale=1024,
    griffin_lim_power=1.2,
    sr=22050,
    n_fft=1024,
    n_mels=80,
    fmax=8000,
):
    _, spec_target, mel_postnet, gate, gate_target, alignments = tensors
    if log_images and step % log_images_freq == 0:
        swriter.add_image(
            f"{tag}_alignment", plot_alignment_to_numpy(alignments[0].data.cpu().numpy().T), step, dataformats="HWC",
        )
        swriter.add_image(
            f"{tag}_mel_target", plot_spectrogram_to_numpy(spec_target[0].data.cpu().numpy()), step, dataformats="HWC",
        )
        swriter.add_image(
            f"{tag}_mel_predicted",
            plot_spectrogram_to_numpy(mel_postnet[0].data.cpu().numpy()),
            step,
            dataformats="HWC",
        )
        swriter.add_image(
            f"{tag}_gate",
            plot_gate_outputs_to_numpy(gate_target[0].data.cpu().numpy(), torch.sigmoid(gate[0]).data.cpu().numpy(),),
            step,
            dataformats="HWC",
        )
        if add_audio:
            filterbank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=fmax)
            log_mel = mel_postnet[0].data.cpu().numpy().T
            mel = np.exp(log_mel)
            magnitude = np.dot(mel, filterbank) * griffin_lim_mag_scale
            audio = griffin_lim(magnitude.T ** griffin_lim_power)
            swriter.add_audio(f"audio/{tag}_predicted", audio / max(np.abs(audio)), step, sample_rate=sr)

            log_mel = spec_target[0].data.cpu().numpy().T
            mel = np.exp(log_mel)
            magnitude = np.dot(mel, filterbank) * griffin_lim_mag_scale
            audio = griffin_lim(magnitude.T ** griffin_lim_power)
            swriter.add_audio(f"audio/{tag}_target", audio / max(np.abs(audio)), step, sample_rate=sr)


def plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower', interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_spectrogram_to_numpy(spectrogram):
    spectrogram = spectrogram.astype(np.float32)
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_gate_outputs_to_numpy(gate_targets, gate_outputs):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(
        range(len(gate_targets)), gate_targets, alpha=0.5, color='green', marker='+', s=1, label='target',
    )
    ax.scatter(
        range(len(gate_outputs)), gate_outputs, alpha=0.5, color='red', marker='.', s=1, label='predicted',
    )

    plt.xlabel("Frames (Green target, Red predicted)")
    plt.ylabel("Gate State")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


@rank_zero_only
def waveglow_log_to_tb_func(
    swriter, tensors, step, tag="train", n_fft=1024, hop_length=256, window="hann", mel_fb=None,
):
    _, audio_pred, spec_target, mel_length = tensors
    mel_length = mel_length[0]
    spec_target = spec_target[0].data.cpu().numpy()[:, :mel_length]
    swriter.add_image(
        f"{tag}_mel_target", plot_spectrogram_to_numpy(spec_target), step, dataformats="HWC",
    )
    if mel_fb is not None:
        mag, _ = librosa.core.magphase(
            librosa.core.stft(
                np.nan_to_num(audio_pred[0].cpu().detach().numpy()), n_fft=n_fft, hop_length=hop_length, window=window,
            )
        )
        mel_pred = np.matmul(mel_fb.cpu().numpy(), mag).squeeze()
        log_mel_pred = np.log(np.clip(mel_pred, a_min=1e-5, a_max=None))
        swriter.add_image(
            f"{tag}_mel_predicted", plot_spectrogram_to_numpy(log_mel_pred[:, :mel_length]), step, dataformats="HWC",
        )


def remove(conv_list):
    new_conv_list = torch.nn.ModuleList()
    for old_conv in conv_list:
        old_conv = torch.nn.utils.remove_weight_norm(old_conv)
        new_conv_list.append(old_conv)
    return new_conv_list


def eval_tts_scores(
    y_clean: ndarray, y_est: ndarray, T_ys: Sequence[int] = (0,), sampling_rate=22050
) -> Dict[str, float]:
    """
    calculate metric using EvalModule. y can be a batch.
    Args:
        y_clean: real audio
        y_est: estimated audio
        T_ys: length of the non-zero parts of the histograms
        sampling_rate: The used Sampling rate.

    Returns:
        A dictionary mapping scoring systems (string) to numerical scores.
        1st entry: 'STOI'
        2nd entry: 'PESQ'
    """

    if y_clean.ndim == 1:
        y_clean = y_clean[np.newaxis, ...]
        y_est = y_est[np.newaxis, ...]
    if T_ys == (0,):
        T_ys = (y_clean.shape[1],) * y_clean.shape[0]

    clean = y_clean[0, : T_ys[0]]
    estimated = y_est[0, : T_ys[0]]
    stoi_score = stoi(clean, estimated, sampling_rate, extended=False)
    pesq_score = pesq(16000, np.asarray(clean), estimated, 'wb')
    ## fs was set 16,000, as pesq lib doesnt currently support felxible fs.

    return {'STOI': stoi_score, 'PESQ': pesq_score}
