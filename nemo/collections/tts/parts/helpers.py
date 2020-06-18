# Copyright (c) 2019 NVIDIA Corporation
import librosa
import matplotlib.pylab as plt
import numpy as np
import torch

from nemo.utils import logging

__all__ = [
    "waveglow_log_to_tb_func",
    "waveglow_process_eval_batch",
    "waveglow_eval_log_to_tb_func",
    "tacotron2_log_to_tb_func",
    "tacotron2_process_eval_batch",
    "tacotron2_process_final_eval",
    "tacotron2_eval_log_to_tb_func",
]


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


def waveglow_log_to_tb_func(
    swriter,
    tensors,
    step,
    tag="train",
    log_images=False,
    log_images_freq=1,
    n_fft=1024,
    hop_length=256,
    window="hann",
    mel_fb=None,
):
    loss, audio_pred, spec_target, mel_length = tensors
    if loss:
        swriter.add_scalar("loss", loss, step)
    if log_images and step % log_images_freq == 0:
        mel_length = mel_length[0]
        spec_target = spec_target[0].data.cpu().numpy()[:, :mel_length]
        swriter.add_image(
            f"{tag}_mel_target", plot_spectrogram_to_numpy(spec_target), step, dataformats="HWC",
        )
        if mel_fb is not None:
            mag, _ = librosa.core.magphase(
                librosa.core.stft(
                    np.nan_to_num(audio_pred[0].cpu().detach().numpy()),
                    n_fft=n_fft,
                    hop_length=hop_length,
                    window=window,
                )
            )
            mel_pred = np.matmul(mel_fb.cpu().numpy(), mag).squeeze()
            log_mel_pred = np.log(np.clip(mel_pred, a_min=1e-5, a_max=None))
            swriter.add_image(
                f"{tag}_mel_predicted",
                plot_spectrogram_to_numpy(log_mel_pred[:, :mel_length]),
                step,
                dataformats="HWC",
            )


def waveglow_process_eval_batch(tensors: dict, global_vars: dict):
    if 'tensorboard' not in global_vars.keys():
        global_vars['tensorboard'] = {}
        for k, v in tensors.items():
            if k.startswith("processed_signal"):
                global_vars['tensorboard']['mel_target'] = v[0]
            if k.startswith("audio"):
                global_vars['tensorboard']['audio_pred'] = v[0]
            if k.startswith("processed_length"):
                global_vars['tensorboard']['mel_length'] = v[0]


def waveglow_eval_log_to_tb_func(
    swriter, global_vars, step, tag=None, n_fft=1024, hop_length=256, window="hann", mel_fb=None,
):
    spec_target = global_vars['tensorboard']["mel_target"]
    audio_pred = global_vars['tensorboard']["audio_pred"]
    mel_length = global_vars['tensorboard']['mel_length']
    waveglow_log_to_tb_func(
        swriter,
        [None, audio_pred, spec_target, mel_length],
        step,
        tag=tag,
        log_images=True,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        mel_fb=mel_fb,
    )


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
    loss, spec_target, mel_postnet, gate, gate_target, alignments = tensors
    if loss:
        swriter.add_scalar("loss", loss, step)
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


def tacotron2_process_eval_batch(tensors: dict, global_vars: dict):
    if 'EvalLoss' not in global_vars.keys():
        global_vars['EvalLoss'] = []
    if 'tensorboard' not in global_vars.keys():
        global_vars['tensorboard'] = {}
        for k, v in tensors.items():
            if k.startswith("processed_signal"):
                global_vars['tensorboard']['mel_target'] = v[0]
            if k.startswith("mel_output"):
                global_vars['tensorboard']['mel_pred'] = v[0]
            if k.startswith("gate_output"):
                global_vars['tensorboard']['gate'] = v[0]
            if k.startswith("alignments"):
                global_vars['tensorboard']['alignments'] = v[0]
            if k.startswith("gate_target"):
                global_vars['tensorboard']['gate_target'] = v[0]

    for k in tensors.keys():
        if k.startswith("loss"):
            loss_key = k
    global_vars['EvalLoss'].append(torch.mean(torch.stack(tensors[loss_key])))


def tacotron2_process_final_eval(global_vars: dict, tag=None):
    eloss = torch.mean(torch.stack(global_vars['EvalLoss'])).item()
    global_vars['EvalLoss'] = eloss
    logging.info(f"==========>>>>>>Evaluation Loss {tag}: {eloss}")
    return global_vars


def tacotron2_eval_log_to_tb_func(
    swriter,
    global_vars,
    step,
    tag=None,
    add_audio=True,
    griffin_lim_mag_scale=1024,
    griffin_lim_power=1.2,
    sr=22050,
    n_fft=1024,
    n_mels=80,
    fmax=8000,
):
    spec_target = global_vars['tensorboard']["mel_target"]
    mel_postnet = global_vars['tensorboard']["mel_pred"]
    gate = global_vars['tensorboard']["gate"]
    gate_target = global_vars['tensorboard']["gate_target"]
    alignments = global_vars['tensorboard']["alignments"]
    swriter.add_scalar(f"{tag}.loss", global_vars['EvalLoss'], step)
    tacotron2_log_to_tb_func(
        swriter,
        [None, spec_target, mel_postnet, gate, gate_target, alignments],
        step,
        tag=tag,
        log_images=True,
        add_audio=add_audio,
        griffin_lim_mag_scale=griffin_lim_mag_scale,
        griffin_lim_power=griffin_lim_power,
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmax=fmax,
    )


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


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
