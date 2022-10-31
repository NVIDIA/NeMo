from typing import Dict, List, Optional

import IPython.display as ipd
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nemo.collections.asr.parts.utils.vad_utils import (
    gen_pred_from_speech_segments,
    generate_vad_segment_table_per_tensor,
    load_tensor_from_file,
    prepare_gen_segment_table,
)


def extract_labels(path2ground_truth_label: str, time: list) -> list:
    """
    Extract ground-truth label for given time period.
    path2ground_truth_label (str): path of groundtruth label file
    time (list) : a list of array representing time period.
    """

    data = pd.read_csv(path2ground_truth_label, sep=" ", delimiter=None, header=None)
    if path2ground_truth_label.endswith(".rttm"):
        data = data.rename(columns={3: "start", 4: "dur", 7: "speaker"})
    else:
        data = data.rename(columns={0: "start", 1: "dur", 2: "speaker"})

    labels = []
    for pos in time:
        line = data[(data["start"] <= pos) & (data["start"] + data["dur"] > pos)]
        if len(line) >= 1:
            labels.append(1)
        else:
            labels.append(0)
    return labels


def plot_sample(
    path2audio_file: str,
    path2_vad_pred: str,
    path2ground_truth_label: str = None,
    offset: float = 0,
    duration: float = None,
    threshold: float = None,
    per_args: dict = None,
    save_path: str = '',
) -> ipd.Audio:
    """
    Plot VAD outputs for demonstration in tutorial
    Args:
        path2audio_file (str):  path to audio file.
        path2_vad_pred (str): path to vad prediction file,
        path2ground_truth_label(str): path to groundtruth label file.
        threshold (float): threshold for prediction score (from 0 to 1).
        per_args(dict): a dict that stores the thresholds for postprocessing.
    """
    plt.figure(figsize=[20, 2])
    UNIT_FRAME_LEN = 0.01

    audio, sample_rate = librosa.load(path=path2audio_file, sr=16000, mono=True, offset=offset, duration=duration)
    dur = librosa.get_duration(y=audio, sr=sample_rate)

    time = np.arange(offset, offset + dur, UNIT_FRAME_LEN)
    frame, _ = load_tensor_from_file(path2_vad_pred)

    frame_snippet = frame[int(offset / UNIT_FRAME_LEN) : int((offset + dur) / UNIT_FRAME_LEN)]

    len_pred = len(frame_snippet)
    ax1 = plt.subplot()
    ax1.set_title(path2audio_file)
    ax1.plot(np.arange(audio.size) / sample_rate, audio, 'gray')
    ax1.set_xlim([0, int(dur) + 1])
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylabel('Signal')
    ax1.set_ylim([-1, 1])
    ax2 = ax1.twinx()

    if threshold and per_args:
        raise ValueError("threshold and per_args cannot be used at same time!")
    if not threshold and not per_args:
        raise ValueError("One and only one of threshold and per_args must have been used!")

    if threshold:
        pred_snippet = np.where(frame_snippet >= threshold, 1, 0)
    if per_args:
        _, per_args_float = prepare_gen_segment_table(
            frame, per_args
        )  # take whole frame here for calculating onset and offset
        speech_segments = generate_vad_segment_table_per_tensor(frame, per_args_float)
        pred = gen_pred_from_speech_segments(speech_segments, frame)
        pred_snippet = pred[int(offset / UNIT_FRAME_LEN) : int((offset + dur) / UNIT_FRAME_LEN)]

    if path2ground_truth_label:
        # label = extract_labels(path2ground_truth_label, time)
        label, _ = load_tensor_from_file(path2ground_truth_label)
        label = label[:len_pred]
        ax2.plot(np.arange(len_pred) * UNIT_FRAME_LEN, label, 'r', label='label')

    ax2.plot(np.arange(len_pred) * UNIT_FRAME_LEN, pred_snippet, 'b', label='pred')
    ax2.plot(np.arange(len_pred) * UNIT_FRAME_LEN, frame_snippet, 'g--', label='speech prob')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='lower right', shadow=True)
    ax2.set_ylabel('Preds and Probas')
    ax2.set_ylim([-0.1, 1.1])
    plt.show()
    if save_path != "":
        plt.savefig(save_path)
    return ipd.Audio(audio, rate=16000)


def plot_sample_from_manifest(data: Dict, max_duration: Optional[float] = None, repeat: int = 1, save_path: str = ""):
    audio_file = data['audio_filepath']
    labels = [float(x) for x in data['label'].split()]
    duration = data['duration']
    offset = data['offset']
    if repeat > 1:
        labels = np.repeat(labels, repeat)
    if max_duration:
        duration = min(duration, max_duration)

    plt.figure(figsize=[20, 2])
    UNIT_FRAME_LEN = 0.01

    audio, sample_rate = librosa.load(path=audio_file, sr=16000, mono=True, offset=offset, duration=duration)
    dur = librosa.get_duration(y=audio, sr=sample_rate)

    length = len(labels)
    ax1 = plt.subplot()
    ax1.set_title(audio_file)
    ax1.plot(np.arange(audio.size) / sample_rate, audio, 'gray')
    ax1.set_xlim([0, int(dur) + 1])
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylabel('Signal')
    ax1.set_ylim([-1, 1])
    ax2 = ax1.twinx()

    ax2.plot(np.arange(length) * UNIT_FRAME_LEN, labels, 'r', label='label')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='lower right', shadow=True)
    ax2.set_ylabel('Labels')
    ax2.set_ylim([-0.1, 1.1])
    plt.show()
    if save_path != "":
        plt.savefig(save_path)
    return ipd.Audio(audio, rate=16000)


def load_rttm_file(filepath):
    data = pd.read_csv(filepath, sep=" ", delimiter=None, header=None)
    data = data.rename(columns={3: "start", 4: "dur", 7: "speaker"})

    data['start'] = data['start'].astype(float)
    data['dur'] = data['dur'].astype(float)
    data['end'] = data['start'] + data['dur']

    data = data.sort_values(by=['start'])
    data['segment'] = list(zip(data['start'], data['end']))

    return data


def merge_intervals(intervals: List[List[float]]) -> List[List[float]]:
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        # if the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # otherwise, there is overlap, so we merge the current and previous
            # intervals.
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged


def get_speech_segments(rttm_file):
    speech_segments = list(load_rttm_file(rttm_file)['segment'])
    speech_segments = [list(x) for x in speech_segments]
    speech_segments = merge_intervals(speech_segments)
    return speech_segments


def get_frame_labels(segments: List[List[float]], frame_length: float, offset: float, duration: float, sid: int = 0):
    labels = []
    n_frames = np.ceil(duration / frame_length)

    for i in range(n_frames):
        t = offset + i * frame_length
        while sid < len(segments) - 1 and segments[sid][1] < t:
            sid += 1
        if segments[sid][0] <= t <= segments[sid][1]:
            labels.append('1')
        else:
            labels.append('0')
    return ' '.join(labels), sid


def plot_sample_from_rttm(audio_file: str, rttm_file: str, max_duration: Optional[float] = None, save_path: str = ""):
    plt.figure(figsize=[20, 2])
    UNIT_FRAME_LEN = 0.01

    audio, sample_rate = librosa.load(path=audio_file, sr=16000, mono=True, offset=0, duration=max_duration)
    dur = librosa.get_duration(y=audio, sr=sample_rate)

    segments = get_speech_segments(rttm_file)
    labels, _ = get_frame_labels(segments, UNIT_FRAME_LEN, 0.0, dur)
    labels = [float(x) for x in labels.split()]

    length = len(labels)
    ax1 = plt.subplot()
    ax1.set_title(audio_file)
    ax1.plot(np.arange(audio.size) / sample_rate, audio, 'gray')
    ax1.set_xlim([0, int(dur) + 1])
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylabel('Signal')
    ax1.set_ylim([-1, 1])
    ax2 = ax1.twinx()

    ax2.plot(np.arange(length) * UNIT_FRAME_LEN, labels, 'r', label='label')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='lower right', shadow=True)
    ax2.set_ylabel('Labels')
    ax2.set_ylim([-0.1, 1.1])
    plt.show()
    if save_path != "":
        plt.savefig(save_path)
    return ipd.Audio(audio, rate=16000)
