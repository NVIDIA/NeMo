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
