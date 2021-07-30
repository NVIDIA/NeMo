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
import glob
import json
import os
from itertools import repeat
from multiprocessing import Pool
import shutil
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics import detection

from nemo.utils import logging
from sklearn.model_selection import ParameterGrid


"""
This file contains all the utility functions required for speaker embeddings part in diarization scripts
"""


def prepare_manifest(config):
    """
    Perform VAD on long audio snippet might cause CUDA out of memory issue. 
    Automatically split manifest entry by split_duration to avoid the potential memory issue.
    """
    manifest_vad_input = config.get('manifest_vad_input', "manifest_vad_input.json")
    input_audios = []
    with open(config['manifest_filepath'], 'r') as manifest:
        for line in manifest.readlines():
            input_audios.append(json.loads(line.strip()))

    p = Pool(processes=config['num_workers'])
    args_func = {
        'label': 'infer',
        'split_duration': config['split_duration'],
        'time_length': config['time_length'],
    }
    results = p.starmap(write_vad_infer_manifest, zip(input_audios, repeat(args_func)))
    p.close()

    if os.path.exists(manifest_vad_input):
        logging.info("The prepared manifest file exists. Overwriting!")
        os.remove(manifest_vad_input)

    with open(manifest_vad_input, 'a') as fout:
        for res in results:
            for r in res:
                json.dump(r, fout)
                fout.write('\n')
                fout.flush()

    return manifest_vad_input


def write_vad_infer_manifest(file, args_func):
    """
    Used by prepare_manifest.
    Given a list of files, write them to manifest for dataloader with restrictions.
    Args:
        files (json) : file to be processed
        label (str): label for audio snippet.
        split_duration (float): max duration of each audio clip (each line in json)
        time_length (float) : length of window for generating the frame. Used for taking care of joint. 
    Returns:
        res (list) : list of generated metadata line of json for file
    """
    res = []
    label = args_func['label']
    split_duration = args_func['split_duration']
    time_length = args_func['time_length']

    filepath = file['audio_filepath']
    in_duration = file['duration']
    in_offset = file['offset']

    try:
        sr = 16000
        x, _sr = librosa.load(filepath, sr=sr, offset=in_offset, duration=in_duration)
        duration = librosa.get_duration(x, sr=sr)
        left = duration
        current_offset = in_offset

        status = 'single'
        while left > 0:
            if left <= split_duration:
                if status == 'single':
                    write_duration = left
                    current_offset = 0
                else:
                    status = 'end'
                    write_duration = left + time_length
                    current_offset -= time_length
                offset_inc = left
                left = 0
            else:
                if status == 'start' or status == 'next':
                    status = 'next'
                else:
                    status = 'start'

                if status == 'start':
                    write_duration = split_duration
                    offset_inc = split_duration
                else:
                    write_duration = split_duration + time_length
                    current_offset -= time_length
                    offset_inc = split_duration + time_length

                left -= split_duration

            metadata = {
                'audio_filepath': filepath,
                'duration': write_duration,
                'label': label,
                'text': '_',
                'offset': current_offset,
            }
            res.append(metadata)

            current_offset += offset_inc

    except Exception as e:
        err_file = "error.log"
        with open(err_file, 'w') as fout:
            fout.write(file + ":" + str(e))
    return res


def get_vad_stream_status(data):
    """
    Generate a list of status for each snippet in manifest. A snippet should be in single, start, next or end status. 
    Used for concatenate to full audio file.
    Args:
        data (list): list of filepath of audio snippet
    Returns:
        status (list): list of status of each snippet.
    """
    if len(data) == 1:
        return ['single']

    status = [None] * len(data)
    for i in range(len(data)):
        if i == 0:
            status[i] = 'start' if data[i] == data[i + 1] else 'single'
        elif i == len(data) - 1:
            status[i] = 'end' if data[i] == data[i - 1] else 'single'
        else:
            if data[i] != data[i - 1] and data[i] == data[i + 1]:
                status[i] = 'start'
            elif data[i] == data[i - 1] and data[i] == data[i + 1]:
                status[i] = 'next'
            elif data[i] == data[i - 1] and data[i] != data[i + 1]:
                status[i] = 'end'
            else:
                status[i] = 'single'
    return status


def generate_overlap_vad_seq(frame_pred_dir, smoothing_method, overlap, seg_len, shift_len, num_workers):
    """
    Gnerate predictions with overlapping input windows/segments. Then a smoothing filter is applied to decide the label for a frame spanned by multiple windows. 
    Two common smoothing filters are supported: majority vote (median) and average (mean).
    Args:
        frame_pred_dir (str): Directory of frame prediction file to be processed.
        smoothing_method (str): median or mean smoothing filter.
        overlap (float): amounts of overlap of adjacent windows.
        seg_len (float): length of window for generating the frame.
        shift_len (float): amount of shift of window for generating the frame.
        out_dir (str): directory of generated predictions.
        num_workers(float): number of process for multiprocessing
    Returns:
        overlap_out_dir(str): directory of generate predictions.
    """

    p = Pool(processes=num_workers)
    frame_filepathlist = glob.glob(frame_pred_dir + "/*.frame")

    overlap_out_dir = frame_pred_dir + "/overlap_smoothing_output" + "_" + smoothing_method + "_" + str(overlap)

    if not os.path.exists(overlap_out_dir):
        os.mkdir(overlap_out_dir)

    per_args = {
        "out_dir": overlap_out_dir,
        "smoothing_method": smoothing_method,
        "overlap": overlap,
        "seg_len": seg_len,
        "shift_len": shift_len,
    }
    p.starmap(generate_overlap_vad_seq_per_file, zip(frame_filepathlist, repeat(per_args)))
    p.close()
    p.join()

    return overlap_out_dir


def generate_overlap_vad_seq_per_file(frame_filepath, per_args):
    """
    Use generated frame prediction (generated by shifting window of shift_len (10ms)) to generate prediction with overlapping input window/segments
    See discription in generate_overlap_vad_seq.
    """
    try:
        smoothing_method = per_args['smoothing_method']
        overlap = per_args['overlap']
        seg_len = per_args['seg_len']
        shift_len = per_args['shift_len']
        out_dir = per_args['out_dir']

        frame = np.loadtxt(frame_filepath)
        name = os.path.basename(frame_filepath).split(".frame")[0] + "." + smoothing_method
        overlap_filepath = os.path.join(out_dir, name)

        shift = int(shift_len / 0.01)  # number of units of shift
        seg = int((seg_len / 0.01 + 1))  # number of units of each window/segment

        jump_on_target = int(seg * (1 - overlap))  # jump on target generated sequence
        jump_on_frame = int(jump_on_target / shift)  # jump on input frame sequence

        if jump_on_frame < 1:
            raise ValueError(
                f"Note we jump over frame sequence to generate overlapping input segments. \n \
            Your input makes jump_on_fram={jump_on_frame} < 1 which is invalid because it cannot jump and will stuck.\n \
            Please try different seg_len, shift_len and overlap choices. \n \
            jump_on_target = int(seg * (1 - overlap)) \n \
            jump_on_frame  = int(jump_on_frame/shift) "
            )

        target_len = int(len(frame) * shift)

        if smoothing_method == 'mean':
            preds = np.zeros(target_len)
            pred_count = np.zeros(target_len)

            for i, og_pred in enumerate(frame):
                if i % jump_on_frame != 0:
                    continue
                start = i * shift
                end = start + seg
                preds[start:end] = preds[start:end] + og_pred
                pred_count[start:end] = pred_count[start:end] + 1

            preds = preds / pred_count
            last_non_zero_pred = preds[pred_count != 0][-1]
            preds[pred_count == 0] = last_non_zero_pred

        elif smoothing_method == 'median':
            preds = [[] for _ in range(target_len)]
            for i, og_pred in enumerate(frame):
                if i % jump_on_frame != 0:
                    continue

                start = i * shift
                end = start + seg
                for j in range(start, end):
                    if j <= target_len - 1:
                        preds[j].append(og_pred)

            preds = np.array([np.median(l) for l in preds])
            nan_idx = np.isnan(preds)
            last_non_nan_pred = preds[~nan_idx][-1]
            preds[nan_idx] = last_non_nan_pred

        else:
            raise ValueError("smoothing_method should be either mean or median")

        round_final = np.round(preds, 4)
        np.savetxt(overlap_filepath, round_final, delimiter='\n')
        
        return overlap_filepath

    except Exception as e:
        raise (e)

def generate_vad_segment_table(vad_pred_dir, postprocessing_params, shift_len, num_workers):
    """
    Convert frame level prediction to speech segment in start and end times format.
    And save to csv file  in rttm-like format
            0, 10, speech
            17,18, speech
    Args:
        vad_pred_dir (str): directory of prediction files to be processed.
        threshold (float): threshold for prediction score (from 0 to 1).
        shift_len (float): amount of shift of window for generating the frame.
        out_dir (str): output dir of generated table/csv file.
        num_workers(float): number of process for multiprocessing
    Returns:
        table_out_dir(str): directory of generate table.
    """

    p = Pool(processes=num_workers)
    suffixes = ("frame", "mean", "median")
    vad_pred_filepath_list = [os.path.join(vad_pred_dir, x) for x in os.listdir(vad_pred_dir) if x.endswith(suffixes)]

    table_out_dir = os.path.join(vad_pred_dir, "table_output")
    if not os.path.exists(table_out_dir):
        os.mkdir(table_out_dir)

    per_args = {
        "shift_len": shift_len,
        "out_dir": table_out_dir,
    }

    per_args = {**per_args, **postprocessing_params}

    p.starmap(generate_vad_segment_table_per_file, zip(vad_pred_filepath_list, repeat(per_args)))
    p.close()
    p.join()

    return table_out_dir


def generate_vad_segment_table_per_file(pred_filepath, per_args):
    """
    See discription in generate_overlap_vad_seq.
    """
    shift_len = per_args['shift_len']
    out_dir = per_args['out_dir']

    name = pred_filepath.split("/")[-1].rsplit(".", 1)[0]
    sequence = np.loadtxt(pred_filepath)

    active_segments = binarization(sequence, per_args)
    active_segments = filtering(active_segments, per_args)

    seg_speech_table = pd.DataFrame(active_segments, columns =['start', 'end'])
    seg_speech_table = seg_speech_table.sort_values('start', ascending=True)
    seg_speech_table['dur'] = seg_speech_table['end'] - seg_speech_table['start'] + shift_len
    seg_speech_table['vad'] = 'speech'

    save_name = name + ".txt"
    save_path = os.path.join(out_dir, save_name)
    seg_speech_table.to_csv(save_path, columns=['start', 'dur', 'vad'], sep='\t', index=False, header=False)
    return save_path


def binarization(sequence, per_args):
    """
    Reference
    Paper: Gregory Gelly and Jean-Luc Gauvain. "Minimum Word Error Training of RNN-based Voice Activity Detection", InterSpeech 2015. 
    Implementation: https://github.com/pyannote/pyannote-audio/blob/master/pyannote/audio/utils/signal.py 


    • an onset and offset thresholds for the detection of the beginning and end of a speech segment;
    • padding durations before and after each speech segment;
    
    """
    shift_len = per_args.get('shift_len', 0.01)

    onset = per_args.get('onset', 0.5) # 
    offset = per_args.get('offset', 0.5)
    pad_onset = per_args.get('pad_onset', 0.0)
    pad_offset = per_args.get('pad_offset', 0.0)

    onset, offset = cal_vad_onset_offset(per_args.get('scale', 'absolute'), onset, offset)

    active = False
    active_segments = set() 
    for i in range(1, len(sequence)):
        # Current frame is active
        if active:
            # Switch from active to inactive
            if sequence[i] < offset:
                active_segments.add((start - pad_onset, i * shift_len + pad_offset))
                start = i * shift_len
                active = False

        # Current frame is inactive
        else:
            # Switch from inactive to active
            if sequence[i] > onset:
                start = i * shift_len
                active = True

    # if active at the end, add final segment            
    if active:
        active_segments.add((start - pad_onset, i * shift_len + pad_offset))

    # Merge the overlapped active segments due to padding
    active_segments = merge_overlap_segment(active_segments) 
    
    return active_segments

def filtering(active_segments, per_args):
    """
    • a threshold for short speech segment deletion;
    • and a threshold for small silence deletion.
    """
    min_duration_on = per_args.get('min_duration_on', 0.2) 
    min_duration_off = per_args.get('min_duration_off', 0.3) 

    # Filter out the shorter active segments
    if min_duration_on > 0.0:
        active_segments = filter_short_segments(active_segments, min_duration_on)
    # Filter out the shorter inactive segments and return to be as active segments
    if min_duration_off > 0.0:
        # Find inactive segments
        inactive_segments = get_gap_segments(active_segments)
        # Find shorter inactive segments
        short_inactive_segments= inactive_segments - filter_short_segments(inactive_segments, min_duration_off)
        # Return shorter inactive segments to be as active segments
        active_segments.update(short_inactive_segments)
        # Merge the overlapped active segments
        active_segments = merge_overlap_segment(active_segments)
    return active_segments

def filter_short_segments(segments, threshold):
    res = set()
    for seg in segments:
        if seg[1]-seg[0] >= threshold:
            res.add(seg)
    return res

def get_gap_segments(segments):
    segments=[list(i) for i in segments]
    segments.sort(key=lambda x: x[0])
    gap_segments = set()
    for i in range(len(segments)-1):
        gap_segments.add((segments[i][1], segments[i+1][0]))
    return gap_segments

def merge_overlap_segment(segments):
    segments=[list(i) for i in segments]
    segments.sort(key=lambda x: x[0])
    merged = []
    for segment in segments:
        if not merged or merged[-1][1] < segment[0]:
            merged.append(segment)
        else:
            merged[-1][1] = max(merged[-1][1], segment[1])

    merged_set = set([tuple(t) for t in merged])
    return merged_set


def cal_vad_onset_offset(scale, onset, offset):
    if scale == "absolute":
        mini = 0
        maxi = 1
    elif scale == "relative":
        mini = np.nanmin(data)
        maxi = np.nanmax(data)
    elif scale == "percentile":
        mini = np.nanpercentile(data, 1)
        maxi = np.nanpercentile(data, 99)

    onset = mini + onset * (maxi - mini)
    offset = mini + offset * (maxi - mini)
    return onset, offset


def vad_construct_pyannote_object_per_file(vad_table_filepath, groundtruth_RTTM_file):
    """
    Construct pyannote object for evaluation.
    Args:
        vad_table_filepath(str) : path of vad rttm-like table.
        groundtruth_RTTM_file(str): path of groundtruth rttm file.
    Returns:
        reference(pyannote.Annotation): groundtruth
        hypothesise(pyannote.Annotation): prediction
    """

    pred = pd.read_csv(vad_table_filepath, sep="\t", header=None)
    label = pd.read_csv(groundtruth_RTTM_file, sep=" ", delimiter=None, header=None)
    label = label.rename(columns={3: "start", 4: "dur", 7: "speaker"})

    # construct reference
    reference = Annotation()
    for index, row in label.iterrows():
        reference[Segment(row['start'], row['start'] + row['dur'])] = row['speaker']

    # construct hypothsis
    hypothesis = Annotation()
    for index, row in pred.iterrows():
        hypothesis[Segment(row[0], row[0] + row[1])] = 'Speech'
    return reference, hypothesis


def get_parameter_grid(params):
    return list(ParameterGrid(params))

def vad_tune_threshold_on_dev(params, vad_pred, groundtruth_RTTM, vad_pred_method="frame", focus_metric="DetER"):
    """
    Tune threshold on dev set. Return best threshold which gives the lowest detection error rate (DetER) in thresholds.
    Args:
        thresholds (list): list of thresholds.
        vad_pred_method (str): suffix of prediction file. Use to locate file. Should be either in "frame", "mean" or "median".
        vad_pred_dir (str): directory of vad predictions or a file contains the paths of them
        groundtruth_RTTM_dir (str): directory of groundtruch rttm files or a file contains the paths of them.
        focus_metric (str): metrics we care most when tuning threshold. Should be either in "DetER", "FA", "MISS"
    Returns:
        best_threhsold (float): threshold that gives lowest DetER.
    """
    all_perf = {}
    min_score = 100

    # try:
    #     onsets[0] >= 0 and onsets[-1] <= 1
    # except:
    #     raise ValueError("Invalid onset! Should be in [0, 1]")

    params_grid = get_parameter_grid(params)

    for param in params_grid:
        metric = detection.DetectionErrorRate()
        paired_filenames, groundtruth_RTTM_dict, vad_pred_dict = pred_rttm_map(
            vad_pred, groundtruth_RTTM, vad_pred_method
        )
        table_out_dir = "ami_table_output_tmp"
        if not os.path.exists(table_out_dir):
            os.makedirs(table_out_dir)
                
        for filename in paired_filenames:
            vad_pred_filepath = vad_pred_dict[filename]
            groundtruth_RTTM_file = groundtruth_RTTM_dict[filename]

            # todo
            # if os.path.isdir(vad_pred):
            #     table_out_dir = os.path.join(vad_pred, "table_output_" + str(param))
            # else:
            #     table_out_dir = os.path.join("tmp_table_outputs", "table_output_" + str(param))

        
            per_args = {"shift_len": 0.01, "out_dir": table_out_dir}
            per_args = {**per_args, **param}
            
            vad_table_filepath = generate_vad_segment_table_per_file(vad_pred_filepath, per_args)
            
        
            reference, hypothesis = vad_construct_pyannote_object_per_file(vad_table_filepath, groundtruth_RTTM_file)
            metric(reference, hypothesis)  # accumulation
        
        shutil.rmtree(table_out_dir, ignore_errors=True)

        report = metric.report(display=False)
        DetER = report.iloc[[-1]][('detection error rate', '%')].item()
        FA = report.iloc[[-1]][('false alarm', '%')].item()
        MISS = report.iloc[[-1]][('miss', '%')].item()

        assert (focus_metric == "DetER" or focus_metric == "FA" or focus_metric == "MISS"  ), "Metric we care most should be only in 'DetER', 'FA'or 'MISS'!"
        all_perf[str(param)] = {'DetER (%)': DetER, 'FA (%)': FA, 'MISS (%)': MISS}
        logging.info(f"parameter {param}, {all_perf[str(param)] }")

        score = all_perf[str(param)][focus_metric + ' (%)']

        del report
        metric.reset()  # reset internal accumulator
        if score < min_score:
            best_threhsold = param
            optimal_scores= all_perf[str(param)]
            min_score = score

    return best_threhsold, optimal_scores


def pred_rttm_map(vad_pred, groundtruth_RTTM, vad_pred_method="frame"):
    """
    Find paired files in vad_pred and groundtruth_RTTM
    """
    groundtruth_RTTM_dict = {}
    if os.path.isfile(groundtruth_RTTM):
        with open(groundtruth_RTTM, "r") as fp:
            groundtruth_RTTM_files = fp.read().splitlines()
    elif os.path.isdir(groundtruth_RTTM):
        groundtruth_RTTM_files = glob.glob(os.path.join(groundtruth_RTTM, "*.rttm"))
    else:
        raise ValueError(
            "groundtruth_RTTM should either be a directory contains rttm files or a file contains paths to them!"
        )
    for f in groundtruth_RTTM_files:
        filename = os.path.basename(f).rsplit(".", 1)[0]
        groundtruth_RTTM_dict[filename] = f

    vad_pred_dict = {}
    if os.path.isfile(vad_pred):
        with open(vad_pred, "r") as fp:
            vad_pred_files = fp.read().splitlines()
    elif os.path.isdir(vad_pred):
        vad_pred_files = glob.glob(os.path.join(vad_pred, "*." + vad_pred_method))
    else:
        raise ValueError(
            "vad_pred should either be a directory contains vad pred files or a file contains paths to them!"
        )
    for f in vad_pred_files:
        filename = os.path.basename(f).rsplit(".", 1)[0]
        vad_pred_dict[filename] = f

    paired_filenames = groundtruth_RTTM_dict.keys() & vad_pred_dict.keys()
    return paired_filenames, groundtruth_RTTM_dict, vad_pred_dict


def plot(path2audio_file, path2_vad_pred, path2ground_truth_label=None, threshold=0.85):
    """
    Plot VAD outputs for demonstration in tutorial
    Args:
        path2audio_file (str):  path to audio file.
        path2_vad_pred (str): path to vad prediction file,
        path2ground_truth_label(str): path to groundtruth label file.
        threshold (float): threshold for prediction score (from 0 to 1).
    """
    plt.figure(figsize=[20, 2])
    FRAME_LEN = 0.01
    audio, sample_rate = librosa.load(path=path2audio_file, sr=16000, mono=True)
    dur = librosa.get_duration(audio, sr=sample_rate)
    time = np.arange(0, dur, FRAME_LEN)
    frame = np.loadtxt(path2_vad_pred)
    len_pred = len(frame)
    ax1 = plt.subplot()
    ax1.plot(np.arange(audio.size) / sample_rate, audio, 'gray')
    ax1.set_xlim([0, int(dur) + 1])
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylabel('Signal')
    ax1.set_ylim([-1, 1])
    ax2 = ax1.twinx()
    prob = frame
    pred = np.where(prob >= threshold, 1, 0)
    if path2ground_truth_label:
        label = extract_labels(path2ground_truth_label, time)
        ax2.plot(np.arange(len_pred) * FRAME_LEN, label, 'r', label='label')
    ax2.plot(np.arange(len_pred) * FRAME_LEN, pred, 'b', label='pred')
    ax2.plot(np.arange(len_pred) * FRAME_LEN, prob, 'g--', label='speech prob')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='lower right', shadow=True)
    ax2.set_ylabel('Preds and Probas')
    ax2.set_ylim([-0.1, 1.1])
    return None


def extract_labels(path2ground_truth_label, time):
    """
    Extract groundtruth label for given time period.
    path2ground_truth_label (str): path of groundtruth label file 
    time (list) : a list of array represent time period.
    """

    data = pd.read_csv(path2ground_truth_label, sep=" ", delimiter=None, header=None)
    data = data.rename(columns={3: "start", 4: "dur", 7: "speaker"})
    labels = []
    for pos in time:
        line = data[(data["start"] <= pos) & (data["start"] + data["dur"] > pos)]
        if len(line) >= 1:
            labels.append(1)
        else:
            labels.append(0)
    return labels
