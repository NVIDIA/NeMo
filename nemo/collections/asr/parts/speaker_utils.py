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
import json
import math
import os
import pickle as pkl
from copy import deepcopy

import numpy as np
from omegaconf import ListConfig
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from spectralcluster import SpectralClusterer

from nemo.utils import logging


"""
This file contains all the utility functions required for speaker embeddings part in diarization scripts
"""


def audio_rttm_map(audio_file_list, rttm_file_list=None):
    """
    Returns a AUDIO_TO_RTTM dictionary thats maps all the unique file names with 
    audio file paths and corresponding ground truth rttm files calculated from audio
    file list and rttm file list

    Args: 
    audio_file_list(list,str): either list of audio file paths or file containing paths to audio files (required)
    rttm_file_list(lisr,str): either list of rttm file paths or file containing paths to rttm files (optional) 
    [Required if DER needs to be calculated]
    Returns:
    AUDIO_RTTM_MAP (dict): dictionary thats maps all the unique file names with 
    audio file paths and corresponding ground truth rttm files 
    """
    rttm_notfound = False
    if type(audio_file_list) in [list, ListConfig]:
        audio_files = audio_file_list
    else:
        audio_pointer = open(audio_file_list, 'r')
        audio_files = audio_pointer.read().splitlines()

    if rttm_file_list:
        if type(rttm_file_list) in [list, ListConfig]:
            rttm_files = rttm_file_list
        else:
            rttm_pointer = open(rttm_file_list, 'r')
            rttm_files = rttm_pointer.read().splitlines()
    else:
        rttm_notfound = True
        rttm_files = ['-'] * len(audio_files)

    assert len(audio_files) == len(rttm_files)

    AUDIO_RTTM_MAP = {}
    rttm_dict = {}
    audio_dict = {}
    for audio_file, rttm_file in zip(audio_files, rttm_files):
        uniq_audio_name = audio_file.split('/')[-1].rsplit('.', 1)[0]
        uniq_rttm_name = rttm_file.split('/')[-1].rsplit('.', 1)[0]

        if rttm_notfound:
            uniq_rttm_name = uniq_audio_name

        audio_dict[uniq_audio_name] = audio_file
        rttm_dict[uniq_rttm_name] = rttm_file

    for key, value in audio_dict.items():

        AUDIO_RTTM_MAP[key] = {'audio_path': audio_dict[key], 'rttm_path': rttm_dict[key]}

    assert len(rttm_dict.items()) == len(audio_dict.items())

    return AUDIO_RTTM_MAP


def get_contiguous_stamps(stamps):
    """
    Return contiguous time stamps 
    """
    lines = deepcopy(stamps)
    contiguous_stamps = []
    for i in range(len(lines) - 1):
        start, end, speaker = lines[i].split()
        next_start, next_end, next_speaker = lines[i + 1].split()
        if float(end) > float(next_start):
            avg = str((float(next_start) + float(end)) / 2.0)
            lines[i + 1] = ' '.join([avg, next_end, next_speaker])
            contiguous_stamps.append(start + " " + avg + " " + speaker)
        else:
            contiguous_stamps.append(start + " " + end + " " + speaker)
    start, end, speaker = lines[-1].split()
    contiguous_stamps.append(start + " " + end + " " + speaker)
    return contiguous_stamps


def merge_stamps(lines):
    """
    merge time stamps of same speaker
    """
    stamps = deepcopy(lines)
    overlap_stamps = []
    for i in range(len(stamps) - 1):
        start, end, speaker = stamps[i].split()
        next_start, next_end, next_speaker = stamps[i + 1].split()
        if float(end) == float(next_start) and speaker == next_speaker:
            stamps[i + 1] = ' '.join([start, next_end, next_speaker])
        else:
            overlap_stamps.append(start + " " + end + " " + speaker)

    start, end, speaker = stamps[-1].split()
    overlap_stamps.append(start + " " + end + " " + speaker)

    return overlap_stamps


def labels_to_pyannote_object(labels):
    """
    converts labels to pyannote object to calculate DER and for visualization
    """
    annotation = Annotation()
    for label in labels:
        start, end, speaker = label.strip().split()
        start, end = float(start), float(end)
        annotation[Segment(start, end)] = speaker

    return annotation


def labels_to_rttmfile(labels, uniq_id, out_rttm_dir):
    """
    write rttm file with uniq_id name in out_rttm_dir with time_stamps in labels
    """
    filename = os.path.join(out_rttm_dir, uniq_id + '.rttm')
    with open(filename, 'w') as f:
        for line in labels:
            line = line.strip()
            start, end, speaker = line.split()
            duration = float(end) - float(start)
            start = float(start)
            log = 'SPEAKER {} 1   {:.3f}   {:.3f} <NA> <NA> {} <NA> <NA>\n'.format(uniq_id, start, duration, speaker)
            f.write(log)


def rttm_to_labels(rttm_filename):
    """
    prepares time stamps label list from rttm file
    """
    labels = []
    with open(rttm_filename, 'r') as f:
        for line in f.readlines():
            rttm = line.strip().split()
            start, end, speaker = float(rttm[3]), float(rttm[4]) + float(rttm[3]), rttm[7]
            labels.append('{} {} {}'.format(start, end, speaker))
    return labels


def get_time_stamps(embeddings_file, reco2num, manifest_path, sample_rate, window, shift):
    """
    Loads embedding file and generates time stamps based on window and shift for speaker embeddings 
    clustering.

    Args:
    embeddings_file (str): Path to embeddings pickle file
    reco2num (int,str): common integer number of speakers for every recording or path to 
                        file that states unique_file id with number of speakers
    manifest_path (str): path to manifest for time stamps matching
    sample_rate (int): sample rate
    window (float): window length in sec for speaker embeddings extraction
    shift (float): shift length in sec for speaker embeddings extraction window shift

    Returns:
    embeddings (dict): Embeddings with key as unique_id
    time_stamps (dict): time stamps list for each audio recording
    speakers (dict): number of speaker for each audio recording
    """
    embeddings = pkl.load(open(embeddings_file, 'rb'))
    all_uniq_files = list(embeddings.keys())
    num_files = len(all_uniq_files)
    logging.info("Number of files to diarize: {}".format(num_files))
    sample = all_uniq_files[0]
    logging.info("sample '{}' embeddings shape is {}\n".format(sample, embeddings[sample][0].shape))

    speakers = {}
    if type(reco2num) is int:
        for key in embeddings.keys():
            speakers[key] = reco2num
    else:
        for key in open(reco2num).readlines():
            key = key.strip()
            wav_id, num = key.split()
            speakers[wav_id] = int(num)

    with open(manifest_path, 'r') as manifest:
        time_stamps = {}
        for line in manifest.readlines():
            line = line.strip()
            line = json.loads(line)
            audio, offset, duration = line['audio_filepath'], line['offset'], line['duration']
            audio = os.path.basename(audio).rsplit('.', 1)[0]
            if audio not in time_stamps:
                time_stamps[audio] = []
            start = offset
            slice_end = start + duration
            base = math.ceil((duration - window) / shift)
            slices = 1 if base < 0 else base + 1
            for slice_id in range(slices):
                end = start + window
                if end > slice_end:
                    end = slice_end
                stamp = '{:.3f} {:.3f} '.format(start, end)
                time_stamps[audio].append(stamp)
                start = offset + (slice_id + 1) * shift

    return embeddings, time_stamps, speakers


def perform_clustering(embeddings, time_stamps, speakers, audio_rttm_map, out_rttm_dir):
    """
    performs spectral clustering on embeddings with time stamps generated from VAD output
    Args:
    
    embeddings (dict): Embeddings with key as unique_id
    time_stamps (dict): time stamps list for each audio recording
    speakers (dict): number of speaker for each audio recording 
    audio_rttm_map (dict): AUDIO_RTTM_MAP for mapping unique id with audio file path and rttm path
    out_rttm_dir (str): Path to write predicted rttms
    
    Returns:
    all_reference (list[Annotation]): reference annotations for score calculation
    all_hypothesis (list[Annotation]): hypothesis annotations for score calculation

    """
    all_hypothesis = []
    all_reference = []
    no_references = False

    for uniq_key in embeddings.keys():
        NUM_speakers = speakers[uniq_key]
        if NUM_speakers >= 2:
            emb = embeddings[uniq_key]
            emb = np.asarray(emb)

            cluster_method = SpectralClusterer(min_clusters=2, max_clusters=NUM_speakers)
            cluster_labels = cluster_method.predict(emb)

            lines = time_stamps[uniq_key]
            assert len(cluster_labels) == len(lines)
            for idx, label in enumerate(cluster_labels):
                tag = 'speaker_' + str(label)
                lines[idx] += tag

            a = get_contiguous_stamps(lines)
            labels = merge_stamps(a)
            if out_rttm_dir:
                labels_to_rttmfile(labels, uniq_key, out_rttm_dir)
            hypothesis = labels_to_pyannote_object(labels)
            all_hypothesis.append(hypothesis)

            rttm_file = audio_rttm_map[uniq_key]['rttm_path']
            if os.path.exists(rttm_file) and not no_references:
                ref_labels = rttm_to_labels(rttm_file)
                reference = labels_to_pyannote_object(ref_labels)
                all_reference.append(reference)
            else:
                no_references = True
                all_reference = []

    return all_reference, all_hypothesis


def get_DER(all_reference, all_hypothesis):
    """
    calculates DER, CER, FA and MISS

    Args:
    all_reference (list[Annotation]): reference annotations for score calculation
    all_hypothesis (list[Annotation]): hypothesis annotations for score calculation

    Returns:
    DER (float): Diarization Error Rate
    CER (float): Confusion Error Rate
    FA (float): False Alarm
    Miss (float): Miss Detection 

    """
    metric = DiarizationErrorRate(collar=0.25, skip_overlap=True)
    DER = 0

    for reference, hypothesis in zip(all_reference, all_hypothesis):
        metric(reference, hypothesis, detailed=True)

    DER = abs(metric)
    CER = metric['confusion'] / metric['total']
    FA = metric['false alarm'] / metric['total']
    MISS = metric['missed detection'] / metric['total']

    metric.reset()

    return DER, CER, FA, MISS


def perform_diarization(
    embeddings_file=None,
    reco2num=2,
    manifest_path=None,
    sample_rate=16000,
    window=1.5,
    shift=0.75,
    audio_rttm_map=None,
    out_rttm_dir=None,
):
    """
    Performs diarization with embeddings generated based on VAD time stamps with recording 2 num of speakers (reco2num)
    for spectral clustering 
    """
    embeddings, time_stamps, speakers = get_time_stamps(
        embeddings_file, reco2num, manifest_path, sample_rate, window, shift
    )
    logging.info("Performing Clustering")
    all_reference, all_hypothesis = perform_clustering(embeddings, time_stamps, speakers, audio_rttm_map, out_rttm_dir)

    if len(all_reference) and len(all_hypothesis):
        DER, CER, FA, MISS = get_DER(all_reference, all_hypothesis)
        logging.info(
            "Cumulative results of all the files:  \n FA: {:.3f}\t MISS {:.3f}\t \
                Diarization ER: {:.3f}\t, Confusion ER:{:.3f}".format(
                FA, MISS, DER, CER
            )
        )
    else:
        logging.warning("Please check if each ground truth RTTMs was present in provided path2groundtruth_rttm_files")
        logging.warning("Skipping calculation of Diariazation Error Rate")


def write_rttm2manifest(paths2audio_files, paths2rttm_files, manifest_file):
    """
    writes manifest file based on rttm files (or vad table out files). This manifest file would be used by 
    speaker diarizer to compute embeddings and cluster them. This function also takes care of overlap time stamps

    Args:
    audio_file_list(list,str): either list of audio file paths or file containing paths to audio files (required)
    rttm_file_list(lisr,str): either list of rttm file paths or file containing paths to rttm files (optional) 
    manifest (str): path to write manifest file

    Returns:
    manifest (str): path to write manifest file 
    """
    AUDIO_RTTM_MAP = audio_rttm_map(paths2audio_files, paths2rttm_files)

    with open(manifest_file, 'w') as outfile:
        for key in AUDIO_RTTM_MAP:
            f = open(AUDIO_RTTM_MAP[key]['rttm_path'], 'r')
            audio_path = AUDIO_RTTM_MAP[key]['audio_path']
            lines = f.readlines()
            time_tup = (-1, -1)
            for line in lines:
                vad_out = line.strip().split()
                if len(vad_out) > 3:
                    start, dur, activity = float(vad_out[3]), float(vad_out[4]), vad_out[7]
                else:
                    start, dur, activity = float(vad_out[0]), float(vad_out[1]), vad_out[2]
                start, dur = float("{:.3f}".format(start)), float("{:.3f}".format(dur))

                if time_tup[0] >= 0 and start > time_tup[1]:
                    dur2 = float("{:.3f}".format(time_tup[1] - time_tup[0]))
                    meta = {"audio_filepath": audio_path, "offset": time_tup[0], "duration": dur2, "label": 'UNK'}
                    json.dump(meta, outfile)
                    outfile.write("\n")
                    time_tup = (start, start + dur)
                else:
                    if time_tup[0] == -1:
                        time_tup = (start, start + dur)
                    else:
                        time_tup = (min(time_tup[0], start), max(time_tup[1], start + dur))
            dur2 = float("{:.3f}".format(time_tup[1] - time_tup[0]))
            meta = {"audio_filepath": audio_path, "offset": time_tup[0], "duration": dur2, "label": 'UNK'}
            json.dump(meta, outfile)
            outfile.write("\n")
            f.close()
    return manifest_file
