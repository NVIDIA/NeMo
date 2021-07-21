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
import torch
from omegaconf import ListConfig
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from tqdm import tqdm

from nemo.collections.asr.parts.utils.nmse_clustering import COSclustering
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


def get_time_stamps(embeddings_file, reco2num, manifest_path):
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
    if isinstance(reco2num, int) or reco2num is None:
        for key in embeddings.keys():
            speakers[key] = reco2num
    elif isinstance(reco2num, str) and os.path.exists(reco2num):
        for key in open(reco2num).readlines():
            key = key.strip()
            wav_id, num = key.split()
            speakers[wav_id] = int(num)
    else:
        raise TypeError("reco2num must be None, int or path to file containing number of speakers for each uniq id")

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
            end = start + duration
            stamp = '{:.3f} {:.3f} '.format(start, end)
            time_stamps[audio].append(stamp)

    return embeddings, time_stamps, speakers


def perform_clustering(embeddings, time_stamps, speakers, audio_rttm_map, out_rttm_dir, max_num_speakers=8):
    """
    performs spectral clustering on embeddings with time stamps generated from VAD output
    Args:
    
    embeddings (dict): Embeddings with key as unique_id
    time_stamps (dict): time stamps list for each audio recording
    speakers (dict): number of speaker for each audio recording 
    audio_rttm_map (dict): AUDIO_RTTM_MAP for mapping unique id with audio file path and rttm path
    out_rttm_dir (str): Path to write predicted rttms
    max_num_speakers (int): maximum number of speakers to consider for spectral clustering. Will be ignored if speakers['key'] is not None
    
    Returns:
    all_reference (list[Annotation]): reference annotations for score calculation
    all_hypothesis (list[Annotation]): hypothesis annotations for score calculation

    """
    all_hypothesis = []
    all_reference = []
    no_references = False

    cuda = True
    if not torch.cuda.is_available():
        logging.warning("cuda=False, using CPU for Eigen decompostion. This might slow down the clustering process.")
        cuda = False

    for uniq_key in tqdm(embeddings.keys()):
        NUM_speakers = speakers[uniq_key]
        emb = embeddings[uniq_key]
        emb = np.asarray(emb)

        cluster_labels = COSclustering(
            uniq_key, emb, oracle_num_speakers=NUM_speakers, max_num_speaker=max_num_speakers, cuda=cuda,
        )

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

    < Caveat >
    Unlike md-eval.pl, "no score" collar in pyannote.metrics is the maximum length of
    "no score" collar from left to right. Therefore, if 0.25s is applied for "no score"
    collar in md-eval.pl, 0.5s should be applied for pyannote.metrics.

    """
    metric = DiarizationErrorRate(collar=0.5, skip_overlap=True)

    for reference, hypothesis in zip(all_reference, all_hypothesis):
        metric(reference, hypothesis, detailed=True)

    DER = abs(metric)
    CER = metric['confusion'] / metric['total']
    FA = metric['false alarm'] / metric['total']
    MISS = metric['missed detection'] / metric['total']

    metric.reset()

    return DER, CER, FA, MISS


def perform_diarization(
    embeddings_file=None, reco2num=2, manifest_path=None, audio_rttm_map=None, out_rttm_dir=None, max_num_speakers=8,
):
    """
    Performs diarization with embeddings generated based on VAD time stamps with recording 2 num of speakers (reco2num)
    for spectral clustering 
    """
    embeddings, time_stamps, speakers = get_time_stamps(embeddings_file, reco2num, manifest_path)
    logging.info("Performing Clustering")
    all_reference, all_hypothesis = perform_clustering(
        embeddings, time_stamps, speakers, audio_rttm_map, out_rttm_dir, max_num_speakers
    )

    if len(all_reference) and len(all_hypothesis):
        DER, CER, FA, MISS = get_DER(all_reference, all_hypothesis)
        logging.info(
            "Cumulative results of all the files:  \n FA: {:.4f}\t MISS {:.4f}\t \
                Diarization ER: {:.4f}\t, Confusion ER:{:.4f}".format(
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


def segments_manifest_to_subsegments_manifest(
    segments_manifest_file: str,
    sub_segments_manifest_file: str = None,
    window: float = 1.5,
    shift: float = 0.75,
    min_subsegement_duration: float = 0.05,
):
    """
    Generate subsegments manifest from segments manifest file
    Args
    input:
        segments_manifest file (str): path to segments manifest file, typically from VAD output
        sub_segments_manifest_file (str): path to output sub segments manifest file (default (None) : writes to current working directory)
        window (float): window length for segments to sub segments division
        shift (float): hop length for sub segments shift 
        min_subsegments_duration (float): exclude sub segments smaller than this duration value
    
    output:
        returns path to subsegment manifest file
    """
    if sub_segments_manifest_file is None:
        pwd = os.getcwd()
        sub_segments_manifest_file = os.path.join(pwd, 'sub_segments.json')

    with open(segments_manifest_file, 'r') as segments_manifest, open(
        sub_segments_manifest_file, 'w'
    ) as sub_segments_manifest:
        segments = segments_manifest.readlines()
        for segment in segments:
            segment = segment.strip()
            dic = json.loads(segment)
            audio, offset, duration, label = dic['audio_filepath'], dic['offset'], dic['duration'], dic['label']
            sub_segments = get_sub_time_segments(offset=offset, window=window, shift=shift, duration=duration)

            for sub_segment in sub_segments:
                start, dur = sub_segment
                if dur > min_subsegement_duration:
                    meta = {"audio_filepath": audio, "offset": start, "duration": dur, "label": label}
                    json.dump(meta, sub_segments_manifest)
                    sub_segments_manifest.write("\n")

    return sub_segments_manifest_file


def get_sub_time_segments(offset: float, window: float, shift: float, duration: float):
    """
    return sub segments from a segment of audio file
    Args
    input:
        offset (float): start time of audio segment
        window (float): window length for segments to sub segments division
        shift (float): hop length for sub segments shift 
        duration (float): duration of segment
    output:
        sub_segments (List[tuple[float, float]]): subsegments generated for the segments as list of tuple of start and duration of each sub segment
    """
    sub_segments = []
    start = offset
    slice_end = start + duration
    base = math.ceil((duration - window) / shift)
    slices = 1 if base < 0 else base + 1
    for slice_id in range(slices):
        end = start + window
        if end > slice_end:
            end = slice_end
        sub_segments.append((start, end - start))
        start = offset + (slice_id + 1) * shift

    return sub_segments


def embedding_normalize(embs, use_std=False, eps=1e-10):
    """
    mean and l2 length normalize the input speaker embeddings
    input:
        embs: embeddings of shape (Batch,emb_size)
    output:
        embs: normalized embeddings of shape (Batch,emb_size)
    """
    embs = embs - embs.mean(axis=0)
    if use_std:
        embs = embs / (embs.std(axis=0) + eps)
    embs_l2_norm = np.expand_dims(np.linalg.norm(embs, ord=2, axis=-1), axis=1)
    embs = embs / embs_l2_norm

    return embs
