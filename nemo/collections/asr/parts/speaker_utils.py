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
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from sklearn.cluster import SpectralClustering

from nemo.utils import logging


def get_contiguous_stamps(stamps):
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
    annotation = Annotation()
    for label in labels:
        start, end, speaker = label.strip().split()
        start, end = float(start), float(end)
        annotation[Segment(start, end)] = speaker

    return annotation


def labels_to_rttmfile(labels, uniq_id, OUT_RTTM_DIR):
    filename = os.path.join(OUT_RTTM_DIR, uniq_id + '.rttm')
    with open(filename, 'w') as f:
        for line in labels:
            line = line.strip()
            start, end, speaker = line.split()
            duration = float(end) - float(start)
            start = float(start)
            log = 'SPEAKER {} 1   {:.3f}   {:.3f} <NA> <NA> {} <NA> <NA>\n'.format(uniq_id, start, duration, speaker)
            f.write(log)


def rttm_to_labels(rttm_filename):
    labels = []
    with open(rttm_filename, 'r') as f:
        for line in f.readlines():
            rttm = line.strip().split()
            start, end, speaker = float(rttm[3]), float(rttm[4]) + float(rttm[3]), rttm[7]
            labels.append('{} {} {}'.format(start, end, speaker))
    return labels


def get_time_stamps(embeddings_file, reco2num, manifest_path, SAMPLE_RATE, WINDOW, SHIFT):

    embeddings = pkl.load(open(embeddings_file, 'rb'))
    all_uniq_files = list(embeddings.keys())
    num_files = len(all_uniq_files)
    logging.info("Number of files to diarize: {}".format(num_files))
    sample = all_uniq_files[0]
    logging.info("sample '{}' embeddings shape is {}\n".format(sample, embeddings[sample][0].shape))

    SPEAKERS = {}
    if type(reco2num) is int:
        for key in embeddings.keys():
            SPEAKERS[key] = reco2num
    else:
        for key in open(reco2num).readlines():
            key = key.strip()
            wav_id, num = key.split()
            SPEAKERS[wav_id] = int(num)

    with open(manifest_path, 'r') as manifest:
        time_stamps = {}
        for line in manifest.readlines():
            line = line.strip()
            line = json.loads(line)
            audio, offset, duration = line['audio_filepath'], line['offset'], line['duration']
            audio = os.path.basename(audio).split('.')[0]
            if audio not in time_stamps:
                time_stamps[audio] = []
            start = offset
            base = math.ceil((duration - WINDOW) / SHIFT)
            slices = 1 if base < 0 else base + 1
            for slice_id in range(slices):
                end = start + WINDOW
                stamp = '{:.3f} {:.3f} '.format(start, end)
                time_stamps[audio].append(stamp)
                start = offset + (slice_id + 1) * SHIFT

    return embeddings, time_stamps, SPEAKERS


def perform_clustering(embeddings, time_stamps, SPEAKERS, GT_RTTM_DIR, OUT_RTTM_DIR):

    metric = DiarizationErrorRate(collar=0.0)
    DER = 0

    all_hypothesis = []
    all_reference = []

    for uniq_key in embeddings.keys():
        NUM_SPEAKERS = SPEAKERS[uniq_key]
        if NUM_SPEAKERS >= 2:
            emb = embeddings[uniq_key]
            emb = np.asarray(emb)
            cluster_method = SpectralClustering(n_clusters=NUM_SPEAKERS, random_state=42)
            cluster_method.fit(emb)
            lines = time_stamps[uniq_key]
            cluster_labels = cluster_method.labels_
            assert len(cluster_labels) == len(lines)
            for idx, label in enumerate(cluster_labels):
                tag = 'speaker_' + str(label)
                lines[idx] += tag

            a = get_contiguous_stamps(lines)
            labels = merge_stamps(a)
            if OUT_RTTM_DIR:
                labels_to_rttmfile(labels, uniq_key, OUT_RTTM_DIR)
            hypothesis = labels_to_pyannote_object(labels)
            all_hypothesis.append(hypothesis)

            if os.path.exists(GT_RTTM_DIR):
                rttm_file = os.path.join(GT_RTTM_DIR, uniq_key + '.rttm')
                ref_labels = rttm_to_labels(rttm_file)
                reference = labels_to_pyannote_object(ref_labels)
                all_reference.append(reference)

    if len(all_reference) == 0:
        logging.warning("Please check if ground truth RTTMs were present in {}".format(GT_RTTM_DIR))
        logging.warning("Skipping calculation of Diariazation Error rate")
        return (0, 0)

    for reference, hypothesis in zip(all_reference, all_hypothesis):
        metric(reference, hypothesis, detailed=True)

    DER = abs(metric)
    CER = metric['confusion'] / metric['total']

    return DER, CER


def get_score(
    embeddings_file=None,
    reco2num=2,
    manifest_path=None,
    SAMPLE_RATE=16000,
    WINDOW=1.5,
    SHIFT=0.75,
    GT_RTTM_DIR=None,
    OUT_RTTM_DIR=None,
):

    embeddings, time_stamps, SPEAKERS = get_time_stamps(
        embeddings_file, reco2num, manifest_path, SAMPLE_RATE, WINDOW, SHIFT
    )
    DER, CER = perform_clustering(embeddings, time_stamps, SPEAKERS, GT_RTTM_DIR, OUT_RTTM_DIR)

    return DER, CER
