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
import os
import pickle as pkl

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from sklearn.cluster import SpectralClustering

from nemo.utils import logging
import numpy as np
from copy import deepcopy
import json
import math

def get_contiguous_stamps(stamps):
    lines = deepcopy(stamps)
    contiguous_stamps = []
    for i in range(len(lines)-1):
        start, end, speaker = lines[i].split()
        next_start, next_end, next_speaker = lines[i+1].split()
        if float(end) > float(next_start):
            avg = str((float(next_start) + float(end)) / 2.0)
            lines[i+1] = ' '.join([avg, next_end, next_speaker])
            contiguous_stamps.append(start + " " + avg + " " + speaker)
        else:
            contiguous_stamps.append(start + " " + end + " " + speaker)
    start, end, speaker = lines[-1].split()
    contiguous_stamps.append(start + " " + end + " " + speaker)
    return contiguous_stamps


def merge_stamps(lines):
    stamps = deepcopy(lines)
    overlap_stamps = []
    for i in range(len(stamps)-1):
        start, end, speaker = stamps[i].split()
        next_start, next_end, next_speaker = stamps[i+1].split()
        if float(end) == float(next_start) and speaker == next_speaker:
            stamps[i+1] = ' '.join([start, next_end, next_speaker])   
        else:
            overlap_stamps.append(start + " " + end + " " + speaker)
    
    start, end, speaker = stamps[-1].split()
    overlap_stamps.append(start + " " + end + " " + speaker)
    
    return overlap_stamps

def labels_to_pyannote_object(labels):
    annotation = Annotation()
    for label in labels:
        start,end,speaker = label.strip().split()
        start,end = float(start),float(end)
        annotation[Segment(start,end)]=speaker
    
    return annotation

def rttm_to_labels(rttm_filename,write=False):
    labels=[]
    with open(rttm_filename,'r') as f:
        for line in f.readlines():
            rttm = line.strip().split()
            start,end,speaker = float(rttm[3]),float(rttm[4])+float(rttm[3]),rttm[7]
            labels.append('{} {} {}'.format(start,end,speaker))
    return labels

def get_time_stamps(embeddings_file,reco2num,manifest_path,SAMPLE_RATE,WINDOW,SHIFT):
    # embeddings_file = '/data/samsungSSD/NVIDIA/repos/NeMo/examples/speaker_recognition/embeddings/test_diarize_embeddings.pkl'
    # reco2num = '/disk2/datasets/NIST_SRE_2000_LDC2001S97/NIST_SRE_2000_LDC2001S97_16k/reco2num'
    # '/disk2/datasets/FisherEnglishTrainingSpeech/RTTM/'

    # manifest_path = '/disk2/datasets/NIST_SRE_2000_LDC2001S97/NIST_SRE_2000_LDC2001S97_16k/test_diarize.json'


    embeddings = pkl.load(open(embeddings_file,'rb'))
    num_files = len(embeddings.keys())
    print("Number of files to diarize: ",num_files)
    sample = list(embeddings.keys())[0]
    print("sample '{}' embeddings shape is {}\n".format(sample, embeddings[sample][0].shape))

    # WINDOW=1.5
    # SHIFT=0.75
    # SAMPLE_RATE=16000

    SPEAKERS = {}
    if type(reco2num) is int:
        for key in embeddings.keys():
            SPEAKERS[key]=reco2num
    else:
        for key in open(reco2num).readlines():
            key = key.strip()
            wav_id, num = key.split()
            SPEAKERS[wav_id] = int(num)

    with open(manifest_path,'r') as manifest:
        time_stamps={}
        for line in manifest.readlines():
            line = line.strip()
            line = json.loads(line)
            audio,offset,duration = line['audio_filepath'],line['offset'],line['duration']
            audio = os.path.basename(audio).split('.')[0]
            if audio not in time_stamps:
                time_stamps[audio]=[]
            start = offset
            base = math.ceil((duration-WINDOW)/SHIFT) 
            slices = 1 if base < 0 else base + 1
            for slice_id in range(slices):
                end = start + WINDOW
                stamp = '{:.3f} {:.3f} '.format(start,end)
                time_stamps[audio].append(stamp)
                start = offset + (slice_id+1)*SHIFT
    
    return embeddings,time_stamps,SPEAKERS

def perform_clustering(embeddings,time_stamps,SPEAKERS,RTTM_DIR):

    metric = DiarizationErrorRate(collar=0.0)
    DER = 0

    all_hypothesis=[]
    all_reference=[]

    for uniq_key in embeddings.keys():
        NUM_SPEAKERS = SPEAKERS[uniq_key]
        if NUM_SPEAKERS==2:
            emb = embeddings[uniq_key]
            emb = np.asarray(emb)
            cluster_method = SpectralClustering(n_clusters=NUM_SPEAKERS,random_state=42)
            cluster_method.fit(emb)
            lines = time_stamps[uniq_key]
            cluster_labels = cluster_method.labels_
            assert len(cluster_labels)==len(lines)
            for idx,label in enumerate(cluster_labels):
                tag = 'speaker_'+str(label)
                lines[idx]+=tag
            #ReSegmentation -> VAD and Segmented Results 
            a = get_contiguous_stamps(lines)
            labels = merge_stamps(a)
            hypothesis = labels_to_pyannote_object(labels)
            all_hypothesis.append(hypothesis)
            
            rttm_file=os.path.join(RTTM_DIR,uniq_key+'.rttm')
            ref_labels = rttm_to_labels(rttm_file)
            reference = labels_to_pyannote_object(ref_labels)
            all_reference.append(reference)
    

    for reference, hypothesis in zip(all_reference,all_hypothesis):
        metric(reference,hypothesis,detailed=True)

    DER = abs(metric)
    print(metric[:])
    return DER


def get_score(embeddings_file,reco2num,manifest_path,SAMPLE_RATE,WINDOW,SHIFT,RTTM_DIR):

    embeddings,time_stamps,SPEAKERS = get_time_stamps(embeddings_file,reco2num,manifest_path,SAMPLE_RATE,WINDOW,SHIFT)
    DER = perform_clustering(embeddings,time_stamps,SPEAKERS,RTTM_DIR)

    return DER
