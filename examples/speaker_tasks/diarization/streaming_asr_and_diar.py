#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import pyaudio as pa
import argparse
import os
import nemo
import nemo.collections.asr as nemo_asr
import soundfile as sf
from scipy.io import wavfile
import scipy.signal as sps
from pyannote.metrics.diarization import DiarizationErrorRate
import sklearn.metrics.pairwise as pw
from scipy.optimize import linear_sum_assignment
import librosa
import ipdb
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
### From speaker_diarize.py
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from nemo.collections.asr.parts.utils.streaming_utils import longest_common_subsequence_merge as lcs_alg
# from nemo.collections.asr.models.classification_models import EncDecClassificationModel
from nemo.collections.asr.parts.utils.decoder_timestamps_utils import FrameBatchASR_Logits, WERBPE_TS, ASR_TIMESTAMPS, WER_TS
from nemo.collections.asr.data.audio_to_label import repeat_signal
from nemo.collections.asr.parts.utils.diarization_utils import ASR_DIAR_OFFLINE, write_txt
from nemo.collections.asr.parts.utils.speaker_utils import audio_rttm_map
from nemo.collections.asr.parts.utils.speaker_utils import get_contiguous_stamps, merge_stamps, labels_to_pyannote_object, rttm_to_labels, labels_to_rttmfile, get_uniqname_from_filepath, get_embs_and_timestamps, get_subsegments, isOverlap, getOverlapRange, getMergedRanges, getSubRangeList, fl2int, int2fl, combine_int_overlaps
from nemo.collections.asr.parts.utils.vad_utils import (
    generate_overlap_vad_seq,
    generate_vad_segment_table,
    get_vad_stream_status,
    prepare_manifest,
)
from nemo.collections import nlp as nemo_nlp
from typing import Dict, List, Tuple, Type, Union
from nemo.collections.asr.models import ClusteringDiarizer, EncDecCTCModel, EncDecCTCModelBPE
from sklearn.preprocessing import OneHotEncoder
from nemo.collections.asr.parts.utils.streaming_utils import AudioFeatureIterator, FrameBatchASR
from nemo.collections.asr.parts.utils.nmesc_clustering import (
    NMESC,
    SpectralClustering,
    getEnhancedSpeakerCount,
    COSclustering,
    getCosAffinityMatrix,
    getAffinityGraphMat,
    getMultiScaleCosAffinityMatrix
)
from nemo.collections.asr.parts.utils.nmesc_clustering import COSclustering
from nemo.collections.asr.parts.utils.streaming_utils import get_samples, AudioFeatureIterator, FrameBatchVAD

from nemo.core.config import hydra_runner
from nemo.utils import logging
import hydra
from typing import List, Optional, Dict
from omegaconf import DictConfig, OmegaConf, open_dict
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf
import copy
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
from nemo.utils import logging, model_utils
import torch
from torch.utils.data import DataLoader
import math

from collections import Counter
from functools import reduce
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

# For streaming ASR
from nemo.core.classes import IterableDataset
from torch.utils.data import DataLoader
import math
import difflib
from sklearn.manifold import TSNE
from nemo.core.classes import IterableDataset


TOKEN_OFFSET = 100

import contextlib
import json
import os

import editdistance
from sklearn.model_selection import ParameterGrid

import nemo
import nemo.collections.asr as nemo_asr

from nemo.utils import logging

from pyctcdecode import build_ctcdecoder

# GRadio input
import gradio as gr
import torch
import time
import librosa
import soundfile
import nemo.collections.asr as nemo_asr
import tempfile
import os
import uuid
import time
import time
import numpy as np
seed_everything(42)

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            logging.info('%2.2fms %r'%((te - ts) * 1000, method.__name__))
            pass
        return result
    return timed

def process_audio_file(file):
    TARGET_SR = 16000
    data, sr = librosa.load(file, sr=TARGET_SR)
    os.remove(file)
    data = librosa.to_mono(data)
    return data

def get_partial_ref_labels(pred_labels, ref_labels):
    last_pred_time = float(pred_labels[-1].split()[1])
    ref_labels_out = []
    for label in ref_labels:
        start, end, speaker = label.split()
        start, end = float(start), float(end)
        if last_pred_time <= start:
            pass
        elif start < last_pred_time <= end:
            label = f"{start} {last_pred_time} {speaker}"
            ref_labels_out.append(label) 
        elif end < last_pred_time:
            ref_labels_out.append(label) 
    return ref_labels_out 


def get_DER(all_reference, all_hypothesis, collar=0.5, skip_overlap=True):
    """
    """
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap, uem=None)

    mapping_dict = {}
    for k, (reference, hypothesis) in enumerate(zip(all_reference, all_hypothesis)):
        metric(reference, hypothesis, detailed=True)
        mapping_dict[k] = metric.optimal_mapping(reference, hypothesis)

    DER = abs(metric)
    CER = metric['confusion'] / metric['total']
    FA = metric['false alarm'] / metric['total']
    MISS = metric['missed detection'] / metric['total']

    metric.reset()

    return DER, CER, FA, MISS, mapping_dict

def get_wer_feat_logit_single(samples, frame_asr, frame_len, tokens_per_chunk, delay, model_stride_in_secs, frame_mask):
    """
    Create a preprocessor to convert audio samples into raw features,
    Normalization will be done per buffer in frame_bufferer.
    """
    hyps, tokens_list = [], []
    frame_asr.reset()
    feature_frame_shape = frame_asr.read_audio_file_and_return_samples(samples, delay, model_stride_in_secs, frame_mask)
    if frame_mask is not None:
        hyp, tokens, _ = frame_asr.transcribe_with_ts(tokens_per_chunk, delay)
    else:
        hyp, tokens = None, None
    hyps.append(hyp)
    tokens_list.append(tokens)
    return hyps, tokens_list, feature_frame_shape

def get_vad_feat_logit_single(samples, frame_vad, frame_len, tokens_per_chunk, vad_delay, model_stride_in_secs, threshold):
    """
    Create a preprocessor to convert audio samples into raw features,
    Normalization will be done per buffer in frame_bufferer.
    """
    hyps, tokens_list = [], []
    frame_vad.reset()
    feature_frame_shape = frame_vad.read_audio_file_and_return_samples(samples, delay=vad_delay, model_stride_in_secs=model_stride_in_secs)
    streaming_vad_logits, speech_segments = frame_vad.decode(threshold=threshold)
    return streaming_vad_logits, speech_segments, feature_frame_shape

class MaskedFeatureIterator(AudioFeatureIterator):
    def __init__(self, samples, frame_len, preprocessor, device, frame_mask=None):
        super().__init__(samples, frame_len, preprocessor, device)
        if frame_mask is not None:
            self._features = torch.log(torch.mul(np.exp(1) ** self._features, frame_mask.to(device)))

class FrameBatchASR_Logits_Sample(FrameBatchASR_Logits):
    """
    A class for streaming frame-based ASR.
    Inherits from FrameBatchASR and adds new capability of returning the logit output.
    Please refer to FrameBatchASR for more detailed information.
    """

    def __init__(self, asr_model, frame_len=1.0, total_buffer=4.0, batch_size=4):
        super().__init__(asr_model, frame_len, total_buffer, batch_size)
    
    @timeit
    def read_audio_file_and_return_samples(self, _samples, delay: float, model_stride_in_secs: float, frame_mask):
        self.device = self.asr_model.device
        samples = np.pad(_samples, (0, int(delay * model_stride_in_secs * self.asr_model._cfg.sample_rate)))
        frame_reader = MaskedFeatureIterator(samples, self.frame_len, self.raw_preprocessor, self.device, frame_mask)
        self.set_frame_reader(frame_reader)
        return frame_reader._features.shape

class FrameBatchVAD_sample(FrameBatchVAD):
    """
    A class for streaming frame-based ASR.
    Inherits from FrameBatchASR and adds new capability of returning the logit output.
    Please refer to FrameBatchASR for more detailed information.
    """

    def __init__(self, vad_model, frame_len, total_buffer, batch_size, patience, total_asr_buffer):
        super().__init__(vad_model, frame_len, total_buffer, batch_size, patience, total_asr_buffer)
    
    @timeit
    def read_audio_file_and_return_samples(self, samples, delay: float, model_stride_in_secs: float):
        self.device = self.vad_model.device
        self.pad_end_len = int(delay * model_stride_in_secs * self.vad_model._cfg.sample_rate)
        samples = np.pad(samples, (0, self.pad_end_len))
        frame_reader = AudioFeatureIterator(samples, self.frame_len, self.raw_preprocessor, self.vad_model.device)
        self.set_frame_reader(frame_reader)
        return frame_reader._features.shape

class OnlineDiarizer(ClusteringDiarizer, ASR_DIAR_OFFLINE):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        
        # Convert config to support Hydra 1.0+ instantiation
        self.uniq_id = None
        self.AUDIO_RTTM_MAP = None
        self.sample_rate = cfg.sample_rate
        cfg = model_utils.maybe_update_config_version(cfg)
        self._cfg = cfg
        self.device =  cfg.diarizer.device
        self._out_dir = self._cfg.diarizer.out_dir
        if not os.path.exists(self._out_dir):
            os.mkdir(self._out_dir)
        self.base_scale_index = max(self.multiscale_args_dict['scale_dict'].keys())
        
        torch.manual_seed(0)
        self.memory_segment_ranges= []
        self.memory_cluster_labels = []
        self.memory_segment_indexes = []
        self.memory_margin = 5
        self._speaker_model.to(self.device)
        self._speaker_model.eval()
        self.paths2session_audio_files = []
        self.all_hypothesis = []
        self.all_reference = []
        self.out_rttm_dir = None

        self.ROUND = 2
        self.embed_seg_len = self._cfg.diarizer.speaker_embeddings.parameters.window_length_in_sec
        self.embed_seg_hop = self._cfg.diarizer.speaker_embeddings.parameters.shift_length_in_sec
        self.max_num_speakers = 8
        self.current_n = 0
        self.history_n = 0
        self.MINIMUM_CLUS_BUFFER_SIZE = 10
        self.MINIMUM_HIST_BUFFER_SIZE = 32
        self._minimum_segments_per_buffer = int(self.history_n/self.max_num_speakers)
        self.Y_fullhist = []
        self.p_value_skip_frame_thres = 50
        self.p_value_update_frequency = 10
        
        self.history_embedding_buffer_emb = np.array([])
        self.history_embedding_buffer_label = np.array([])
        self.history_buffer_seg_end = None
        self.frame_index = None
        self.index_dict = {'max_embed_count': 0}
        self.p_value_hist = []

        self.history_buffer_size = self._cfg.diarizer.clustering.parameters.history_buffer_size
        self.current_buffer_size = self._cfg.diarizer.clustering.parameters.current_buffer_size
        self.max_num_speaker = self._cfg.diarizer.clustering.parameters.max_num_speakers
        self.num_spk_stat = []
        self.min_spk_counting_buffer_size = 3
        self.min_segment_per_spk = 10
        self.p_value_queue_size = 2
        self.oracle_num_speakers = None

        self.diar_eval_count = 0
        self.DER_csv_list = []
        self.der_dict = {}
        self.der_stat_dict = {"avg_DER":0, "avg_CER":0, "max_DER":0, "max_CER":0, "cum_DER":0, "cum_CER":0}
        self.color_palette = {'speaker_0': '\033[1;32m',
                              'speaker_1': '\033[1;34m',
                              'speaker_2': '\033[1;30m',
                              'speaker_3': '\033[1;35m',
                              'speaker_4': '\033[1;31m',
                              'speaker_5': '\033[1;36m',
                              'speaker_6': '\033[1;37m',
                              'speaker_7': '\033[1;30m',
                              'speaker_8': '\033[1;33m',
                              'speaker_9': '\033[0;34m',
                              'white': '\033[0;37m'}
   
    def _init_segment_variables(self):
        self.embs_array = {self.uniq_id : {} }
        self.time_stamps = {self.uniq_id : {} }
        self.segment_abs_time_range_list = {self.uniq_id: {} }
        self.segment_raw_audio_list = {self.uniq_id: {} }
        self.segment_indexes = {self.uniq_id: {} }

        for scale_idx, (window, shift) in self.multiscale_args_dict['scale_dict'].items():
            self.multiscale_embeddings_and_timestamps[scale_idx] = [None, None]
            self.embs_array[self.uniq_id][scale_idx] = None
            self.time_stamps[self.uniq_id][scale_idx] = []
            self.segment_abs_time_range_list[self.uniq_id][scale_idx] = []
            self.segment_raw_audio_list[self.uniq_id][scale_idx] = []
            self.segment_indexes[self.uniq_id][scale_idx] = []

    @property 
    def history_buffer_size(self, value):
        return self.current_n

    @history_buffer_size.setter
    def history_buffer_size(self, value):
        logging.info(f"Setting online diarization buffer to : {value}")
        assert value >= self.MINIMUM_CLUS_BUFFER_SIZE, f"Online diarization clustering buffer should be bigger than {self.MINIMUM_CLUS_BUFFER_SIZE}"
        self.current_n = value # How many segments we want to use as clustering buffer
    
    @property 
    def current_buffer_size(self, value):
        return self.current_n

    @current_buffer_size.setter
    def current_buffer_size(self, value):
        logging.info(f"Setting online diarization buffer to : {value}")
        assert value >= self.MINIMUM_HIST_BUFFER_SIZE, f"Online diarization history buffer should be bigger than {self.MINIMUM_HIST_BUFFER_SIZE}"
        self.history_n = value # How many segments we want to use as history buffer

    def getMergeQuantity(self, new_emb_n, before_cluster_labels):
        """
        Determine which embeddings we need to reduce or merge in history buffer.
        We want to merge or remove the embedding in the bigger cluster first.
        At the same time, we keep the minimum number of embedding per cluster
        with the variable named self._minimum_segments_per_buffer.
        The while loop creates a numpy array emb_n_per_cluster.
        that tells us how many embeddings we should remove/merge per cluster.

        Args:
            new_emb_n: (int)
                the quantity of the newly obtained embedding from the new stream of input.

            before_cluster_labels: (np.array)
                the speaker labels of (the history_embedding_buffer_emb) + (the new embeddings to be added)
        """
        targeted_total_n = new_emb_n
        count_dict = Counter(before_cluster_labels)
        spk_freq_count = np.bincount(before_cluster_labels)
        class_vol = copy.deepcopy(spk_freq_count)
        emb_n_per_cluster = np.zeros_like(class_vol).astype(int)
        arg_max_spk_freq = np.argsort(spk_freq_count)[::-1]
        count = 0
        while np.sum(emb_n_per_cluster) < new_emb_n:
            recurr_idx = np.mod(count, len(count_dict))
            curr_idx = arg_max_spk_freq[recurr_idx]
            margin = (spk_freq_count[curr_idx] - emb_n_per_cluster[curr_idx]) - self._minimum_segments_per_buffer
            if margin > 0:
                target_number = min(margin, new_emb_n)
                emb_n_per_cluster[curr_idx] += target_number
                new_emb_n -= target_number
            count += 1
        assert sum(emb_n_per_cluster) == targeted_total_n, "emb_n_per_cluster does not match with targeted number new_emb_n."
        return emb_n_per_cluster

    def run_reduction_alg(self, cmat, tick2d, emb_ndx, cluster_labels):
        """


        """
        LI, RI = tick2d[0, :], tick2d[1, :]
        LI_argdx = tick2d[0].argsort()
        LI, RI = LI[LI_argdx], RI[LI_argdx]
        result_emb = 0.5*(emb_ndx[LI, :] + emb_ndx[RI, :])
        merged_cluster_labels = cluster_labels[np.array(list(set(LI)))]
        bypass_ndx = np.array(list(set(range(emb_ndx.shape[0])) - set(list(LI)+list(RI)) ) )
        if len(bypass_ndx) > 0:
            result_emb = np.vstack((emb_ndx[bypass_ndx], result_emb))  
            merged_cluster_labels = np.hstack((cluster_labels[bypass_ndx], merged_cluster_labels))
        return result_emb, merged_cluster_labels

    def prepare_embedding_update(self, emb_in):
        """
        Case-1
            This else statement is for the very first diarization loop.
            This is the very first reduction frame.
        Case-2
            If the number of embeddings is decreased compared to the last trial,
            then skip embedding merging.
        Case-3
            Since there are new embeddings, we push the same amount (new_emb_n)
            of old embeddings to the history buffer.
            We should also update self.history_buffer_seg_end which is a pointer.

        """
        segment_indexes_mat = np.array(self.segment_indexes[self.uniq_id][0]).astype(int)
        self.total_segments_processed_count = segment_indexes_mat[-1] + 1
        print("total_segments_processed_count: ", self.total_segments_processed_count)
        history_n, current_n = self.history_n, self.current_n
        update_speaker_register = True
        
        # Case-1
        if len(self.history_embedding_buffer_emb) == 0:
            hist_curr_boundary = self.total_segments_processed_count - self.current_n
            new_emb_n = self.total_segments_processed_count - (self.current_n + self.history_n)
            hist_curr_boundary_emb_idx = np.where(segment_indexes_mat==hist_curr_boundary)[0][0]
            emb_hist= emb_in[:hist_curr_boundary_emb_idx]
            self.before_cluster_labels = self.Y_fullhist[:hist_curr_boundary]
            self.history_buffer_seg_end = hist_curr_boundary
        else: 
            # Case-2 
            if self.total_segments_processed_count <= self.index_dict['max_embed_count']:
                hist_curr_boundary = self.history_buffer_seg_end
                new_emb_n, emb_hist = None, None
                update_speaker_register = False
            # Case-3
            else:
                hist_curr_boundary = self.total_segments_processed_count - self.current_n
                _stt, _end = self.history_buffer_seg_end, hist_curr_boundary
                new_emb_n = _end - _stt
                assert new_emb_n > 0, "new_emb_n should be a positve integer number."
                emb_idx_stt, emb_idx_end = np.where(segment_indexes_mat == _stt)[0][0], np.where(segment_indexes_mat==_end)[0][0]
                update_to_history_emb = emb_in[emb_idx_stt:emb_idx_end]
                update_to_history_label = self.Y_fullhist[_stt:_end]
                emb_hist = np.vstack((self.history_embedding_buffer_emb, update_to_history_emb))
                self.before_cluster_labels = np.hstack((self.history_embedding_buffer_label, update_to_history_label))
                self.history_buffer_seg_end = hist_curr_boundary
        print(f"hist_curr_boundary: {hist_curr_boundary}")
        print(f"self.history_buffer_seg_end: {self.history_buffer_seg_end}")
        return update_speaker_register, new_emb_n, emb_hist
       
    def make_constant_length_emb(self, emb_in):
        """
        Edge case when the number of segments decreases and the number of embedding falls short for the labels.
        ASR decoder occasionally returns less number of words compared to the previous frame. In this case,
        we obtain fewer embedding vectors for the short period of time. To match the pre-defined length, yhe last
        embedding vector is repeated to fill the voidness. The repeated embedding will be soon replaced by the actual
        embeddings once the system takes new frames.
        """
        segment_indexes_mat = np.array(self.segment_indexes[self.uniq_id][0]).astype(int)
        curr_clustered_segments =  np.where(segment_indexes_mat >= self.history_buffer_seg_end)[0]
        if emb_in[curr_clustered_segments].shape[0] < self.current_n:
            delta_count = self.current_n - emb_in[curr_clustered_segments].shape[0]
            fill_in_emb = np.tile(emb_in[curr_clustered_segments][-1], (delta_count,1))
            emb_curr = np.vstack((emb_in[curr_clustered_segments], fill_in_emb))
        else:
            emb_curr = emb_in[curr_clustered_segments]
        return emb_curr

    def run_reducer(self, emb_hist, mat, spk_idx, target_num):
        """

        """
        ndx = np.where(self.before_cluster_labels == spk_idx)[0]
        if target_num > 0:
            cmat = np.tril(mat[:,ndx][ndx,:])
            tick2d = self.getIndecesForEmbeddingReduction(cmat, ndx, target_num)
            spk_cluster_labels, emb_ndx = self.before_cluster_labels[ndx], emb_hist[ndx]
            result_emb, merged_cluster_labels = self.run_reduction_alg(cmat, tick2d, emb_ndx, spk_cluster_labels)
            assert (ndx.shape[0] - target_num) == result_emb.shape[0], ipdb.set_trace()
        else:
            result_emb = emb_hist[ndx]
            merged_cluster_labels = self.before_cluster_labels[ndx]
        return result_emb, merged_cluster_labels

    def reduce_embedding_sets(self, emb_in, mat):
        """

        Example:
            self.history_n = 10
            self.current_n = 20

        Step (1)
        |-----------|XXYYZZ--------------|

        If we get two more segments, "NN" as in the description:
        history buffer = 10
        current buffer = 22

        Step (2)
        |-----------|XXYYZZ--------------NN|

        The newly accepted embeddings go through a queue (first embedding, first merged)
        history buffer = 12
        current buffer = 20

        Step (3)
        |-----------XX|YYZZ--------------NN|

        After merging (reducing) the embedding set:
        history buffer = 10
        current buffer = 20

        Step(3)
        |-----------|------------------NN|

        After clustering:

        |00000111111|11110000110010010011|

        This label is self.Y_fullhist (shape is (history_n + current_n) )

        self.history_buffer_seg_end (int):
            The total number of segments that have been merged from the beginning of the session.
            (=hist_curr_boundary)

        """
        history_n, current_n = self.history_n, self.current_n
        update_speaker_register, new_emb_n, emb_hist = self.prepare_embedding_update(emb_in)
       
        # Update the history/current_buffer boundary cursor
        total_emb, total_cluster_labels = [], []
        
        if update_speaker_register:
            class_target_vol = self.getMergeQuantity(new_emb_n, self.before_cluster_labels)
            # Merge the segments in the history buffer
            for spk_idx, target_num in enumerate(list(class_target_vol)):
                result_emb, merged_cluster_labels = self.run_reducer(emb_hist, mat, spk_idx, target_num)
                total_emb.append(result_emb)
                total_cluster_labels.append(merged_cluster_labels)
            self.history_embedding_buffer_emb = np.vstack(total_emb)
            self.history_embedding_buffer_label = np.hstack(total_cluster_labels)
            assert self.history_embedding_buffer_emb.shape[0] == self.history_n, f"History embedding size is not maintained correctly."
        else:
            total_emb.append(self.history_embedding_buffer_emb)
            total_cluster_labels.append(self.history_embedding_buffer_label)

        emb_curr = self.make_constant_length_emb(emb_in)
        total_emb.append(emb_curr)
        
        # Before perform clustering, we attach the current_n number of estimated speaker labels 
        # from the previous clustering result.
        total_cluster_labels.append(self.Y_fullhist[-current_n:])

        history_and_current_emb = np.vstack(total_emb)
        history_and_current_labels = np.hstack(total_cluster_labels)
        assert history_and_current_emb.shape[0] == len(history_and_current_labels)

        self.index_dict['max_embed_count'] = max(self.total_segments_processed_count, self.index_dict['max_embed_count'])
        return history_and_current_emb, history_and_current_labels, current_n, update_speaker_register
    
    def getIndecesForEmbeddingReduction(self, cmat, ndx, target_num):
        """
        Get indeces of the embeddings we want to merge or drop.

        Args:
            cmat: (np.array)
            ndx: (np.array)
            target_num: (int)

        Output:
            tick2d: (numpy.array)
        """
        comb_limit = int(ndx.shape[0]/2)
        assert target_num <= comb_limit, f" target_num is {target_num}: {target_num} is bigger than comb_limit {comb_limit}"
        idx2d = np.unravel_index(np.argsort(cmat, axis=None)[::-1], cmat.shape)
        num_of_lower_half = int((cmat.shape[0]**2 - cmat.shape[0])/2)
        idx2d = (idx2d[0][:num_of_lower_half], idx2d[1][:num_of_lower_half])
        cdx, left_set, right_set, total_set = 0, [], [], []
        while len(left_set) <  target_num and len(right_set) < target_num:
            Ldx, Rdx = idx2d[0][cdx], idx2d[1][cdx] 
            if (not Ldx in total_set) and (not Rdx in total_set):
                left_set.append(Ldx)
                right_set.append(Rdx)
                total_set = left_set + right_set
            cdx += 1
        tick2d = np.array([left_set, right_set])
        return tick2d
    
    @timeit
    def getReducedMat(self, mat, emb):
        """
        Choose whether we want to add embeddings to the memory or not.
        """
        margin_seg_n = mat.shape[0] - (self.current_n + self.history_n)
        if margin_seg_n > 0:
            self.isOnline = True
            mat = 0.5*(mat + mat.T)
            np.fill_diagonal(mat, 0)
            merged_emb, cluster_labels, current_n, add_new = self.reduce_embedding_sets(emb, mat)
            assert merged_emb.shape[0] == len(cluster_labels)
        else:
            self.isOnline = False
            merged_emb = emb
            current_n = self.current_n
            cluster_labels, add_new = None, True
        return merged_emb, cluster_labels, add_new
    
    def online_eval_diarization(self, pred_labels, rttm_file, ROUND=2):
        pred_diar_labels, ref_labels_list = [], []
        all_hypotheses, all_references = [], []

        if os.path.exists(rttm_file):
            ref_labels_total = rttm_to_labels(rttm_file)
            ref_labels = get_partial_ref_labels(pred_labels, ref_labels_total)
            reference = labels_to_pyannote_object(ref_labels)
            all_references.append(reference)
        else:
            raise ValueError("No reference RTTM file provided.")

        pred_diar_labels.append(pred_labels)

        self.der_stat_dict['ref_n_spk'] = self.get_num_of_spk_from_labels(ref_labels)
        self.der_stat_dict['est_n_spk'] = self.get_num_of_spk_from_labels(pred_labels)
        hypothesis = labels_to_pyannote_object(pred_labels)
        if ref_labels == [] and pred_labels != []:
            logging.info(
                "Streaming Diar [{}][frame-  {}th  ]:".format(
                    self.uniq_id, self.frame_index
                )
            )
            DER, CER, FA, MISS = 100.0, 0.0, 100.0, 0.0
            der_dict, der_stat_dict = self.get_online_DER_stats(DER, CER, FA, MISS)
            return der_dict, der_stat_dict
        else:
            all_hypotheses.append(hypothesis)
            try:
                DER, CER, FA, MISS, _= get_DER(all_references, all_hypotheses, collar=0.25, skip_overlap=True)
            except:
                DER, CER, FA, MISS = 100.0, 0.0, 100.0, 0.0
            logging.info(
                "Streaming Diar [{}][frame-    {}th    ]: DER:{:.4f} MISS:{:.4f} FA:{:.4f}, CER:{:.4f}".format(
                    self.uniq_id, self.frame_index, DER, MISS, FA, CER
                )
            )

            der_dict, der_stat_dict = self.get_online_DER_stats(DER, CER, FA, MISS)
            return der_dict, der_stat_dict
    
    def get_online_DER_stats(self, DER, CER, FA, MISS):
        der_dict = {"DER": round(100*DER, self.ROUND), 
                    "CER": round(100*CER, self.ROUND), 
                    "FA":  round(100*FA, self.ROUND), 
                    "MISS": round(100*MISS, self.ROUND)}
        self.diar_eval_count += 1
        self.der_stat_dict['cum_DER'] += DER
        self.der_stat_dict['cum_CER'] += CER
        self.der_stat_dict['avg_DER'] = round(100*self.der_stat_dict['cum_DER']/self.diar_eval_count, self.ROUND)
        self.der_stat_dict['avg_CER'] = round(100*self.der_stat_dict['cum_CER']/self.diar_eval_count, self.ROUND)
        self.der_stat_dict['max_DER'] = round(max(der_dict['DER'], self.der_stat_dict['max_DER']), self.ROUND)
        self.der_stat_dict['max_CER'] = round(max(der_dict['CER'], self.der_stat_dict['max_CER']), self.ROUND)
        return der_dict, self.der_stat_dict


    def print_time_colored(self, string_out, speaker, start_point, end_point, params, replace_time=False, space=' '):
        params['color'] == False
        if params['color']:
            color = self.color_palette[speaker]
        else:
            color = ''

        datetime_offset = 16 * 3600
        if float(start_point) > 3600:
            time_str = "%H:%M:%S.%f"
        else:
            time_str = "%M:%S.%f"
        start_point_str = datetime.fromtimestamp(float(start_point) - datetime_offset).strftime(time_str)[:-4]
        end_point_str = datetime.fromtimestamp(float(end_point) - datetime_offset).strftime(time_str)[:-4]
        
        if replace_time:
            old_start_point_str = string_out.split('\n')[-1].split(' - ')[0].split('[')[-1]
            word_sequence = string_out.split('\n')[-1].split(' - ')[-1].split(':')[-1].strip() + space
            string_out = '\n'.join(string_out.split('\n')[:-1])
            time_str = "[{} - {}]".format(old_start_point_str, end_point_str)
        else:
            time_str = "[{} - {}]".format(start_point_str, end_point_str)
            word_sequence = ''
        
        if not params['print_time']:
            time_str = ''
        
        strd = "\n{}{} {}: {}".format(color, time_str, speaker, word_sequence)
        return string_out + strd
    
    @staticmethod
    def print_word_colored(string_out, word, params, space=" "):
        word = word.strip()
        return string_out + " " +  word
    
    def OnlineCOSclustering(
        self,
        uniq_embs_and_timestamps,
        oracle_num_speakers=None,
        max_num_speaker: int = 8,
        min_samples_for_NMESC: int = 25,
        enhanced_count_thres: int = 50,
        max_rp_threshold: float = 0.25,
        sparse_search_volume: int= 15,
        fixed_thres: float = 0.0,
        cuda=False,
    ):
        """
        Clustering method for speaker diarization based on cosine similarity.
        NME-SC part is converted to torch.tensor based operations in NeMo 1.9.

        Args:
            uniq_embs_and_timestamps: (dict)
                The dictionary containing embeddings, timestamps and multiscale weights.
                If uniq_embs_and_timestamps contains only one scale, single scale diarization
                is performed.

            oracle_num_speaker: (int or None)
                The oracle number of speakers if known else None

            max_num_speaker: (int)
                The maximum number of clusters to consider for each session

            min_samples_for_NMESC: (int)
                The minimum number of samples required for NME clustering. This avoids
                zero p_neighbour_lists. If the input has fewer segments than min_samples,
                it is directed to the enhanced speaker counting mode.

            enhanced_count_thres: (int)
                For the short audio recordings under 60 seconds, clustering algorithm cannot
                accumulate enough amount of speaker profile for each cluster.
                Thus, getEnhancedSpeakerCount() employs anchor embeddings (dummy representations)
                to mitigate the effect of cluster sparsity.
                enhanced_count_thres = 80 is recommended.

            max_rp_threshold: (float)
                Limits the range of parameter search.
                Clustering performance can vary depending on this range.
                Default is 0.15.

            sparse_search_volume: (int)
                Number of p_values we search during NME analysis.
                Default is 30. The lower the value, the faster NME-analysis becomes.
                Lower than 20 might cause a poor parameter estimation.

            fixed_thres: (float)
                If fixed_thres value is provided, NME-analysis process will be skipped.
                This value should be optimized on a development set to obtain a quality result.
                Default is None and performs NME-analysis to estimate the threshold.

        Returns:
            Y: (torch.tensor[int])
                Speaker label for each segment.
        """
        device = torch.device("cuda") if cuda else torch.device("cpu")

        # Get base-scale (the highest index) information from uniq_embs_and_timestamps.
        uniq_scale_dict = uniq_embs_and_timestamps['scale_dict']
        _emb = uniq_scale_dict[max(uniq_scale_dict.keys())]['embeddings']
      
        _mat = getCosAffinityMatrix(_emb)
        org_mat = copy.deepcopy(_mat)
        
        _emb, _mat = _emb.cpu().numpy(), _mat.cpu().numpy()
        emb, reduced_labels, add_new = self.getReducedMat(_mat, _emb)
        emb = torch.tensor(emb).to(device)
        mat = getCosAffinityMatrix(emb)

        self.index_dict[self.frame_index] = (org_mat.shape[0], self.history_buffer_seg_end)

        if emb.shape[0] == 1:
            return torch.zeros((1,), dtype=torch.int32)

        if oracle_num_speakers:
            max_num_speaker = oracle_num_speakers

        nmesc = NMESC(
            mat,
            max_num_speaker=max_num_speaker,
            max_rp_threshold=max_rp_threshold,
            sparse_search=True,
            sparse_search_volume=20,
            fixed_thres=fixed_thres,
            NME_mat_size=256,
            min_samples_for_NMESC=min_samples_for_NMESC,
            use_mode_est_num_spk=True,
            device=device,
        )
        if emb.shape[0] < min_samples_for_NMESC:
            nmesc.max_rp_threshold = 0.5
        est_num_of_spk, p_hat_value = self.onlineNMEanalysis(nmesc) 

        affinity_mat = getAffinityGraphMat(mat, p_hat_value)

        if oracle_num_speakers:
            est_num_of_spk = oracle_num_speakers

        est_num_of_spk = min(est_num_of_spk, int(1+ emb.shape[0]/self.min_segment_per_spk))
        self.num_spk_stat.append(est_num_of_spk)
        spk_counting_buffer_size = max(max(self.num_spk_stat), self.min_spk_counting_buffer_size)
        if len(self.num_spk_stat) > spk_counting_buffer_size:
            self.num_spk_stat.pop(0)
        num_spks_bincount = torch.bincount(torch.tensor(self.num_spk_stat))
        print("num _spks bincount:", num_spks_bincount[1:])
        print("spk_counting_buffer_size:", spk_counting_buffer_size)
        print("emb ssize: ", emb.shape, int(1+emb.shape[0]/self.min_segment_per_spk))
        maj_est_num_of_spk = torch.argmax(num_spks_bincount)

        spectral_model = SpectralClustering(n_clusters=maj_est_num_of_spk, cuda=cuda, device=device)
        Y = spectral_model.predict(affinity_mat)
        Y = Y.cpu().numpy()
        Y_out = self.matchLabels(org_mat, Y, add_new)
        return Y_out
    
    def onlineNMEanalysis(self, nmesc):
        """
        To save the running time, the p-value is only estimated in the beginning of the session.
        After switching to online mode, the system uses the most common estimated p-value.
        Estimating p-value requires a plenty of computational resource. The less frequent estimation of
        p-value can speed up the clustering algorithm by a huge margin.
        Args:
            nmesc: (NMESC)
                nmesc instance.
            isOnline: (bool)
                Indicates whether the system is running on online mode or not.

        Returns:
            est_num_of_spk: (int)
                The estimated number of speakers.
            p_hat_value: (int)
                The estimated p-value from NMESC method.
        """
        if self.frame_index > self.p_value_skip_frame_thres:
            if len(self.p_value_hist)  == 0 or self.frame_index % self.p_value_update_frequency == 0:
                est_num_of_spk, p_hat_value = nmesc.NMEanalysis()
                self.p_value_hist.append(p_hat_value)
                if len(self.p_value_hist) > self.p_value_queue_size:
                    self.p_value_hist.pop(0)
            p_hat_value =  max(self.p_value_hist, key = self.p_value_hist.count)
            est_num_of_spk, g_p = nmesc.getEigRatio(p_hat_value)
            p_value_stat = torch.bincount(torch.tensor(self.p_value_hist))
            print(f"p_value_stat: ", p_value_stat)
        else:
            est_num_of_spk, p_hat_value = nmesc.NMEanalysis()
        return est_num_of_spk, p_hat_value
    
    def matchLabels(self, org_mat, Y, add_new):
        """
        self.history_buffer_seg_end is a timestamp that tells to which point is history embedding contains from self.Y_fullhist.
        If embedding reducing is done correctly, we should discard  (0, self.history_n) amount and take
        (self.history_n, len(Y) ) from the new clustering output Y.

        """
        if self.isOnline:
            # Online clustering mode with history buffer
            y_new_update_start = self.history_n
            Y_matched = self.matchNewOldclusterLabels(self.Y_fullhist[self.history_buffer_seg_end:], Y, with_history=True)
            if add_new:
                assert Y_matched[y_new_update_start:].shape[0] == self.current_n, "Update point sync is not correct."
                Y_out = np.hstack((self.Y_fullhist[:self.history_buffer_seg_end], Y_matched[y_new_update_start:]))
                self.Y_fullhist = Y_out
            else:
                # Do not update cumulative labels since there are no new segments.
                Y_out = self.Y_fullhist[:org_mat.shape[0]]
        else:
            # If no memory is used, offline clustering is applied.
            Y_out = self.matchNewOldclusterLabels(self.Y_fullhist, Y, with_history=False)
            self.Y_fullhist = Y_out
        return Y_out
    
    def process_cluster_labels(self, audio_signal_list, embs_array, new_segment_ranges, segment_indexes, new_cluster_labels):
        """
        Clustering is done for (hist_N + curr_N) number of embeddings. Thus, we need to remove the clustering results on
        the embedding memory. If self.diar.history_buffer_seg_end is not None, that indicates streaming diarization system
        is starting to save embeddings to its memory. Thus, the new incoming clustering label should be separated.
        """
        keep_range = self.history_n  + self.current_n + self.memory_margin
        new_cluster_labels = new_cluster_labels.tolist() 
        if not self.isOnline:
            self.memory_segment_ranges = copy.deepcopy(new_segment_ranges)
            self.memory_segment_indexes = copy.deepcopy(segment_indexes)
            self.memory_cluster_labels = copy.deepcopy(new_cluster_labels)
        
        elif segment_indexes[-1] > self.memory_segment_indexes[-1]:
            segment_indexes_mat = np.array(segment_indexes).astype(int)
            existing_max = max(self.memory_segment_indexes)
            update_idx = existing_max - self.memory_margin
            mp_idx = np.where(segment_indexes_mat == update_idx)[0][0]
            self.memory_segment_ranges[update_idx:] = copy.deepcopy(new_segment_ranges[mp_idx:])
            self.memory_segment_indexes[update_idx:] = copy.deepcopy(segment_indexes[mp_idx:])
            self.memory_cluster_labels[update_idx:] = copy.deepcopy(new_cluster_labels[update_idx:])
            assert len(self.memory_cluster_labels) == len(self.memory_segment_ranges)

            # Remove unnecessary values
            embs_array = embs_array[-keep_range:]
            audio_signal_list = audio_signal_list[-keep_range:]
            new_segment_ranges = new_segment_ranges[-keep_range:]
            segment_indexes = segment_indexes[-keep_range:]

        assert len(self.memory_cluster_labels) == len(self.memory_segment_ranges)
        lines = []
        for idx, label in enumerate(self.memory_cluster_labels):
            tag = 'speaker_' + str(label)
            lines.append(f"{self.memory_segment_ranges[idx][0]} {self.memory_segment_ranges[idx][1]} {tag}")
        
        # cont_lines = self.get_contiguous_stamps(lines)
        cont_lines = get_contiguous_stamps(lines)
        string_labels = merge_stamps(cont_lines)
        return string_labels, embs_array, audio_signal_list,  new_segment_ranges, segment_indexes
    
    def _get_new_cursor_for_update(self, asr_diar, segment_raw_audio_list, segment_abs_time_range_list, segment_indexes):
        """
        Remove the old segments that overlap with the new frame (self.frame_start)
        cursor_for_old_segments is set to the onset of the t_range popped lastly.
        """
        cursor_for_old_segments = asr_diar.frame_start
        while True and len(segment_raw_audio_list) > 0:
            t_range = segment_abs_time_range_list[-1]

            mid = np.mean(t_range)
            if asr_diar.frame_start <= t_range[1]:
                segment_abs_time_range_list.pop()
                segment_raw_audio_list.pop()
                segment_indexes.pop()
                cursor_for_old_segments = t_range[0]
            else:
                break
        return cursor_for_old_segments
    
    def get_online_segments_from_slices(self, asr_diar, subsegments, sig, window, sigs_list, sig_rangel_list, sig_indexes, seg_index_offset):
        """
        Create short speech segments from sclices for online processing purpose.

        Args:
            slices (int): the number of slices to be created
            slice_length (int): the lenghth of each slice
            shift (int): the amount of slice window shift
            sig (FloatTensor): the tensor that contains input signal

        Returns:
            sigs_list  (list): list of sliced input signal
            audio_lengths (list): list of audio sample lengths
        """
        slice_length = int(window * asr_diar.sample_rate)
        for (start_sec, dur) in subsegments:
            if start_sec > asr_diar.buffer_end:
                continue
            if (start_sec + dur) > (asr_diar.buffer_end - asr_diar.buffer_start):
                end_sec = min(start_sec+dur, (asr_diar.buffer_end - asr_diar.buffer_start))
            else:
                end_sec = start_sec + dur
            start_idx = int(start_sec*asr_diar.sample_rate)
            end_idx = min(int(end_sec*asr_diar.sample_rate), slice_length + start_idx)
            signal = sig[start_idx:end_idx]
            if len(signal) == 0:
                raise ValueError("len(signal) is zero. Signal length should not be zero.")
            if len(signal) < slice_length:
                signal = repeat_signal(signal, len(signal), slice_length)
            sigs_list.append(signal)
            start_abs_sec = round(float(asr_diar.buffer_start + start_sec), asr_diar.ROUND)
            end_abs_sec = round(float(asr_diar.buffer_start + end_sec), asr_diar.ROUND)
            sig_rangel_list.append([start_abs_sec, end_abs_sec])
            seg_index_offset += 1
            sig_indexes.append(seg_index_offset)
        assert len(sigs_list) == len(sig_rangel_list) == len(sig_indexes)
        return seg_index_offset, sig_indexes
    
    def get_segments_from_buffer(self, asr_diar, speech_labels_for_update, source_buffer, segment_indexes, window, shift):
        sigs_list, sig_rangel_list, sig_indexes = [], [], []
        if len(segment_indexes) > 0:
            seg_index_offset = segment_indexes[-1] 
        else:
            seg_index_offset = -1

        src_len = source_buffer.shape[0] 
        for idx, range_t in enumerate(speech_labels_for_update):
            range_t = [range_t[0] - asr_diar.buffer_start, range_t[1] - asr_diar.buffer_start]
            range_t[0] = max(0, range_t[0])
            
            subsegments = get_subsegments(offset=range_t[0], window=window, shift=shift, duration=(range_t[1]-range_t[0]))
            target_sig = torch.from_numpy(source_buffer)
            seg_index_offset, sig_indexes = self.get_online_segments_from_slices(
                                                               asr_diar,
                                                               subsegments, 
                                                               target_sig,
                                                               window,
                                                               sigs_list, 
                                                               sig_rangel_list,
                                                               sig_indexes,
                                                               seg_index_offset,
                                                               )
        
        assert len(sigs_list) == len(sig_rangel_list) == len(sig_indexes)
        return sigs_list, sig_rangel_list, sig_indexes

    
    def _get_speech_labels_for_update(self, asr_diar, vad_timestamps, cursor_for_old_segments):
        """
        Bring the new speech labels from the current buffer. Then

        1. Concatenate the old speech labels from self.cumulative_speech_labels for the overlapped region.
            - This goes to new_speech_labels.
        2. Update the new 1 sec of speech label (speech_label_for_new_segments) to self.cumulative_speech_labels.
        3. Return the speech label from cursor_for_old_segments to buffer end.

        """
        if cursor_for_old_segments < asr_diar.frame_start:
            update_overlap_range = [cursor_for_old_segments, asr_diar.frame_start]
        else:
            update_overlap_range = []

        new_incoming_speech_labels = getSubRangeList(target_range=(asr_diar.frame_start, asr_diar.buffer_end),
                                                     source_range_list=vad_timestamps)

        update_overlap_speech_labels = getSubRangeList(target_range=update_overlap_range, 
                                                       source_range_list=asr_diar.cumulative_speech_labels)
        
        speech_label_for_new_segments = getMergedRanges(update_overlap_speech_labels, 
                                                             new_incoming_speech_labels) 
        
        asr_diar.cumulative_speech_labels = getMergedRanges(asr_diar.cumulative_speech_labels, 
                                                             new_incoming_speech_labels) 

        return speech_label_for_new_segments
    
    def get_diar_segments(self, asr_diar, vad_timestamps, segment_raw_audio_list, segment_abs_time_range_list, segment_indexes, window, shift):
        """
        Remove the old segments that overlap with the new frame (self.frame_start)
        cursor_for_old_segments is pointing at the onset of the t_range popped most recently.

        Frame is in the middle of the buffer.

        |___Buffer___[   Frame   ]___Buffer___|

        """
        if asr_diar.buffer_start >= 0:
            
            asr_diar._get_update_abs_time()
            
            if segment_raw_audio_list == [] and vad_timestamps != []:
                vad_timestamps[0][0] = max(vad_timestamps[0][0], 0.0)
                speech_labels_for_update = copy.deepcopy(vad_timestamps)
                asr_diar.cumulative_speech_labels = speech_labels_for_update
            
            else: 
                cursor_for_old_segments = self._get_new_cursor_for_update(asr_diar, 
                                                                          segment_raw_audio_list, 
                                                                          segment_abs_time_range_list, 
                                                                          segment_indexes)

                speech_labels_for_update = self._get_speech_labels_for_update(asr_diar, vad_timestamps,
                                                                              cursor_for_old_segments)
                
            source_buffer = copy.deepcopy(asr_diar.buffer)
            sigs_list, sig_rangel_list, sig_indexes = self.get_segments_from_buffer(asr_diar, 
                                                                                  speech_labels_for_update, 
                                                                                   source_buffer, 
                                                                                   segment_indexes,
                                                                                   window, 
                                                                                   shift)
            segment_raw_audio_list.extend(sigs_list)
            segment_abs_time_range_list.extend(sig_rangel_list)
            segment_indexes.extend(sig_indexes)

        assert len(segment_raw_audio_list) == len(segment_abs_time_range_list) == len(segment_indexes)
        return segment_raw_audio_list, \
               segment_abs_time_range_list, \
               segment_indexes

    @timeit
    def online_diarization(self, asr_diar, vad_ts):
        if asr_diar.buffer_start < 0:
            return []
        
        for scale_idx, (window, shift) in self.multiscale_args_dict['scale_dict'].items():
            self.embs_array[self.uniq_id][scale_idx] = None

            # Get subsegments for diarization.
            self.get_diar_segments(asr_diar, 
                                    vad_ts, 
                                    self.segment_raw_audio_list[self.uniq_id][scale_idx],
                                    self.segment_abs_time_range_list[self.uniq_id][scale_idx], 
                                    self.segment_indexes[self.uniq_id][scale_idx], 
                                    window, 
                                    shift)

            # Extract speaker embeddings from the subsegment timestamps.
            embeddings = asr_diar._extract_speaker_embeddings(shift, 
                                                              self.embs_array[self.uniq_id][scale_idx], 
                                                              self.segment_raw_audio_list[self.uniq_id][scale_idx], 
                                                              self.segment_abs_time_range_list[self.uniq_id][scale_idx] )
            
            embeddings = embeddings.to(self.device) 
            self.embs_array[self.uniq_id][scale_idx] = embeddings
            segment_ranges_str = [ f'{start:.3f} {end:.3f} ' for (start, end) in self.segment_abs_time_range_list[self.uniq_id][scale_idx] ]
            self.multiscale_embeddings_and_timestamps[scale_idx] = [{self.uniq_id: embeddings}, {self.uniq_id: segment_ranges_str}]
        
        embs_and_timestamps = get_embs_and_timestamps(
            self.multiscale_embeddings_and_timestamps, self.multiscale_args_dict
        )
        
        cluster_labels = self.OnlineCOSclustering(
            embs_and_timestamps[self.uniq_id], 
            oracle_num_speakers=self.oracle_num_speakers,
            max_num_speaker=self.max_num_speaker, 
            cuda=True,
        )
        truncated_data_len = len(self.segment_indexes[self.uniq_id][scale_idx])
        assert len(cluster_labels[-truncated_data_len:]) == self.embs_array[self.uniq_id][self.base_scale_index].shape[0]

        string_labels, embs_array, sigs_list, segment_ranges, segment_indexes = self.process_cluster_labels(
                                                                        self.segment_raw_audio_list[self.uniq_id][scale_idx],
                                                                        self.embs_array[self.uniq_id][scale_idx],
                                                                        self.segment_abs_time_range_list[self.uniq_id][self.base_scale_index], 
                                                                        self.segment_indexes[self.uniq_id][self.base_scale_index],
                                                                        cluster_labels)
        self.embs_array[self.uniq_id][scale_idx] = embs_array
        self.segment_raw_audio_list[self.uniq_id][scale_idx] = sigs_list
        self.segment_abs_time_range_list[self.uniq_id][scale_idx] = segment_ranges
        self.segment_indexes[self.uniq_id][scale_idx] = segment_indexes
        return string_labels
   

    
    @timeit
    def matchNewOldclusterLabels(self, Y_cumul, Y, with_history=True):
        """
        Run Hungarian algorithm (linear sum assignment) to find the best permuation mapping between
        the cumulated labels in history and the new clustering output labels.

        Args:
            Y_cumul (np.array):
                Cumulated diarization labels. This will be concatenated with history embedding speaker label
                then compared with the predicted label Y.

            Y (np.array):
                Contains predicted labels for reduced history embeddings concatenated with the predicted label.
                Permutation is not matched yet.

        Returns:
            mapping_array[Y] (np.array):
                An output numpy array where the input Y is mapped with mapping_array.

        """
        if len(Y_cumul) == 0:
            return Y
        spk_count = max(len(set(Y_cumul)), len(set(Y)))
        P_raw = np.hstack((self.history_embedding_buffer_label, Y_cumul)).astype(int)
        Q_raw = Y.astype(int)
        U_set = set(P_raw) | set(Q_raw)
        min_len = min(P_raw.shape[0], Q_raw.shape[0])
        P, Q = P_raw[:min_len], Q_raw[:min_len]
        PiQ, PuQ = (set(P) & set(Q)), (set(P) | set(Q))
        PmQ, QmP =  set(P) - set(Q),  set(Q) - set(P)
        
        if len(PiQ) == 0:
            pass
            # In len(PiQ) == 0 cas, the label is totally flipped (0<->1)
            # without any commom labels.
            # This should be differentiated from the second case.
        elif with_history and (len(PmQ) > 0 or len(QmP) > 0):
            # Keep only common speaker labels.
            # This is mainly for the newly added speakers from the labels in Y.
            keyQ = ~np.zeros_like(Q).astype(bool)
            keyP = ~np.zeros_like(P).astype(bool)
            for spk in list(QmP):
                keyQ[Q == spk] = False
            for spk in list(PmQ):
                keyP[P == spk] = False
            common_key = keyP*keyQ
            if all(~common_key) != True:
                P, Q = P[common_key], Q[common_key]
            elif all(~common_key) == True:
                P, Q = P, Q

        if len(U_set) == 1:
            # When two speaker vectors are exactly the same: No need to encode.
            mapping_array = np.array([0, 0])
            return mapping_array[Y]
        else:
            # Use one-hot encodding to find the best match.
            enc = OneHotEncoder(handle_unknown='ignore') 
            all_spks_labels = [[x] for x in range(len(U_set))]
            enc.fit(all_spks_labels)
            enc_P = enc.transform(P.reshape(-1, 1)).toarray()
            enc_Q = enc.transform(Q.reshape(-1, 1)).toarray()
            stacked = np.hstack((enc_P, enc_Q))
            cost = -1*linear_kernel(stacked.T)[spk_count:, :spk_count]
            row_ind, col_ind = linear_sum_assignment(cost)

            # If number of are speakers in each vector is not the same
            mapping_array = np.arange(len(U_set)).astype(int)
            for x in range(col_ind.shape[0]):
                if x in (set(PmQ) | set(QmP)):
                    mapping_array[x] = x
                else:
                    mapping_array[x] = col_ind[x]
            return mapping_array[Y]

class ASR_DIAR_ONLINE(ASR_DIAR_OFFLINE):
    def __init__(self, 
                 diar, 
                 cfg,
                 params):
        super().__init__(**cfg)
        '''
        Args:
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
        '''
        self._cfg_diarizer = cfg
        self.audio_queue_buffer = np.array([])
        self.params = params
        self.params['print_time'] = self._cfg_diarizer.asr.parameters.print_time
        self.params['color'] = self._cfg_diarizer.asr.parameters.colored_text
        self.params['round_float'] = 2
        self.use_cuda = self.params['use_cuda']
        self.rttm_file_path = diar.rttm_file_path
        self.sample_rate = diar.sample_rate
        if self.use_cuda:
            self.device = torch.device("cuda:0")
            self.cuda = True
        else: 
            self.device = torch.device("cpu")
            self.cuda = False

        self.frame_len = float(self._cfg_diarizer.asr.parameters.frame_len)
        self.frame_overlap = float(self._cfg_diarizer.asr.parameters.frame_overlap)

        self.string_out = ""
        self.asr_batch_size = 16
        self.asr_model_path = self._cfg_diarizer.asr.model_path
        self._load_vad_model(params)
        self._load_ASR_model(params)
        self._init_FrameBatchASR()
        self._init_FrameBatchVAD()
        self._load_punctuation_model()
        self.ROUND = 2
        self.audio_off_count = 0

        # For diarization
        self.diar = diar
        self.n_embed_seg_len = int(self.sample_rate * self.diar.embed_seg_len)
         
        self.fine_embs_array = None
        self.frame_index = 0
        self.Y_fullhist = []
        
        # minimun width to consider non-speech activity 
        self.asr_based_vad_threshold = self._cfg_diarizer.asr.parameters.asr_based_vad_threshold
        self.CHUNK_SIZE = int(self.frame_len*self.sample_rate)
        self.n_frame_len = int(self.frame_len * self.sample_rate)
        self.n_frame_overlap = int(self.frame_overlap * self.sample_rate)
        self.buffer = np.zeros(shape=2*self.n_frame_overlap + self.n_frame_len,
                               dtype=np.float32)
        self.buffer_length = self.buffer.shape[0]
        self.overlap_frames_count = int(self.n_frame_overlap/self.sample_rate)
        self.cumulative_speech_labels = []

        self.buffer_start = None
        self.frame_start = 0
        self.word_seq = []
        self.word_ts_seq = []
        self.merged_cluster_labels = []
        self.offline_logits = None
        self.debug_mode = False
        self.online_diar_label_update_sec = 30
        self.streaming_buffer_list = []
        self.reset()
        self.segment_ranges = []
        self.cluster_labels = []
    
    def _load_vad_model(self, params):
        if self._cfg_diarizer.vad.model_path is not None:
            self.vad_model = nemo_asr.models.EncDecClassificationModel.restore_from(self._cfg_diarizer.vad.model_path)
            self.vad_model = self.vad_model.to(self.device)
            self.vad_model.eval()
    
    def _load_ASR_model(self, params):
        if 'citrinet' in  self.asr_model_path.lower():
            self.asr_stride = 8
            self.asr_delay_sec = 0.12
            encdec_class = nemo_asr.models.EncDecCTCModelBPE
        elif 'conformer' in self.asr_model_path.lower():
            self.asr_stride = 4
            self.asr_delay_sec = 0.06
            encdec_class = nemo_asr.models.EncDecCTCModelBPE
        else:
            raise ValueError(f"{self.asr_model_path} is not compatible with the streaming launcher.")
        
        if '.nemo' in self.asr_model_path.lower():
            self.new_asr_model = encdec_class.restore_from(restore_path=self.asr_model_path, map_location=self.device)
        else:
            self.new_asr_model = encdec_class.from_pretrained(self.asr_model_path, map_location=self.device)

        self.new_asr_model = self.new_asr_model.to(self.device)
        self.new_asr_model = self.new_asr_model.eval()
        self.params['offset'] = 0
        self.params['time_stride'] = self.asr_stride
        self.buffer_list = []

    def _load_punctuation_model(self):
        self.punctuation_model = self._cfg_diarizer.asr.parameters.punctuation_model
        if self.punctuation_model is not None:
            self.punctuation_model = nemo_nlp.models.PunctuationCapitalizationModel.from_pretrained(self._cfg_diarizer.asr.parameters.punctuation_model)
            self.punct_cursor = 0

    def _convert_to_torch_var(self, audio_signal):
        audio_signal = torch.stack(audio_signal).float().to(self.new_asr_model.device)
        audio_signal_lens = torch.from_numpy(np.array([self.n_embed_seg_len for k in range(audio_signal.shape[0])])).to(self.new_asr_model.device)
        return audio_signal, audio_signal_lens
    
    def _update_word_and_word_ts(self, words, word_timetamps):
        """
        Stitch the existing word sequence in the buffer with the new word sequence.
        """

        if len(self.word_seq) == 0:
            if self.punctuation_model:
                words = self.punctuate_words(words)
            self.word_seq.extend(words)
            self.word_ts_seq.extend(word_timetamps)
        elif len(words) > 0:
            # Find the first word that starts after frame_start point in state buffer self.word_seq
            before_frame_start_old = torch.tensor(self.word_ts_seq)[:,0] < self.frame_start
            if all(before_frame_start_old):
                old_end = len(self.word_seq)
            else:
                old_end = torch.where(before_frame_start_old == False)[0][0].item()
            
            # Find the first word that starts after frame_start point in incoming words
            before_frame_start_new = torch.tensor(word_timetamps)[:,0] < self.frame_start
            if all(before_frame_start_new):
                new_stt = len(word_timetamps)
            else:
                new_stt = torch.where(before_frame_start_new == False)[0][0].item()
            
            del self.word_seq[old_end:]
            del self.word_ts_seq[old_end:]
            if self.punctuation_model:
                # Prevent the capitialization of the first word.
                if new_stt == 0:
                    words.insert(0, words[0])
                    cas = 0 
                else:
                    cas = max(new_stt-1, 0)
                _words = self.punctuate_words(words[cas:])[1:]
                self.word_seq.extend(_words)
            else:
                self.word_seq.extend(words[new_stt:])
            
            self.word_ts_seq.extend(word_timetamps[new_stt:])
    
    @torch.no_grad()
    def _run_embedding_extractor(self, audio_signal):
        torch_audio_signal, torch_audio_signal_lens = self._convert_to_torch_var(audio_signal)
        _, torch_embs = self.diar._speaker_model.forward(input_signal=torch_audio_signal, 
                                                         input_signal_length=torch_audio_signal_lens)
        return torch_embs
    
    @timeit 
    def punctuate_words(self, words):
        """
        This is 

        """
        if self.punctuation_model is not None:
            try:
                words = self.punctuation_model.add_punctuation_capitalization([' '.join(words)])[0].split()
                words[0] = words[0].lower()
                words[-1] = words[-1].replace(".", "")
                print("After  Punctuation:", words)
            except:
                logging.info("Punctuation Failed.")
        return words

    @timeit
    def _extract_speaker_embeddings(self, hop, embs_array, audio_signal, segment_ranges, online_extraction=True):
        """
        Extract speaker embeddings based on audio_signal and segment_ranges varialbes. Unlike offline speaker diarization,
        speaker embedding and subsegment ranges are not saved on the disk.


        """
        if online_extraction:
            if embs_array is None:
                target_segment_count = len(segment_ranges)
                stt, end = 0, len(segment_ranges)
            else:
                target_segment_count = int(min(np.ceil((2*self.frame_overlap + self.frame_len)/hop), len(segment_ranges)))
                stt, end = len(segment_ranges)-target_segment_count, len(segment_ranges)
                print(f" stt {stt} end {end} ")
             
            if end > stt:
                torch_embs = self._run_embedding_extractor(audio_signal[stt:end])
                if embs_array is None:
                    embs_array = torch_embs
                else:
                    embs_array = torch.vstack((embs_array[:stt,:], torch_embs))
            assert len(segment_ranges) == embs_array.shape[0], "Segment ranges and embs_array shapes do not match."
            
        else:
            torch_embs = self._run_embedding_extractor(audio_signal)
            embs_array = torch_embs.cpu().numpy()
        return embs_array
    
    def realign_speaker_labels(self, words, memory_cluster_labels):
        max_corrected_words = 2
        cluster_labels = torch.tensor(memory_cluster_labels)
        words_mat = np.array(words)
        scale_idx = 0
        sent_turn = [ self.check_t(x) for x in words]
        stt_idx = torch.where( torch.cat((torch.tensor([False]), (torch.tensor(sent_turn) == True)))[:-1] == True )[0]
        end_idx = torch.where( (torch.tensor(sent_turn) == True) == True )[0]
        for idx in range(stt_idx.shape[0]-1):
            range_spk_labels = cluster_labels[stt_idx[idx]:end_idx[idx+1]+1].tolist()
            range_words = words[stt_idx[idx]:end_idx[idx+1]+1]
            if stt_idx[idx] == end_idx[idx+1] or len(set(range_spk_labels)) <= 1:
                continue
            range_spk_labels = torch.tensor(range_spk_labels)
            fixed_range_labels = torch.mode(range_spk_labels)[0] * torch.ones_like(range_spk_labels)
            spk_count = torch.bincount(range_spk_labels)
            if torch.min(spk_count) != max_corrected_words and torch.min(spk_count) <= max_corrected_words:
                memory_cluster_labels[stt_idx[idx]: end_idx[idx+1]+1] = fixed_range_labels.tolist()
        self.fixed_speaker_labels  = copy.deepcopy([ f"speaker_{x}" for x in memory_cluster_labels])
    
    def check_t(self, x):
        return any([ _str in x for _str in ['.', ',', '?'] ] )
    
    def _fix_speaker_label_per_word(self, words, word_ts_list, pred_diar_labels):
        word_speaker_labels = []
        start_point, end_point, speaker = pred_diar_labels[0].split()
        word_pos, idx = 0, 0
        for j, word_ts_stt_end in enumerate(word_ts_list):
            word_pos = np.mean(word_ts_stt_end) + 0.1
            if word_pos < float(end_point):
                spk_int = int(speaker.split('_')[1])
                word_speaker_labels.append(spk_int)  
            else:
                idx += 1
                idx = min(idx, len(pred_diar_labels)-1)
                old_speaker = speaker
                start_point, end_point, speaker = pred_diar_labels[idx].split()
                spk_int = int(speaker.split('_')[1])
                word_speaker_labels.append(spk_int)  
                   
        assert len(word_speaker_labels) == len(words)
        self.realign_speaker_labels(words, word_speaker_labels)
    
    def get_speaker_label_per_word(self, words, word_ts_list, pred_diar_labels):
        params = self.params
        start_point, end_point, speaker = pred_diar_labels[0].split()
        old_speaker = speaker
        word_pos, idx = 0, 0
        string_out = ''
        string_out = self.diar.print_time_colored(string_out, speaker, start_point, end_point, params)
        for j, word_ts_stt_end in enumerate(word_ts_list):
            word_pos = np.mean(word_ts_stt_end) + 0.1
            if word_pos < float(end_point):
                string_out = self.diar.print_word_colored(string_out, words[j], params)
            else:
                idx += 1
                idx = min(idx, len(pred_diar_labels)-1)
                old_speaker = speaker
                start_point, end_point, speaker = pred_diar_labels[idx].split()
                if speaker != old_speaker:
                    string_out = self.diar.print_time_colored(string_out, speaker, start_point, end_point, params)
                else:
                    string_out = self.diar.print_time_colored(string_out, speaker, start_point, end_point, params, replace_time=True)
                string_out = self.diar.print_word_colored(string_out, words[j], params)

            stt_sec, end_sec = self.get_timestamp_in_sec(word_ts_stt_end, params)
        # string_out = self.break_lines(string_out)
        string_out = string_out.replace('#', ' ')
        if self.rttm_file_path:
            string_out = self.print_online_DER_info(self.diar.uniq_id, string_out, pred_diar_labels, params)
        else:
            logging.info(
                "Streaming Diar [{}][frame-  {}th  ]:".format(
                    self.diar.uniq_id, self.frame_index
                )
            )

        return string_out 
    
    def _get_speaker_label_per_word(self, words, word_ts_list, pred_diar_labels):
        params = self.params
        start_point, end_point = word_ts_list[0]
        speaker = self.fixed_speaker_labels[0]
        old_speaker = speaker
        word_pos, idx = 0, 0
        string_out = ''
        string_out = self.diar.print_time_colored(string_out, speaker, start_point, end_point, params)
        for j, word_ts_stt_end in enumerate(word_ts_list):
            word_pos = np.mean(word_ts_stt_end) + 0.1
            old_speaker = speaker
            if old_speaker == self.fixed_speaker_labels[j]:
                speaker = self.fixed_speaker_labels[j]
                start_point, end_point = word_ts_stt_end
                string_out = self.diar.print_time_colored(string_out, speaker, start_point, end_point, params, replace_time=True, space='')
                string_out = self.diar.print_word_colored(string_out, words[j], params)
                old_speaker = self.fixed_speaker_labels[j]
            else:
                idx += 1
                idx = min(idx, len(pred_diar_labels)-1)
                start_point, end_point = word_ts_stt_end
                speaker = self.fixed_speaker_labels[j]
                if speaker != old_speaker:
                    string_out = self.diar.print_time_colored(string_out, speaker, start_point, end_point, params)
                else:
                    string_out = self.diar.print_time_colored(string_out, speaker, start_point, end_point, params, replace_time=True, space='')
                string_out = self.diar.print_word_colored(string_out, words[j], params)

            stt_sec, end_sec = self.get_timestamp_in_sec(word_ts_stt_end, params)
        # string_out = self.break_lines(string_out)
        string_out = string_out.replace('#', ' ')
        if self.rttm_file_path:
            string_out = self.print_online_DER_info(self.diar.uniq_id, string_out, pred_diar_labels, params)
        else:
            logging.info(
                "Streaming Diar [{}][frame-  {}th  ]:".format(
                    self.diar.uniq_id, self.frame_index
                )
            )

        return string_out 
    
    @staticmethod
    def break_lines(string_out, max_line_N=90):
        split_string_out = string_out.split('\n')
        return_string_out = []
        for org_chunk in split_string_out:
            buffer = []
            color_str = org_chunk[:7]
            chunk = org_chunk[7:]
            if len(chunk) > max_line_N:
                for i in range(0, len(chunk), max_line_N):
                    buffer.append(color_str+chunk[i:i+max_line_N])
                return_string_out.extend(buffer)
            else:
                return_string_out.append(org_chunk)
        return '\n'.join(return_string_out)

    @staticmethod
    def get_timestamp_in_sec(word_ts_stt_end, params):
        stt = round(params['offset'] + word_ts_stt_end[0] * params['time_stride'], params['round_float'])
        end = round(params['offset'] + word_ts_stt_end[1] * params['time_stride'], params['round_float'])
        return stt, end
    
    @timeit  
    def print_online_DER_info(self, uniq_id, string_out, pred_diar_labels, params):
        if params['color']:
            color = self.diar.color_palette['white']
        else:
            color = ''
        der_dict, der_stat_dict = self.diar.online_eval_diarization(pred_diar_labels, self.rttm_file_path)
        DER, FA, MISS, CER = der_dict['DER'], der_dict['FA'], der_dict['MISS'], der_dict['CER']
        string_out += f'\n{color}============================================================================='
        string_out += f'\n{color}[Session: {uniq_id}, DER:{DER:.2f}%, FA:{FA:.2f}% MISS:{MISS:.2f}% CER:{CER:.2f}%]'
        string_out += f'\n{color}[Num of Speakers (Est/Ref): {der_stat_dict["est_n_spk"]}/{der_stat_dict["ref_n_spk"]}]'
        self.diar.DER_csv_list.append(f"{self.frame_index}, {DER}, {FA}, {MISS}, {CER}\n")
        write_txt(f"{self.diar._out_dir}/{uniq_id}.csv", ''.join(self.diar.DER_csv_list))
        return string_out
    
    def update_frame_to_buffer(self, frame): 
        if frame is None:
            frame = np.zeros(shape=self.n_frame_len, dtype=np.float32)
        if len(frame) < self.n_frame_len:
            frame = np.pad(frame, [0, self.n_frame_len - len(frame)], 'constant')
        assert len(frame)==self.n_frame_len
        self.buffer_start = round(float(self.frame_index - 2*self.overlap_frames_count), 2)
        self.buffer[:-self.n_frame_len] = copy.deepcopy(self.buffer[self.n_frame_len:])
        self.buffer[-self.n_frame_len:] = copy.deepcopy(frame)
        
    @timeit
    def get_VAD_from_ASR(self, input_word_ts):
        speech_labels = []
        word_ts = copy.deepcopy(input_word_ts)
        if word_ts == []:
            return speech_labels
        else:
            count = len(word_ts)-1
            while count > 0:
                if len(word_ts) > 1: 
                    if word_ts[count][0] - word_ts[count-1][1] <= self.asr_based_vad_threshold:
                        trangeB = word_ts.pop(count)
                        trangeA = word_ts.pop(count-1)
                        word_ts.insert(count-1, [trangeA[0], trangeB[1]])
                count -= 1
        return word_ts 
    
    def set_buffered_infer_params(self, asr_model: Type[EncDecCTCModelBPE], frame_asr) -> Tuple[float, float, float]:
        """
        Prepare the parameters for the buffered inference.
        """
        cfg = copy.deepcopy(asr_model._cfg)
        OmegaConf.set_struct(cfg.preprocessor, False)

        # some changes for streaming scenario
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        cfg.preprocessor.normalize = "None"

        preprocessor = nemo_asr.models.EncDecCTCModelBPE.from_config_dict(cfg.preprocessor)
        preprocessor.to(asr_model.device)
        frame_asr.raw_preprocessor = preprocessor

        # Disable config overwriting
        OmegaConf.set_struct(cfg.preprocessor, True)
        self.offset_to_mid_window = (self.total_buffer_in_secs - self.chunk_len_in_sec) / 2
        self.onset_delay = (
            math.ceil(((self.total_buffer_in_secs - self.chunk_len_in_sec) / 2) / self.model_stride_in_secs) + 1
        )
        self.mid_delay = math.ceil(
            (self.chunk_len_in_sec + (self.total_buffer_in_secs - self.chunk_len_in_sec) / 2)
            / self.model_stride_in_secs
        )
        self.tokens_per_chunk = math.ceil(self.chunk_len_in_sec / self.model_stride_in_secs)
        return self.onset_delay, self.mid_delay, self.tokens_per_chunk
    
    def _init_FrameBatchVAD(self):
        torch.manual_seed(0)
        torch.set_grad_enabled(False)

        self.chunk_len_in_sec = self.frame_len
        context_len_in_secs = self.frame_overlap
        self.total_buffer_in_secs = 2*context_len_in_secs + self.chunk_len_in_sec
        self.model_stride_in_secs = 0.04

        self.frame_vad = FrameBatchVAD_sample(
                    vad_model=self.vad_model, 
                    frame_len=self.chunk_len_in_sec, 
                    total_buffer=self.total_buffer_in_secs,
                    batch_size=self.asr_batch_size, 
                    patience=1,
                    total_asr_buffer=self.total_buffer_in_secs
                    )
        self.frame_vad.reset()

    def _init_FrameBatchASR(self):
        torch.manual_seed(0)
        torch.set_grad_enabled(False)

        self.chunk_len_in_sec = self.frame_len
        context_len_in_secs = self.frame_overlap
        self.total_buffer_in_secs = 2*context_len_in_secs + self.chunk_len_in_sec
        self.model_stride_in_secs = 0.04

        self.werbpe_ts = WERBPE_TS(
            tokenizer=self.new_asr_model.tokenizer,
            batch_dim_index=0,
            use_cer=self.new_asr_model._cfg.get('use_cer', False),
            ctc_decode=True,
            dist_sync_on_step=True,
            log_prediction=self.new_asr_model._cfg.get("log_prediction", False),
        )
            
        self.frame_asr = FrameBatchASR_Logits_Sample(
            asr_model=self.new_asr_model,
            frame_len=self.chunk_len_in_sec,
            total_buffer=self.total_buffer_in_secs,
            batch_size=self.asr_batch_size,
        )
        self.frame_asr.reset()

        self.set_buffered_infer_params(self.new_asr_model, self.frame_asr)
        self.onset_delay_in_sec = round(self.onset_delay * self.model_stride_in_secs, 2)
    
    @timeit
    def _run_VAD_decoder(self, buffer):
        """
        Place holder for VAD integration. This function returns vad_mask that is identical for ASR feature matrix for
        the current buffer.
        """
        # vad_logits, speech_segments, feats_shape = get_vad_feat_logit_single(buffer,
                                                    # self.frame_vad,
                                                    # self.chunk_len_in_sec,
                                                    # self.tokens_per_chunk,
                                                    # self.mid_delay,
                                                    # self.model_stride_in_secs,
                                                    # threshold=0.05,
                                                # )
        # A Placeholder which should be replaced with streaming VAD instance
        hyps, tokens_list, feats_shape  = get_wer_feat_logit_single(buffer,
                                                    self.frame_asr,
                                                    self.chunk_len_in_sec,
                                                    self.tokens_per_chunk,
                                                    self.mid_delay,
                                                    self.model_stride_in_secs,
                                                    frame_mask=None,
                                                )
        vad_mask = torch.ones(feats_shape)
        vad_timestamps = None
        return vad_mask, vad_timestamps
    
    @timeit
    def run_ASR_decoder(self, buffer, frame_mask):
        hyps, tokens_list, _  = get_wer_feat_logit_single(buffer,
                                                    self.frame_asr,
                                                    self.chunk_len_in_sec,
                                                    self.tokens_per_chunk,
                                                    self.mid_delay,
                                                    self.model_stride_in_secs,
                                                    frame_mask,
                                                )
        greedy_predictions_list = tokens_list[0]
        logits_len = torch.from_numpy(np.array([len(greedy_predictions_list)]))
        greedy_predictions = torch.from_numpy(np.array(greedy_predictions_list)).unsqueeze(0)
        text, char_ts, _word_ts = self.werbpe_ts.ctc_decoder_predictions_tensor_with_ts(
            self.model_stride_in_secs, greedy_predictions, predictions_len=logits_len
        )
        words, word_ts = text[0].split(), _word_ts[0]
        assert len(words) == len(word_ts)
        self.asr_offset = self.buffer_start - self.onset_delay_in_sec
        words_adj, word_ts_adj = [], []
        for w, x in zip(words, word_ts):
            word_range = [round(x[0] + self.asr_offset,2), round(x[1] + self.asr_offset,2)] 
            if word_range[1] >  0.0:
                word_ts_adj.append(word_range)
                words_adj.append(w)

        return words_adj, word_ts_adj

    @timeit
    def _run_ASR_decoder(self, buffer, frame_mask):
        hyps, tokens_list, _  = get_wer_feat_logit_single(buffer,
                                                    self.frame_asr,
                                                    self.chunk_len_in_sec,
                                                    self.tokens_per_chunk,
                                                    self.mid_delay,
                                                    self.model_stride_in_secs,
                                                    frame_mask,
                                                )
        greedy_predictions_list = tokens_list[0]
        logits_len = torch.from_numpy(np.array([len(greedy_predictions_list)]))
        greedy_predictions = torch.from_numpy(np.array(greedy_predictions_list)).unsqueeze(0)
        text, char_ts, _word_ts = self.werbpe_ts.ctc_decoder_predictions_tensor_with_ts(
            self.model_stride_in_secs, greedy_predictions, predictions_len=logits_len
        )
        words, word_ts = text[0].split(), _word_ts[0]
        assert len(words) == len(word_ts)
        self.asr_offset = self.buffer_start - self.onset_delay_in_sec
        words_adj, word_ts_adj = [], []
        for w, x in zip(words, word_ts):
            word_range = [round(x[0] + self.asr_offset,2), round(x[1] + self.asr_offset,2)] 
            if word_range[1] >  0.0:
                word_ts_adj.append(word_range)
                words_adj.append(w)

        return words_adj, word_ts_adj

    def _get_update_abs_time(self):
        new_bufflen_sec = self.n_frame_len / self.sample_rate
        n_buffer_samples = int(len(self.buffer)/self.sample_rate)
        total_buffer_len_sec = n_buffer_samples/self.frame_len
        self.buffer_end = self.buffer_start + total_buffer_len_sec
        self.frame_start = round(self.buffer_start + int(self.n_frame_overlap/self.sample_rate), self.ROUND)

    def callback_sim(self, sample_audio):
        assert len(sample_audio) == int(self.sample_rate * self.frame_len)
        start_time = time.time()
        words, timestamps, pred_diar_labels = self.transcribe(sample_audio)
        if self.buffer_start >= 0 and (pred_diar_labels != [] and pred_diar_labels != None):
            assert len(words) == len(timestamps)
            self._update_word_and_word_ts(words, timestamps)
            self._fix_speaker_label_per_word(self.word_seq, self.word_ts_seq, pred_diar_labels)
            self.string_out = self._get_speaker_label_per_word(self.word_seq, self.word_ts_seq, pred_diar_labels)
            # self.string_out = self.get_speaker_label_per_word(self.word_seq, self.word_ts_seq, pred_diar_labels)
            write_txt(f"{self.diar._out_dir}/print_script.sh", self.string_out.strip())
            
        ETA = time.time()-start_time 

    def audio_queue_launcher(self, Audio, state=""):
        try:
            audio_queue = process_audio_file(Audio)
            self.audio_queue_buffer = np.append(self.audio_queue_buffer, audio_queue)
            while len(self.audio_queue_buffer) > self.CHUNK_SIZE:
                sample_audio, self.audio_queue_buffer = self.audio_queue_buffer[:self.CHUNK_SIZE], self.audio_queue_buffer[self.CHUNK_SIZE:]
                self.callback_sim(sample_audio)
        except:
            logging.info("Audio stream is off.")
            time.sleep(0.1)
        return f"Audio Queue Length {len(self.audio_queue_buffer)/self.sample_rate:.2f}s", ""
    
    @torch.no_grad()
    def transcribe(self, frame=None, merge=True):
        self.update_frame_to_buffer(frame)

        vad_mask, vad_ts = self._run_VAD_decoder(self.buffer) 
        text, word_ts = self._run_ASR_decoder(self.buffer, frame_mask=vad_mask)

        if vad_ts is None:
            vad_ts = self.get_VAD_from_ASR(word_ts)
        
        self.diar.frame_index = self.frame_index
        pred_diar_labels = self.diar.online_diarization(self, vad_ts)

        self.frame_index += 1
        return text, word_ts, pred_diar_labels
    
    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        self.buffer=np.zeros(shape=self.buffer.shape, dtype=np.float32)
        self.prev_char = ''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    hydra.initialize(config_path="conf")
    cfg = hydra.compose(config_name="/online_diarization_with_asr.yaml")
    
    params = {}
    params['use_cuda'] = True
    AUDIO_RTTM_MAP = audio_rttm_map(cfg.diarizer.manifest_filepath)
    
    diar = OnlineDiarizer(cfg)
    diar.uniq_id = "en_0638"
    diar.single_audio_file_path = AUDIO_RTTM_MAP[diar.uniq_id]['audio_filepath']
    diar.rttm_file_path = AUDIO_RTTM_MAP[diar.uniq_id]['rttm_filepath']
    # diar.rttm_file_path = None
    diar._init_segment_variables()
    
    asr_diar = ASR_DIAR_ONLINE(diar, cfg=cfg.diarizer, params=params)
    diar.device = asr_diar.device
    asr_diar.reset()
    
    simulation = True
    # simulation = False
    if simulation:
        samplerate, sdata = wavfile.read(diar.single_audio_file_path)
        for index in range(int(np.floor(sdata.shape[0]/asr_diar.n_frame_len))):
            asr_diar.buffer_counter = index
            sample_audio = sdata[asr_diar.CHUNK_SIZE*(asr_diar.buffer_counter):asr_diar.CHUNK_SIZE*(asr_diar.buffer_counter+1)]
            asr_diar.callback_sim(sample_audio)
    else:
        isTorch = torch.cuda.is_available()
        asr_diar.rttm_file_path = None
        iface = gr.Interface(
            fn=asr_diar.audio_queue_launcher,
            inputs=[
                gr.inputs.Audio(source="microphone", type='filepath'), 
                "state",
            ],
            outputs=[
                "textbox",
                "state",
            ],
            layout="horizontal",
            theme="huggingface",
            title=f"NeMo Streaming Conformer CTC Large - English, CUDA:{isTorch}",
            description="Demo for English speech recognition using Conformer Transducers",
            allow_flagging='never',
            live=True,
        )
        iface.launch(share=False)

