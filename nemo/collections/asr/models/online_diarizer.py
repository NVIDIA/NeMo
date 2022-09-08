# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import json
import os
import pickle as pkl
import shutil
import tarfile
import tempfile
import time
from collections import Counter
from copy import deepcopy
from typing import List, Optional

import librosa
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from nemo.collections.asr.data.audio_to_label import repeat_signal
from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.models.classification_models import EncDecClassificationModel
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.mixins.mixins import DiarizationMixin
from nemo.collections.asr.parts.utils.nmesc_clustering import (
    NMESC,
    SpectralClustering,
    getAffinityGraphMat,
    getCosAffinityMatrix,
    getEnhancedSpeakerCount,
    getMultiScaleCosAffinityMatrix,
    getTempInterpolMultiScaleCosAffinityMatrix,
)
from nemo.collections.asr.parts.utils.speaker_utils import (
    audio_rttm_map,
    combine_int_overlaps,
    fl2int,
    get_contiguous_stamps,
    get_embs_and_timestamps,
    get_subsegments,
    get_uniqname_from_filepath,
    getMergedRanges,
    getOverlapRange,
    getSubRangeList,
    int2fl,
    isOverlap,
    labels_to_pyannote_object,
    labels_to_rttmfile,
    merge_stamps,
    rttm_to_labels,
    score_labels,
)
from nemo.collections.asr.parts.utils.vad_utils import (
    generate_overlap_vad_seq,
    generate_vad_segment_table,
    get_vad_stream_status,
    prepare_manifest,
)
from nemo.core.classes import Model
from nemo.utils import logging, model_utils

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


__all__ = ['OnlineDiarizer']

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
            # pass
        return result
    return timed


def hungarian_algorithm(spk_count, U_set, cmm_P, cmm_Q, PmQ, QmP):
    """
    Use one-hot encodding to find the best match.

    """
    enc = OneHotEncoder(handle_unknown='ignore')
    all_spks_labels = [[x] for x in range(len(U_set))]
    enc.fit(all_spks_labels)
    enc_P = enc.transform(cmm_P.reshape(-1, 1)).toarray()
    enc_Q = enc.transform(cmm_Q.reshape(-1, 1)).toarray()
    stacked = np.hstack((enc_P, enc_Q))
    cost = -1 * linear_kernel(stacked.T)[spk_count:, :spk_count]
    row_ind, col_ind = linear_sum_assignment(cost)

    # If number of are speakers in each vector is not the same
    mapping_array = np.arange(len(U_set)).astype(int)
    for x in range(col_ind.shape[0]):
        if x in (set(PmQ) | set(QmP)):
            mapping_array[x] = x
        else:
            mapping_array[x] = col_ind[x]
    return mapping_array

def get_indices_for_merging(cmat, ndx, target_num):
    """
    Get indeces of the embeddings we want to merge or drop.

    Args:
        cmat: (np.array)
        ndx: (np.array)
        target_num: (int)

    Output:
        tick2d: (numpy.array)
    """
    comb_limit = int(ndx.shape[0] / 2)
    assert (
        target_num <= comb_limit
    ), f" target_num is {target_num}: {target_num} is bigger than comb_limit {comb_limit}"
    idx2d = np.unravel_index(np.argsort(cmat, axis=None)[::-1], cmat.shape)
    num_of_lower_half = int((cmat.shape[0] ** 2 - cmat.shape[0]) / 2)
    idx2d = (idx2d[0][:num_of_lower_half], idx2d[1][:num_of_lower_half])
    cdx, left_set, right_set, total_set = 0, [], [], []
    while len(left_set) < target_num and len(right_set) < target_num:
        Ldx, Rdx = idx2d[0][cdx], idx2d[1][cdx]
        if (not Ldx in total_set) and (not Rdx in total_set):
            left_set.append(Ldx)
            right_set.append(Rdx)
            total_set = left_set + right_set
        cdx += 1
    tick2d = np.array([left_set, right_set])
    return tick2d


def preprocess_mat(mat, symm=True, fill_diag_zero=True):
    if symm:
        mat = 0.5 * (mat + mat.T)
    if fill_diag_zero:
        np.fill_diagonal(mat, 0)
    return mat

def get_mapped_index(mat, index):
    return np.where(mat == index)[0][0]


def get_merge_quantity(
    new_emb_n, 
    pre_merge_cluster_label, 
    min_segs_per_buffer
    ):
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

        pre_merge_cluster_label: (np.array)
            the speaker labels of (the history_embedding_buffer_emb) + (the new embeddings to be added)
    """
    targeted_total_n = new_emb_n
    count_dict = Counter(pre_merge_cluster_label)
    spk_freq_count = np.bincount(pre_merge_cluster_label)
    class_vol = copy.deepcopy(spk_freq_count)
    emb_n_per_cluster = np.zeros_like(class_vol).astype(int)
    arg_max_spk_freq = np.argsort(spk_freq_count)[::-1]
    count = 0
    while np.sum(emb_n_per_cluster) < targeted_total_n:
        recurr_idx = np.mod(count, len(count_dict))
        curr_idx = arg_max_spk_freq[recurr_idx]
        margin = (spk_freq_count[curr_idx] - emb_n_per_cluster[curr_idx]) - min_segs_per_buffer
        if margin > 0:
            target_number = min(margin, new_emb_n)
            emb_n_per_cluster[curr_idx] += target_number
            new_emb_n -= target_number
        count += 1
    assert (
        sum(emb_n_per_cluster) == targeted_total_n
    ), "emb_n_per_cluster does not match with targeted number new_emb_n."
    return emb_n_per_cluster


def run_reduction_alg(
        cmat, 
        tick2d, 
        emb_ndx, 
        cluster_labels
        ):
    """


    """
    LI, RI = tick2d[0, :], tick2d[1, :]
    LI_argdx = tick2d[0].argsort()
    LI, RI = LI[LI_argdx], RI[LI_argdx]
    result_emb = 0.5 * (emb_ndx[LI, :] + emb_ndx[RI, :])
    merged_cluster_labels = cluster_labels[np.array(list(set(LI)))]
    bypass_ndx = np.array(list(set(range(emb_ndx.shape[0])) - set(list(LI) + list(RI))))
    if len(bypass_ndx) > 0:
        result_emb = np.vstack((emb_ndx[bypass_ndx], result_emb))
        merged_cluster_labels = np.hstack((cluster_labels[bypass_ndx], merged_cluster_labels))
    return result_emb, merged_cluster_labels


def get_online_segments_from_slices(
    buffer_start,
    buffer_end,
    subsegments,
    sig,
    window,
    sigs_list,
    sig_rangel_list,
    sig_indexes,
    seg_index_offset,
    sample_rate,
    decimals,
):
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
    slice_length = int(window * sample_rate)
    for (start_sec, dur) in subsegments:
        if start_sec > buffer_end:
            continue
        seg_index_offset += 1
        if (start_sec + dur) > (buffer_end - buffer_start):
            end_sec = min(start_sec + dur, (buffer_end - buffer_start))
        else:
            end_sec = start_sec + dur
        start_idx = int(start_sec * sample_rate)
        end_idx = min(int(end_sec * sample_rate), slice_length + start_idx)
        signal = sig[start_idx:end_idx]
        if len(signal) == 0:
            raise ValueError("len(signal) is zero. Signal length should not be zero.")
        if len(signal) < slice_length:
            signal = repeat_signal(signal, len(signal), slice_length)
        start_abs_sec = round(float(buffer_start + start_sec), decimals)
        end_abs_sec = round(float(buffer_start + end_sec), decimals)
        sigs_list.append(signal)
        sig_rangel_list.append([start_abs_sec, end_abs_sec])
        sig_indexes.append(seg_index_offset)
    assert len(sigs_list) == len(sig_rangel_list) == len(sig_indexes)
    return seg_index_offset, sigs_list, sig_rangel_list, sig_indexes

def generate_cluster_labels(segment_ranges, cluster_labels):
    lines = []
    for idx, label in enumerate(cluster_labels):
        tag = 'speaker_' + str(label)
        stt, end = segment_ranges[idx]
        lines.append(f"{stt} {end} {tag}")
    cont_lines = get_contiguous_stamps(lines)
    diar_hyp = merge_stamps(cont_lines)
    return diar_hyp


def get_new_cursor_for_update(
        frame_start, 
        segment_raw_audio, 
        segment_range_ts, 
        segment_indexes):
    """
    Remove the old segments that overlap with the new frame (self.frame_start)
    cursor_for_old_segments is set to the onset of the t_range popped lastly.
    """
    cursor_for_old_segments = frame_start
    while True and len(segment_raw_audio) > 0:
        t_range = segment_range_ts[-1]

        mid = np.mean(t_range)
        if frame_start <= t_range[1]:
            segment_range_ts.pop()
            segment_raw_audio.pop()
            segment_indexes.pop()
            cursor_for_old_segments = t_range[0]
        else:
            break
    return cursor_for_old_segments


def get_speech_labels_for_update(
    frame_start, 
    buffer_end, 
    vad_timestamps, 
    cumulative_speech_labels, 
    cursor_for_old_segments
):
    """
    Bring the new speech labels from the current buffer. Then

    1. Concatenate the old speech labels from self.cumulative_speech_labels for the overlapped region.
        - This goes to new_speech_labels.
    2. Update the new 1 sec of speech label (speech_label_for_new_segments) to self.cumulative_speech_labels.
    3. Return the speech label from cursor_for_old_segments to buffer end.

    """
    if cursor_for_old_segments < frame_start:
        update_overlap_range = [cursor_for_old_segments, frame_start]
    else:
        update_overlap_range = []

    # Get VAD timestamps that are in (frame_start, buffer_end) range
    new_incoming_speech_labels = getSubRangeList(
        target_range=(frame_start, buffer_end), source_range_list=vad_timestamps
    )
    
    # Update the speech label by including overlapping region with the previous output
    update_overlap_speech_labels = getSubRangeList(
        target_range=update_overlap_range, source_range_list=cumulative_speech_labels
    )

    # Speech segments for embedding extractions
    speech_label_for_new_segments = getMergedRanges(update_overlap_speech_labels, new_incoming_speech_labels)

    # Keep cumulative VAD labels for the future use
    cumulative_speech_labels = getMergedRanges(cumulative_speech_labels, new_incoming_speech_labels)

    return speech_label_for_new_segments, cumulative_speech_labels


def get_segments_from_buffer(
    buffer_start,
    buffer_end,
    sample_rate,
    speech_labels_for_update,
    audio_buffer,
    segment_indexes,
    window,
    shift,
    decimals,
):
    sigs_list, sig_rangel_list, sig_indexes = [], [], []
    if len(segment_indexes) > 0:
        seg_index_offset = segment_indexes[-1]
    else:
        seg_index_offset = -1

    for idx, range_t in enumerate(speech_labels_for_update):
        range_t = [range_t[0] - buffer_start, range_t[1] - buffer_start]
        range_t[0] = max(0, range_t[0])

        subsegments = get_subsegments(
            offset=range_t[0], window=window, shift=shift, duration=(range_t[1] - range_t[0])
        )
        target_sig = torch.from_numpy(audio_buffer)
        subsegment_output = get_online_segments_from_slices(
            buffer_start=buffer_start,
            buffer_end=buffer_end,
            subsegments=subsegments,
            sig=target_sig,
            window=window,
            sigs_list=sigs_list,
            sig_rangel_list=sig_rangel_list,
            sig_indexes=sig_indexes,
            seg_index_offset=seg_index_offset,
            sample_rate=sample_rate,
            decimals=decimals,
        )
        seg_index_offset, sigs_list, sig_rangel_list, sig_indexes = subsegment_output

    assert len(sigs_list) == len(sig_rangel_list) == len(sig_indexes)
    return sigs_list, sig_rangel_list, sig_indexes


@timeit
def stitch_cluster_labels(Y_old, Y_new, with_history=True):
    """
    Run Hungarian algorithm (linear sum assignment) to find the best permuation mapping between
    the cumulated labels in history and the new clustering output labels.

    Args:
        Y_cumul (np.array):
            Cumulated diarization labels. This will be concatenated with history embedding speaker label
            then compared with the predicted label Y_new.

        Y_new (np.array):
            Contains predicted labels for reduced history embeddings concatenated with the predicted label.
            Permutation is not matched yet.

    Returns:
        mapping_array[Y] (np.array):
            An output numpy array where the input Y_new is mapped with mapping_array.

    """
    if len(Y_old) == 0:
        matched_output = Y_new
    else:
        spk_count = max(len(set(Y_old)), len(set(Y_new)))
        P_raw, Q_raw = Y_old.astype(int), Y_new.astype(int)
        U_set = set(P_raw) | set(Q_raw)
        min_len = min(P_raw.shape[0], Q_raw.shape[0])
        P, Q = P_raw[:min_len], Q_raw[:min_len]
        PmQ, QmP = set(P) - set(Q), set(Q) - set(P)

        # P and Q occasionally have no common labels which means totally flipped (0<->1) labels.
        # This should be differentiated from the second case.
        cmm_P, cmm_Q = P, Q

        if len(U_set) == 1:
            # When two speaker vectors are exactly the same: No need to encode.
            mapping_array = np.array([0, 0])
        else:
            # Run Hungarian algorithm if there are more than one speaker in universal set U.
            mapping_array = hungarian_algorithm(spk_count, U_set, cmm_P, cmm_Q, PmQ, QmP)
        matched_output = mapping_array[Y_new]
    return matched_output

class OnlineClustering:
    def __init__(self, cfg_diarizer):
        self.max_num_speaker = cfg_diarizer.clustering.parameters.max_num_speakers
        self.max_rp_threshold = cfg_diarizer.clustering.parameters.max_rp_threshold
        self.sparse_search_volume = cfg_diarizer.clustering.parameters.sparse_search_volume
        self.fixed_thres = None
        self.p_value_skip_frame_thres = 50
        self.p_update_freq = 5
        self.min_spk_counting_buffer_size = 7
        self.min_frame_per_spk = 20
        self.p_value_queue_size = 3
        self.num_spk_stat = []
        self.p_value_hist = []

    @timeit
    def onlineNMEanalysis(self, nmesc, frame_index):
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
        if len(self.p_value_hist) == 0 or \
            (frame_index < self.p_value_skip_frame_thres and frame_index % self.p_update_freq == 0):
            est_num_of_spk, p_hat_value = nmesc.NMEanalysis()
            self.p_value_hist.append(p_hat_value)
            if len(self.p_value_hist) > self.p_value_queue_size:
                self.p_value_hist.pop(0)
        p_hat_value = max(self.p_value_hist, key=self.p_value_hist.count)
        est_num_of_spk, g_p = nmesc.getEigRatio(p_hat_value)
        return est_num_of_spk, p_hat_value
    
    def speaker_counter_buffer(self, est_num_of_spk):
        """
        Use a queue to avoid unstable speaker counting results.
        """
        self.num_spk_stat.append(est_num_of_spk)
        if len(self.num_spk_stat) > self.min_spk_counting_buffer_size:
            self.num_spk_stat.pop(0)
        num_spks_bincount = torch.bincount(torch.tensor(self.num_spk_stat))
        est_num_of_spk = torch.argmax(num_spks_bincount)
        return est_num_of_spk

    def limit_frames_per_speaker(self, frame_index, est_num_of_spk):
        """
        Limit the estimated number of speakers in proportion to the number of speakers.

        Args:
            est_num_of_spk (int): Estimated number of speakers
        Returns:
            (int) Estimated number of speakers capped by `self.min_frame_per_spk`
        """
        return min(est_num_of_spk, int(1 + frame_index// self.min_frame_per_spk))

    def online_spk_num_estimation(self, emb, mat, nmesc, frame_index):
        """
        Online version of speaker estimation involves speaker counting buffer and application of per-speaker
        frame count limit.

        """
        est_num_of_spk, p_hat_value = self.onlineNMEanalysis(nmesc, frame_index)
        affinity_mat = getAffinityGraphMat(mat, p_hat_value)
        est_num_of_spk = self.speaker_counter_buffer(est_num_of_spk)
        est_num_of_spk = self.limit_frames_per_speaker(frame_index, est_num_of_spk)
        return est_num_of_spk, affinity_mat

    def onlineCOSclustering(self, 
        emb: torch.Tensor,
        frame_index: int,
        cuda,
        device
        ):
        mat = getCosAffinityMatrix(emb)
        if emb.shape[0] == 1:
            return torch.zeros((1,), dtype=torch.int32)

        nmesc = NMESC(
            mat,
            max_num_speaker=self.max_num_speaker,
            max_rp_threshold=self.max_rp_threshold,
            sparse_search=True,
            maj_vote_spk_count=False,
            sparse_search_volume=self.sparse_search_volume,
            fixed_thres=self.fixed_thres,
            NME_mat_size=256,
            device=device,
        )
        
        est_num_of_spk, affinity_mat = self.online_spk_num_estimation(emb, mat, nmesc, frame_index)
        spectral_model = SpectralClustering(n_clusters=est_num_of_spk, cuda=cuda, device=device)
        Y = spectral_model.predict(affinity_mat)
        return Y


class OnlineDiarizer(ClusteringDiarizer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        # Convert config to support Hydra 1.0+ instantiation
        self.uniq_id = None
        self.AUDIO_RTTM_MAP = audio_rttm_map(self.cfg.diarizer.manifest_filepath)
        self.sample_rate = self.cfg.sample_rate
        self._cfg_diarizer = self.cfg.diarizer
        torch.manual_seed(0)
        
        self._out_dir = self._cfg_diarizer.out_dir
        if not os.path.exists(self._out_dir):
            os.mkdir(self._out_dir)
        
        self._init_online_clustering_module()
        self._init_memory_buffer_variables()
        self._init_temporal_major_voting_module()
        self._init_buffer_frame_timestamps()
        
        # Set speaker embedding model in eval mode
        self._speaker_model.eval()
    
    def reset(self):
        """
        Reset all the variables
        """
        self._init_segment_variables()
        self._init_online_clustering_module()
        self._init_memory_buffer_variables()
        self._init_temporal_major_voting_module()
        self._init_buffer_frame_timestamps()

    def _init_online_clustering_module(self): 
        self.online_clus = OnlineClustering(self.cfg.diarizer)
        self.max_num_speakers = self.online_clus.max_num_speaker
        self.base_scale_index = max(self.multiscale_args_dict['scale_dict'].keys())
        
    def _init_memory_buffer_variables(self):
        self.memory_segment_ranges = {key: [] for key in self.multiscale_args_dict['scale_dict'].keys()}
        self.memory_segment_indexes = {key: [] for key in self.multiscale_args_dict['scale_dict'].keys()}
        self.memory_cluster_labels = np.array([])
        self.Y_fullhist = []
        self.cumulative_speech_labels = []

        self.embed_seg_len = self.multiscale_args_dict['scale_dict'][self.base_scale_index][0]
        self.n_embed_seg_len = int(self.sample_rate * self.embed_seg_len)
        self.max_embed_count = 0
        self.decimals = 2

        self.MINIMUM_CLUS_BUFFER_SIZE = 32
        self.MINIMUM_HIST_BUFFER_SIZE = 32
        
        self.memory_margin = self.MINIMUM_CLUS_BUFFER_SIZE
        self.history_buffer_size = self._cfg_diarizer.clustering.parameters.history_buffer_size
        self.current_buffer_size = self._cfg_diarizer.clustering.parameters.current_buffer_size

        self._minimum_segments_per_buffer = int(self.history_n / self.max_num_speakers)

        self.history_embedding_buffer_emb = np.array([])
        self.history_embedding_buffer_label = np.array([])
        self.history_buffer_seg_end = 0
    
    def _init_temporal_major_voting_module(self):
        self.use_temporal_label_major_vote = False
        self.temporal_label_major_vote_buffer_size = 11
        self.base_scale_label_dict = {}

    
    def _init_segment_variables(self):
        self.embs_array = {self.uniq_id: {}}
        self.time_stamps = {self.uniq_id: {}}
        self.segment_range_ts = {self.uniq_id: {}}
        self.segment_raw_audio = {self.uniq_id: {}}
        self.segment_indexes = {self.uniq_id: {}}

        for scale_idx, (window, shift) in self.multiscale_args_dict['scale_dict'].items():
            self.multiscale_embeddings_and_timestamps[scale_idx] = [None, None]
            self.embs_array[scale_idx] = None
            self.time_stamps[scale_idx] = []
            self.segment_range_ts[scale_idx] = []
            self.segment_raw_audio[scale_idx] = []
            self.segment_indexes[scale_idx] = []

    def _init_buffer_frame_timestamps(self):
        """
        Variables trasferred from ASR_DIAR_ONLINE class
        """
        self.frame_index = 0
        self.frame_start = 0.0
        self.buffer_start = 0.0
        self.buffer_end = 0.0
        self.buffer = None


    @property
    def history_buffer_size(self, value):
        return self.current_n

    @history_buffer_size.setter
    def history_buffer_size(self, value):
        logging.info(f"Setting online diarization buffer to : {value}")
        assert (
            value >= self.MINIMUM_CLUS_BUFFER_SIZE
        ), f"Online diarization clustering buffer should be bigger than {self.MINIMUM_CLUS_BUFFER_SIZE}"
        self.current_n = value  # How many segments we want to use as clustering buffer

    @property
    def current_buffer_size(self, value):
        return self.current_n

    @current_buffer_size.setter
    def current_buffer_size(self, value):
        logging.info(f"Setting online diarization buffer to : {value}")
        assert (
            value >= self.MINIMUM_HIST_BUFFER_SIZE
        ), f"Online diarization history buffer should be bigger than {self.MINIMUM_HIST_BUFFER_SIZE}"
        self.history_n = value  # How many segments we want to use as history buffer

    def prepare_embedding_update(self, emb_in):
        """

        We only save the index and clustering label of each embedding.

        Case-1
            This else statement is for the very first diarization loop.
            This is the very first reduction frame.
        Case-2
            Since there are new embeddings, we push the same amount (new_emb_n)
            of old embeddings to the history buffer.
            We should also update self.history_buffer_seg_end which is a pointer.
                update to history emb: emb_in[emb_idx_stt:emb_idx_end]
                update to history label: self.Y_fullhist[label_stt:_end]
        Case-3
            If the number of embeddings is decreased compared to the last trial,
            then skip embedding merging.

        """
        segment_indexes_mat = np.array(self.segment_indexes[self.base_scale_index]).astype(int)
        self.total_segments_processed_count = segment_indexes_mat[-1] + 1
        hist_curr_boundary = self.total_segments_processed_count - self.current_n
        new_emb_n, emb_hist = None, None
        update_speaker_register = True

        # Case-1: The very first step
        if len(self.history_embedding_buffer_emb) == 0:
            new_emb_n = self.total_segments_processed_count - (self.current_n + self.history_n)
            hist_curr_boundary_emb_idx = get_mapped_index(segment_indexes_mat, hist_curr_boundary)
            emb_hist = emb_in[:hist_curr_boundary_emb_idx]
            self.pre_merge_cluster_label = self.Y_fullhist[:hist_curr_boundary]
        # Case-2: Number of embedding vectors is increased, need to update history and its label
        elif self.total_segments_processed_count > self.max_embed_count:
            label_stt, label_end = self.history_buffer_seg_end, hist_curr_boundary
            new_emb_n = label_end - label_stt
            assert new_emb_n > 0, "new_emb_n should be a positve integer number."
            emb_idx_stt = get_mapped_index(segment_indexes_mat, label_stt)
            emb_idx_end = get_mapped_index(segment_indexes_mat, label_end)
            emb_hist = np.vstack((self.history_embedding_buffer_emb, emb_in[emb_idx_stt:emb_idx_end]))
            self.pre_merge_cluster_label = np.hstack(
                (self.history_embedding_buffer_label, self.Y_fullhist[label_stt:label_end])
            )
        # Case-3: Number of embedding vectors is decreased
        # There will be no embedding update, so new_emb_n, emb_hist should be None
        else:
            update_speaker_register = False
        self.history_buffer_seg_end = hist_curr_boundary
        return update_speaker_register, new_emb_n, emb_hist

    def make_constant_length_emb(self, emb_in):
        """
        Edge case when the number of segments decreases and the number of embedding falls short for the labels.
        ASR decoder occasionally returns less number of words compared to the previous frame. In this case,
        we obtain fewer embedding vectors for the short period of time. To match the pre-defined length, yhe last
        embedding vector is repeated to fill the voidness. The repeated embedding will be soon replaced by the actual
        embeddings once the system takes new frames.
        """
        segment_indexes_mat = np.array(self.segment_indexes[self.base_scale_index]).astype(int)
        curr_clustered_segments = np.where(segment_indexes_mat >= self.history_buffer_seg_end)[0]
        if emb_in[curr_clustered_segments].shape[0] < self.current_n:
            delta_count = self.current_n - emb_in[curr_clustered_segments].shape[0]
            fill_in_emb = np.tile(emb_in[curr_clustered_segments][-1], (delta_count, 1))
            emb_curr = np.vstack((emb_in[curr_clustered_segments], fill_in_emb))
        else:
            emb_curr = emb_in[curr_clustered_segments]
        return emb_curr

    def run_reducer(self, emb_hist, mat, spk_idx, target_num):
        """

        """
        ndx = np.where(self.pre_merge_cluster_label == spk_idx)[0]
        if target_num > 0:
            cmat = np.tril(mat[:, ndx][ndx, :])
            tick2d = get_indices_for_merging(cmat, ndx, target_num)
            spk_cluster_labels, emb_ndx = self.pre_merge_cluster_label[ndx], emb_hist[ndx]
            result_emb, merged_cluster_labels = run_reduction_alg(cmat, tick2d, emb_ndx, spk_cluster_labels)
            assert (ndx.shape[0] - target_num) == result_emb.shape[0], "Reducer output is not matched to target quantity"
        else:
            result_emb = emb_hist[ndx]
            merged_cluster_labels = self.pre_merge_cluster_label[ndx]
        return result_emb, merged_cluster_labels

    def reduce_embedding_sets(self, emb_in, mat):
        """

        Example:
            self.history_n = 10
            self.current_n = 20

        Step (1)
        |-----------|ABCDEF--------------|

        If we get two more segments, "NN" as in the description:
        history buffer = 10
        current buffer = 22

        Step (2)
        |-----------|ABCDEF--------------XY|

        The newly accepted embeddings go through a queue (first embedding, first merged)
        history buffer = 12
        current buffer = 20

        Step (3)
        |-----------AB|CDEF--------------XY|

        After merging (reducing) the embedding set:
        history buffer = 10
        current buffer = 20

        Step(3)
        |==========|CDEF--------------XY|

        After clustering:

        |0000011111|11110000110010010011|

        This label is self.Y_fullhist (shape is (history_n + current_n) )

        self.history_buffer_seg_end (int):
            The total number of segments that have been merged from the beginning of the session.
            (=hist_curr_boundary)

        """
        update_speaker_register, new_emb_n, emb_hist = self.prepare_embedding_update(emb_in)

        # Update the history/current_buffer boundary cursor
        total_emb, total_cluster_labels = [], []

        if update_speaker_register:
            class_target_vol = get_merge_quantity(
                new_emb_n=new_emb_n,
                pre_merge_cluster_label=self.pre_merge_cluster_label,
                min_segs_per_buffer=self._minimum_segments_per_buffer,
            )
            # Merge the segments in the history buffer
            for spk_idx, target_num in enumerate(list(class_target_vol)):
                result_emb, merged_cluster_labels = self.run_reducer(emb_hist, mat, spk_idx, target_num)
                total_emb.append(result_emb)
                total_cluster_labels.append(merged_cluster_labels)
            self.history_embedding_buffer_emb = np.vstack(total_emb)
            self.history_embedding_buffer_label = np.hstack(total_cluster_labels)
            assert (
                self.history_embedding_buffer_emb.shape[0] == self.history_n
            ), f"History embedding size is not maintained correctly."
        else:
            total_emb.append(self.history_embedding_buffer_emb)
            total_cluster_labels.append(self.history_embedding_buffer_label)

        # `emb_curr` is the incumbent set of embeddings which is the the latest.
        emb_curr = self.make_constant_length_emb(emb_in)
        total_emb.append(emb_curr)

        # Before perform clustering, we attach the current_n number of estimated speaker labels
        # from the previous clustering result.
        total_cluster_labels.append(self.Y_fullhist[-self.current_n :])

        history_and_current_emb = np.vstack(total_emb)
        history_and_current_labels = np.hstack(total_cluster_labels)
        assert history_and_current_emb.shape[0] == len(history_and_current_labels)

        self.max_embed_count = max(
            self.total_segments_processed_count, self.max_embed_count
        )
        return history_and_current_emb, history_and_current_labels, update_speaker_register

    @timeit
    def get_reduced_mat(self, mat, emb):
        """
        Choose whether we want to add embeddings to the memory or not.
        """
        margin_seg_n = mat.shape[0] - (self.current_n + self.history_n)
        if margin_seg_n > 0:
            self.isOnline = True
            mat = preprocess_mat(mat, symm=True, fill_diag_zero=True)
            merged_emb, cluster_labels, add_new = self.reduce_embedding_sets(emb, mat)
            assert merged_emb.shape[0] == len(cluster_labels)
        else:
            self.isOnline = False
            merged_emb = emb
            cluster_labels, add_new = None, True
        return merged_emb, cluster_labels, add_new

    def macth_labels(self, org_mat, Y_new, add_new):
        """
        self.history_buffer_seg_end is a timestamp that tells to which point is history embedding contains from self.Y_fullhist.
        If embedding reducing is done correctly, we should discard  (0, self.history_n) amount and take
        (self.history_n, len(Y_new) ) from the new clustering output Y_new.

        Args:
            org_mat (Tensor):

            Y_new (Tensor):

            add_new (bool):
                This variable indicates whether there is a new set of segments. Depending on the VAD timestamps,
                the number of subsegments can be ocassionally decreased. If `add_new=True`, then it adds the newly
                acquired cluster labels.

        """
        if self.isOnline:
            # Online clustering mode with history buffer
            y_new_update_start = self.history_n
            Y_old = np.hstack((self.history_embedding_buffer_label, 
                               self.Y_fullhist[self.history_buffer_seg_end:])).astype(int)
            Y_matched = stitch_cluster_labels(Y_old=Y_old, Y_new=Y_new, with_history=True)
            if add_new:
                assert Y_matched[y_new_update_start:].shape[0] == self.current_n, "Update point sync is not correct."
                Y_out = np.hstack((self.Y_fullhist[: self.history_buffer_seg_end], Y_matched[y_new_update_start:]))
                self.Y_fullhist = Y_out
            else:
                # Do not update cumulative labels since there are no new segments.
                Y_out = self.Y_fullhist[: org_mat.shape[0]]
        else:
            # If no memory is used, offline clustering is applied.
            Y_out = stitch_cluster_labels(Y_old=self.Y_fullhist, Y_new=Y_new, with_history=False)
            self.Y_fullhist = Y_out
        return Y_out

    def remove_old_data(self, scale_idx):
        """
        Calculate how many segments should be removed from memory.
        """
        scale_buffer_size = int(
            len(set(self.scale_mapping_dict[scale_idx].tolist()))
            / len(set(self.scale_mapping_dict[self.base_scale_index].tolist()))
            * (self.history_n + self.current_n)
        )
        keep_range = scale_buffer_size + self.memory_margin
        self.embs_array[scale_idx] = self.embs_array[scale_idx][-keep_range:]
        self.segment_raw_audio[scale_idx] = self.segment_raw_audio[scale_idx][-keep_range:]
        self.segment_range_ts[scale_idx] = self.segment_range_ts[scale_idx][-keep_range:]
        self.segment_indexes[scale_idx] = self.segment_indexes[scale_idx][-keep_range:]
    
    @timeit    
    def temporal_label_major_vote(self):
        """
        Take a majority voting for every segment on temporal steps. This feature significantly reduces the error coming
        from unstable speaker counting in the beginning of sessions.
        """
        self.maj_vote_labels = []
        for seg_idx in self.memory_segment_indexes[self.base_scale_index]:
            if seg_idx not in self.base_scale_label_dict:
                self.base_scale_label_dict[seg_idx] = [self.memory_cluster_labels[seg_idx]]
            else:
                while len(self.base_scale_label_dict[seg_idx]) > self.temporal_label_major_vote_buffer_size:
                    self.base_scale_label_dict[seg_idx].pop(0)
                self.base_scale_label_dict[seg_idx].append(self.memory_cluster_labels[seg_idx])

            self.maj_vote_labels.append(torch.mode(torch.tensor(self.base_scale_label_dict[seg_idx]))[0].item())
        return self.maj_vote_labels

    def save_history_data(
        self, scale_idx, total_cluster_labels
    ):
        """
        Clustering is done for (hist_N + curr_N) number of embeddings. Thus, we need to remove the clustering results on
        the embedding memory. If self.diar.history_buffer_seg_end is not None, that indicates streaming diarization system
        is starting to save embeddings to its memory. Thus, the new incoming clustering label should be separated.
        If `isOnline = True`, old embeddings outside the window are removed to save GPU memory.
        """
        total_cluster_labels = total_cluster_labels.tolist()
        
        if not self.isOnline:
            self.memory_segment_ranges[scale_idx] = copy.deepcopy(self.segment_range_ts[scale_idx])
            self.memory_segment_indexes[scale_idx] = copy.deepcopy(self.segment_indexes[scale_idx])
            if scale_idx == self.base_scale_index:
                self.memory_cluster_labels = copy.deepcopy(total_cluster_labels)
        
        # Only if there are newly obtained embeddings, update ranges and embeddings.
        elif self.segment_indexes[scale_idx][-1] > self.memory_segment_indexes[scale_idx][-1]:
            global_idx = max(self.memory_segment_indexes[scale_idx]) - self.memory_margin

            # convert global index global_idx to buffer index buffer_idx
            segment_indexes_mat = np.array(self.segment_indexes[scale_idx]).astype(int)
            buffer_idx = get_mapped_index(segment_indexes_mat, global_idx)

            self.memory_segment_ranges[scale_idx][global_idx:] = \
                    copy.deepcopy(self.segment_range_ts[scale_idx][buffer_idx:])
            self.memory_segment_indexes[scale_idx][global_idx:] = \
                    copy.deepcopy(self.segment_indexes[scale_idx][buffer_idx:])
            if scale_idx == self.base_scale_index:
                self.memory_cluster_labels[global_idx:] = copy.deepcopy(total_cluster_labels[global_idx:])
                assert len(self.memory_cluster_labels) == len(self.memory_segment_ranges[scale_idx])

            # Remove unnecessary old values
            self.remove_old_data(scale_idx)
        
        assert len(self.embs_array[scale_idx]) == \
               len(self.segment_raw_audio[scale_idx]) == \
               len(self.segment_indexes[scale_idx]) == \
               len(self.segment_range_ts[scale_idx])
        
        if self.use_temporal_label_major_vote:
            cluster_label_hyp = self.temporal_label_major_vote()
        else:
            cluster_label_hyp = self.memory_cluster_labels
        return cluster_label_hyp
       
    
    def _run_segmentation(
        self, audio_buffer, vad_timestamps, segment_raw_audio, segment_range_ts, segment_indexes, window, shift
    ):
        """
        Remove the old segments that overlap with the new frame (self.frame_start)
        cursor_for_old_segments is pointing at the onset of the t_range popped most recently.

        Frame is in the middle of the buffer.

        |___Buffer___[   Frame   ]___Buffer___|

        """
        if self.buffer_start >= 0:
            if segment_raw_audio == [] and vad_timestamps != []:
                vad_timestamps[0][0] = max(vad_timestamps[0][0], 0.0)
                speech_labels_for_update = copy.deepcopy(vad_timestamps)
                self.cumulative_speech_labels = speech_labels_for_update

            else:
                cursor_for_old_segments = get_new_cursor_for_update(
                    self.frame_start, segment_raw_audio, segment_range_ts, segment_indexes
                )

                speech_labels_for_update, self.cumulative_speech_labels = get_speech_labels_for_update(
                    self.frame_start,
                    self.buffer_end,
                    self.cumulative_speech_labels,
                    vad_timestamps,
                    cursor_for_old_segments,
                )

            audio_buffer = copy.deepcopy(audio_buffer)
            sigs_list, sig_rangel_list, sig_indexes = get_segments_from_buffer(
                self.buffer_start,
                self.buffer_end,
                self.sample_rate,
                speech_labels_for_update,
                audio_buffer,
                segment_indexes,
                window,
                shift,
                self.decimals,
            )
            segment_raw_audio.extend(sigs_list)
            segment_range_ts.extend(sig_rangel_list)
            segment_indexes.extend(sig_indexes)
        assert len(segment_raw_audio) == len(segment_range_ts) == len(segment_indexes)
        segment_ranges_str = [f'{start:.3f} {end:.3f} ' for (start, end) in segment_range_ts]
        return segment_ranges_str

    @torch.no_grad()
    def run_embedding_extractor(self, audio_signal):
        audio_signal = torch.stack(audio_signal).float().to(self.device)
        audio_signal_lens = torch.from_numpy(
            np.array([self.n_embed_seg_len for k in range(audio_signal.shape[0])])
        ).to(self.device)
        _, torch_embs = self._speaker_model.forward(
            input_signal=audio_signal, input_signal_length=audio_signal_lens
        )
        return torch_embs

    @timeit
    def _extract_embeddings(self, audio_signal, segment_ranges, indexes, embeddings):
        """
        Extract speaker embeddings based on audio_signal and segment_ranges varialbes. Unlike offline speaker diarization,
        speaker embedding and subsegment ranges are not saved on the disk.

        Args:
            embeddings (Tensor):
            audio_signal (Tensor):
            segment_ranges(Tensor):
        Returns:
            embeddings (Tensor):
        """
        target_segment_count = len(segment_ranges)
        
        stt_idx = 0 if embeddings is None else embeddings.shape[0]
        end_idx = len(segment_ranges)

        if end_idx > stt_idx:
            torch_embs = self.run_embedding_extractor(audio_signal[stt_idx:end_idx])
            if embeddings is None:
                embeddings = torch_embs
            else:
                embeddings = torch.vstack((embeddings[:stt_idx, :], torch_embs))
        elif end_idx < stt_idx:
            embeddings = embeddings[:len(segment_ranges)]

        assert len(segment_ranges) == embeddings.shape[0], "Segment ranges and embeddings shapes do not match."
        return embeddings
    
    @timeit
    def perform_online_clustering(
        self,
        uniq_embs_and_timestamps,
        oracle_num_speakers=None,
        cuda=False,
    ):
        device = torch.device("cuda") if cuda else torch.device("cpu")

        # Get base-scale (the highest index) information from uniq_embs_and_timestamps.
        _mat, _emb, self.scale_mapping_dict = getTempInterpolMultiScaleCosAffinityMatrix(
            uniq_embs_and_timestamps, device
        )

        org_mat = copy.deepcopy(_mat)
        _emb, _mat = _emb.cpu().numpy(), _mat.cpu().numpy()
        emb, reduced_labels, add_new = self.get_reduced_mat(_mat, _emb)
        emb = torch.tensor(emb).to(device)
        Y_clus = self.online_clus.onlineCOSclustering(emb=emb,
                                                      frame_index=self.frame_index,
                                                      cuda=True,
                                                      device=device,
                                                     )

        Y_clus = Y_clus.cpu().numpy()
        total_cluster_labels = self.macth_labels(org_mat, Y_clus, add_new)
        return total_cluster_labels

    def get_interim_output(self, vad_ts):
        """
        In case buffer is not filled or there is no speech activity in the input, generate temporary output. 
        Args:
            vad_ts (list):

        """
        if len(self.memory_cluster_labels) == 0 or self.buffer_start < 0:
            return generate_cluster_labels([[0.0, self.total_buffer_in_secs]], [0])
        else:
            return generate_cluster_labels(self.memory_segment_ranges[self.base_scale_index], self.memory_cluster_labels)

    @timeit
    def diarize_step(self, audio_buffer, vad_ts):
        """
        See function `diarize()` in `ClusteringDiarizer` class.

        Args:
            audio_buffer (np.ndarray):

            vad_ts (list):
                List containing VAD timestamps

        Returns:
            diar_hyp (list):


        """
        # In case buffer is not filled or there is no speech activity in the input
        if self.buffer_start < 0 or len(vad_ts) == 0:
            return self.get_interim_output(vad_ts)
        
        # Segmentation: (c.f. see `diarize` function in ClusteringDiarizer class)
        for scale_idx, (window, shift) in self.multiscale_args_dict['scale_dict'].items():
            
            # Get subsegments for embedding extraction.
            segment_ranges_str = self._run_segmentation(
                audio_buffer,
                vad_ts,
                self.segment_raw_audio[scale_idx],
                self.segment_range_ts[scale_idx],
                self.segment_indexes[scale_idx],
                window,
                shift,
            )

            # Extract speaker embeddings from the extracted subsegment timestamps.
            embeddings = self._extract_embeddings(
                self.segment_raw_audio[scale_idx],
                self.segment_range_ts[scale_idx],
                self.segment_indexes[scale_idx],
                self.embs_array[scale_idx],
            )
            
            # Save the embeddings and segmentation timestamps in memory 
            self.embs_array[scale_idx] = embeddings
            self.multiscale_embeddings_and_timestamps[scale_idx] = [
                {self.uniq_id: embeddings},
                {self.uniq_id: segment_ranges_str},
            ]

        embs_and_timestamps = get_embs_and_timestamps(
            self.multiscale_embeddings_and_timestamps, self.multiscale_args_dict
        )
        
        # Clustering: Perform an online version of clustering algorithm
        total_cluster_labels = self.perform_online_clustering(
            embs_and_timestamps[self.uniq_id],
            cuda=True,
        )
        
        for scale_idx, (window, shift) in self.multiscale_args_dict['scale_dict'].items():
            cluster_label_hyp = self.save_history_data(scale_idx, total_cluster_labels)
        
        # Generate RTTM style diarization labels from segment ranges and cluster labels
        diar_hyp = generate_cluster_labels(self.memory_segment_ranges[self.base_scale_index], cluster_label_hyp)
        return diar_hyp
   
