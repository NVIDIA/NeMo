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
    OnlineSpeakerClustering,
    NMESC,
    getCosAffinityMatrix,
    split_input_data,
    getTempInterpolMultiScaleCosAffinityMatrix,
)
# from nemo.collections.asr.metric.der import score_labels
from nemo.collections.asr.parts.utils.speaker_utils import (
    audio_rttm_map,
    fl2int,
    generate_cluster_labels,
    get_contiguous_stamps,
    get_embs_and_timestamps,
    get_subsegments,
    get_uniqname_from_filepath,
    combine_float_overlaps,
    getMergedRanges,
    getOverlapRange,
    getSubRangeList,
    int2fl,
    isOverlap,
    labels_to_pyannote_object,
    labels_to_rttmfile,
    merge_stamps,
    rttm_to_labels,
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

def torch_adaptor(method):
    def torch_adapt(*args, **kw):
        args_out = []
        for x in args:
            args_out.append(torch.tensor(x))

        returns = method(*args_out)
        list_returns = []
        for x in args:
            list_returns.append(x.cpu().numpy().tolist())
        return list_returns
    return torch_adapt

def timeit(method):
    """
    Monitor elapsed time of the corresponding function displaying the method name.
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            logging.info('%2.2fms %r'%((te - ts) * 1000, method.__name__))
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

def get_closest_embeddings(cmat, ndx, target_num):
    """
    Get indeces of the embeddings we want to merge or drop.

    Args:
        cmat: (np.array)
        ndx: (np.array)
        target_num: (int)

    Output:
        index_2d: (numpy.array)
    """
    comb_limit = int(ndx.shape[0] / 2)
    assert (
        target_num <= comb_limit
    ), f" target_num is {target_num}: {target_num} is bigger than comb_limit {comb_limit}"

    cmat = preprocess_mat(cmat, symm=True, fill_diag_zero=True)
    if np.trace(cmat) > 0:
        raise ValueError("Trace of the affinity matrix should be 0 to exclude the self-affinity values.")
    
    # Sort the index to get the indices of the closest embedding vectors
    idx2d = np.unravel_index(np.argsort(cmat, axis=None)[::-1], cmat.shape)
    num_of_lower_half = int((cmat.shape[0] ** 2 - cmat.shape[0]) / 2)
    idx2d = (idx2d[0][:num_of_lower_half], idx2d[1][:num_of_lower_half])

    # Until we get the targeted number, add the closest indices
    cdx, left_set, right_set, total_set = 0, [], [], []
    while len(left_set) < target_num and len(right_set) < target_num:
        Ldx, Rdx = idx2d[0][cdx], idx2d[1][cdx]
        if (not Ldx in total_set) and (not Rdx in total_set):
            left_set.append(Ldx)
            right_set.append(Rdx)
            total_set = left_set + right_set
        cdx += 1
    index_2d = np.array([left_set, right_set])
    return index_2d


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
    pre_clus_labels, 
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

        pre_clus_labels: (np.array)
            the speaker labels of (the history_embedding_buffer_emb) + (the new embeddings to be added)
    """
    targeted_total_n = new_emb_n
    count_dict = Counter(pre_clus_labels)
    spk_freq_count = np.bincount(pre_clus_labels)
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


def merge_emb(index_2d, 
              emb_ndx, 
              pre_cluster_labels
            ):
    """


    """
    LI, RI = index_2d[0, :], index_2d[1, :]
    LI_argdx = index_2d[0].argsort()
    LI, RI = LI[LI_argdx], RI[LI_argdx]
    result_emb = 0.5 * (emb_ndx[LI, :] + emb_ndx[RI, :])
    merged_clus_labels = pre_cluster_labels[np.array(list(set(LI)))]
    bypass_ndx = np.array(list(set(range(emb_ndx.shape[0])) - set(list(LI) + list(RI))))
    if len(bypass_ndx) > 0:
        result_emb = np.vstack((emb_ndx[bypass_ndx], result_emb))
        merged_clus_labels = np.hstack((pre_cluster_labels[bypass_ndx], merged_clus_labels))
    return result_emb, merged_clus_labels

def run_reducer(pre_embs, target_spk_idx, target_num, pre_clus_labels):
    """
    Reduce the number of embedding vectors by merging the closest embedding vectors.
    This reducing algorithm is based on the assumption that the closest embeddings are the most redundant
    embedding vectors.

    Args:
        pre_embs (Tensor):
            Potential Embedding vectors to be merged
        affinity_mat (Tensor):
            The affinity matrix of the `pre_embs`
        target_spk_idx (int):
            The targeted speaker index for merging
        target_num (int):
            The count of embeddings to be reduced
        pre_clus_labels (list)
            The original cluster (speaker) index

    Returns:
        result_emb (Tensor):
            Set of merged embedding vectors
        merged_clus_labels (list):
            Cluster (speaker) labels for the merged embedding vectors
    """
    if pre_embs.shape[0] != pre_clus_labels.shape[0]:
            raise ValueError("Dimension mismatch between `pre_embs` and `pre_clus_labels`.")
    ndx = np.where(pre_clus_labels == target_spk_idx)[0]
    if target_num > 0:
        affinity_mat = getCosAffinityMatrix(torch.tensor(pre_embs)).cpu().numpy()
        cmat = np.tril(affinity_mat[:, ndx][ndx, :]) # Get the lower triangle of the affinity_mat array
        if cmat.shape[0] != ndx.shape[0]:
            raise ValueError("Dimension mismatch between targeted speaker affinity `cmat` and targeted speaker index `ndx`.")

        # Get the indices of the closest embedding vectors
        index_2d = get_closest_embeddings(cmat, ndx, target_num)
        spk_cluster_labels, emb_ndx = pre_clus_labels[ndx], pre_embs[ndx]

        # Merge the embeddings targeted by the 2-dim indices `index_2d`
        merged_embs, merged_clus_labels = merge_emb(index_2d, emb_ndx, spk_cluster_labels)
        assert (ndx.shape[0] - target_num) == merged_embs.shape[0], "Reducer output is not matched to the target quantity"
    else:
        merged_embs = pre_embs[ndx]
        merged_clus_labels = pre_clus_labels[ndx]
    return merged_embs, merged_clus_labels


def get_online_segments_from_slices(
    buffer_start,
    buffer_end,
    subsegments,
    ind_offset,
    sig,
    window,
    sample_rate,
    decimals: int = 3,
):
    """
    Create short speech segments from sclices for online processing purpose.

    Args:
        slices (int): the number of slices to be created
        slice_length (int): the lenghth of each slice
        shift (int): the amount of slice window shift
        sig (FloatTensor): the tensor that contains input signal

    Returns:
        sigs_list  (list):
            list of sliced input signal
        audio_lengths (list):
            list of audio sample lengths
    """
    sig_rangel_list, sig_indexes, sigs_list = [], [], []
    slice_length = int(window * sample_rate)
    for (start_sec, dur) in subsegments:
        if start_sec > buffer_end:
            continue
        ind_offset += 1

        buffer_len = buffer_end - buffer_start
        end_sec = start_sec + dur

        if end_sec > buffer_len:
            end_sec = min(end_sec, buffer_len)

        signal = get_target_sig(sig, start_sec, end_sec, slice_length, sample_rate)
        
        if len(signal) == 0:
            raise ValueError("len(signal) is zero. Signal length should not be zero.")
        if len(signal) < slice_length:
            signal = repeat_signal(signal, len(signal), slice_length)
        
        start_abs_sec = round(float(buffer_start + start_sec), decimals)
        end_abs_sec = round(float(buffer_start + end_sec), decimals)
        
        sigs_list.append(signal)
        sig_rangel_list.append([start_abs_sec, end_abs_sec])
        sig_indexes.append(ind_offset)
    if not len(sigs_list) == len(sig_rangel_list) == len(sig_indexes):
        raise ValueError("Signal information lists have a mismatch.")

    return ind_offset, sigs_list, sig_rangel_list, sig_indexes

def get_target_sig(sig, start_sec, end_sec, slice_length, sample_rate):
    start_idx = int(start_sec * sample_rate)
    end_idx = min(int(end_sec * sample_rate), slice_length + start_idx)
    return sig[start_idx:end_idx]


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

def check_ranges(range_list: List[List[float]]):
    """
    Check whether the range list has any faulty timestamp order.

    Args:
        range_list (list):
            List containing the start and end time of the segments.
            Example:
                >>> range_list = [[0.5, 3.12], [3.51, 7.26], ... ]
    """
    for range_tup in range_list:
        if range_tup[1] < range_tup[0]:
            raise ValueError("Range start time should be preceding the end time but we got: {range_tup}")


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
    speech_label_for_new_segments = combine_float_overlaps(update_overlap_speech_labels + new_incoming_speech_labels, margin=0)

    # Keep cumulative VAD labels for the future use
    cumulative_speech_labels = combine_float_overlaps(cumulative_speech_labels + new_incoming_speech_labels, margin=0)

    # Check if there are any faulty timestamps
    check_ranges(speech_label_for_new_segments)
    check_ranges(cumulative_speech_labels)
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
    decimals: int = 2,
):
    sigs_list, sig_rangel_list, sig_indexes = [], [], []
    if len(segment_indexes) > 0:
        ind_offset = segment_indexes[-1]
    else:
        ind_offset = -1

    for idx, range_spl in enumerate(speech_labels_for_update):
        range_offs = [range_spl[0] - buffer_start, range_spl[1] - buffer_start]
        range_t = [max(0, range_offs[0]), range_offs[1] ]

        subsegments = get_subsegments(
            offset=range_t[0], window=window, shift=shift, duration=(range_t[1] - range_t[0])
        )
        target_sig = torch.from_numpy(audio_buffer)
        
        ind_offset, sigs, ranges, inds = get_online_segments_from_slices(
            buffer_start=buffer_start,
            buffer_end=buffer_end,
            subsegments=subsegments,
            sig=target_sig,
            window=window,
            ind_offset=ind_offset,
            sample_rate=sample_rate,
        )
        sigs_list.extend(sigs)
        sig_rangel_list.extend(ranges)
        sig_indexes.extend(inds)

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



class OnlineDiarizer(ClusteringDiarizer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        # Convert config to support Hydra 1.0+ instantiation
        self.uniq_id = None
        self.decimals = 2
        self.AUDIO_RTTM_MAP = audio_rttm_map(self.cfg.diarizer.manifest_filepath)
        self.sample_rate = self.cfg.sample_rate
        self._cfg_diarizer = self.cfg.diarizer
        torch.manual_seed(0)
        
        self._out_dir = self._cfg_diarizer.out_dir
        if not os.path.exists(self._out_dir):
            os.mkdir(self._out_dir)

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.reset()
        
        # Set speaker embedding model in eval mode
        self._speaker_model.eval()
    
    def _init_online_clustering_module(self, clustering_params): 
        self.online_clus = OnlineSpeakerClustering(
                            max_num_speakers=clustering_params.max_num_speakers,
                            max_rp_threshold=clustering_params.max_rp_threshold,
                            sparse_search_volume=clustering_params.sparse_search_volume
                            )
                                
        self.max_num_speakers = self.online_clus.max_num_speaker
        self.base_scale_index = max(self.multiscale_args_dict['scale_dict'].keys())
        
    def _init_memory_buffer_variables(self):
        self.memory_segment_ranges = {key: [] for key in self.multiscale_args_dict['scale_dict'].keys()}
        self.memory_segment_indexes = {key: [] for key in self.multiscale_args_dict['scale_dict'].keys()}
        self.memory_cluster_labels = np.array([])
        self.Y_fullhist = []
        self.cumulative_speech_labels = []

        self.n_embed_seg_len = int(self.sample_rate * self.multiscale_args_dict['scale_dict'][self.base_scale_index][0])
        self.max_embed_count = 0

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
    
    def reset(self):
        """
        Reset all the variables
        """
        self._init_segment_variables()
        self._init_online_clustering_module(self._cfg_diarizer.clustering.parameters)
        self._init_memory_buffer_variables()
        self._init_temporal_major_voting_module()
        self._init_buffer_frame_timestamps()

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
        This function performs the following tasks:
            1. Decide whether to extract more embeddings or not (by setting `update_speaker_register`)
        If we need update:
            2. Calculate how many embeddings should be updated (set `new_emb_n` variable)
            3. Update history embedding vectors and save it to `pre_embs`.

        We only save the index and clustering label of each embedding.

        - Case-1: The very first step
            This else statement is for the very first diarization loop.
            This is the very first reduction frame.

        - Case-2: Number of embedding vectors is increased, therefore we need to update.
            Since there are new embeddings, we push the same amount (new_emb_n)
            of old embeddings to the history buffer.
            We should also update self.history_buffer_seg_end which is a pointer.
                update to history emb: emb_in[emb_idx_stt:emb_idx_end]
                update to history label: self.Y_fullhist[label_stt:_end]

        - Case-3: Number of embedding vectors is decreased
            If the number of embeddings is decreased compared to the last trial,
            then skip embedding merging.

        Args:
            emb_in (Tensor):

        Returns:
            update_speaker_register (bool):
            new_emb_n (int):
            pre_embs (Tensor):
        """
        segment_indexes_mat = np.array(self.segment_indexes[self.base_scale_index]).astype(int)
        self.total_segments_processed_count = segment_indexes_mat[-1] + 1
        hist_curr_boundary = self.total_segments_processed_count - self.current_n
        new_emb_n, pre_embs = None, None
        update_speaker_register = True

        # Case-1: The very first step
        if len(self.history_embedding_buffer_emb) == 0:
            new_emb_n = self.total_segments_processed_count - (self.current_n + self.history_n)
            hist_curr_boundary_emb_idx = get_mapped_index(segment_indexes_mat, hist_curr_boundary)
            pre_embs = emb_in[:hist_curr_boundary_emb_idx]
            self.pre_clus_labels = self.Y_fullhist[:hist_curr_boundary]

        # Case-2: Number of embedding vectors is increased, need to update history and its label
        elif self.total_segments_processed_count > self.max_embed_count:
            
            # Calculate the number of new embedding vectors
            label_stt, label_end = self.history_buffer_seg_end, hist_curr_boundary
            new_emb_n = label_end - label_stt
            assert new_emb_n > 0, "new_emb_n should be a positve integer number."
            
            # Add embedding vectors to `pre_embs` so that we can merge it with reducer function.
            emb_idx_stt = get_mapped_index(segment_indexes_mat, label_stt)
            emb_idx_end = get_mapped_index(segment_indexes_mat, label_end)
            pre_embs = np.vstack((self.history_embedding_buffer_emb, emb_in[emb_idx_stt:emb_idx_end]))
            self.pre_clus_labels = np.hstack(
                (self.history_embedding_buffer_label, self.Y_fullhist[label_stt:label_end])
            )

        # Case-3: Number of embedding vectors is decreased
        # There will be no embedding update, so new_emb_n, pre_embs should be None
        else:
            update_speaker_register = False

        # Update the history buffer index
        self.history_buffer_seg_end = hist_curr_boundary

        return update_speaker_register, new_emb_n, pre_embs

    def make_constant_length_emb(self, emb_in):
        """
        - Edge case when the number of segments decreases and the number of embedding falls short for the labels.
        - ASR decoder occasionally returns less number of words compared to the previous frame.
        - In this case, we obtain fewer embedding vectors for the short period of time. To match the pre-defined
          length, the last embedding vector is repeated to fill the voidness.
        - The repeated embedding will be soon replaced by the actual embeddings once the system takes new frames.
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


    def reduce_embedding_sets(self, emb_in):
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

            Step (4)
            |==========|CDEF--------------XY|

            After clustering:

            |0000011111|11110000110010010011|

            This label is self.Y_fullhist (shape is (history_n + current_n) )

        self.history_buffer_seg_end (int):
            The total number of segments that have been merged from the beginning of the session.
            (=hist_curr_boundary)

        """
        update_speaker_register, new_emb_n, pre_embs = self.prepare_embedding_update(emb_in)

        # Update the history/current_buffer boundary cursor
        total_emb, total_cluster_labels = [], []

        if update_speaker_register:
            
            # Calculate how many embedding vectors should be reduced per speaker
            class_target_vol = get_merge_quantity(new_emb_n=new_emb_n,
                                                  pre_clus_labels=self.pre_clus_labels,
                                                  min_segs_per_buffer=self._minimum_segments_per_buffer)
            
            # Merge the segments in the history buffer
            for spk_idx, target_num in enumerate(list(class_target_vol)):
                merged_embs, merged_clus_labels = run_reducer(pre_embs=pre_embs, 
                                                              target_spk_idx=spk_idx, 
                                                              target_num=target_num, 
                                                              pre_clus_labels=self.pre_clus_labels)
                total_emb.append(merged_embs)
                total_cluster_labels.append(merged_clus_labels)
            
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
    def _get_reduced_mat(self, emb):
        """
        Choose whether we want to add embeddings to the memory or not.
        """
        margin_seg_n = emb.shape[0] - (self.current_n + self.history_n)
        if margin_seg_n > 0:
            self.isOnline = True
            merged_emb, cluster_labels, add_new = self.reduce_embedding_sets(emb)
            assert merged_emb.shape[0] == len(cluster_labels)
        else:
            self.isOnline = False
            merged_emb = emb
            cluster_labels, add_new = None, True
        return merged_emb, cluster_labels, add_new

    def macth_labels(self, Y_new, add_new):
        """
        self.history_buffer_seg_end is a timestamp that tells to which point is history embedding contains from self.Y_fullhist.
        If embedding reducing is done correctly, we should discard  (0, self.history_n) amount and take
        (self.history_n, len(Y_new) ) from the new clustering output Y_new.

        Args:
            Y_new (Tensor):

            add_new (bool):
                This variable indicates whether there is a new set of segments. Depending on the VAD timestamps,
                the number of subsegments can be ocassionally decreased. If `add_new=True`, then it adds the newly
                acquired cluster labels.

        """
        if self.isOnline:
            # Online clustering mode with history buffer
            Y_old = np.hstack((self.history_embedding_buffer_label, 
                               self.Y_fullhist[self.history_buffer_seg_end:])).astype(int)
            
            # Stitch the old history and new cluster labels
            Y_matched = stitch_cluster_labels(Y_old=Y_old, Y_new=Y_new, with_history=True)
            if add_new:
                if Y_matched[self.history_n:].shape[0] != self.current_n:
                    raise ValueError("Update point sync is not correct.")
                Y_out = np.hstack((self.Y_fullhist[:self.history_buffer_seg_end], Y_matched[self.history_n:]))
                self.Y_fullhist = Y_out
            else:
                # Do not update cumulative labels since there are no new segments.
                Y_out = self.Y_fullhist[:Y_new.shape[0]]
        else:
            # If no memory is used, offline clustering is applied.
            Y_out = stitch_cluster_labels(Y_old=self.Y_fullhist, Y_new=Y_new, with_history=False)
            self.Y_fullhist = Y_out
        return Y_out

    def _clear_memory(self, scale_idx):
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
    def _temporal_label_major_vote(self):
        """
        Take a majority voting for every segment on temporal steps. This feature significantly reduces the error coming
        from unstable speaker counting in the beginning of sessions.
        """
        maj_vote_labels = []
        for seg_idx in self.memory_segment_indexes[self.base_scale_index]:
            if seg_idx not in self.base_scale_label_dict:
                self.base_scale_label_dict[seg_idx] = [self.memory_cluster_labels[seg_idx]]
            else:
                while len(self.base_scale_label_dict[seg_idx]) > self.temporal_label_major_vote_buffer_size:
                    self.base_scale_label_dict[seg_idx].pop(0)
                self.base_scale_label_dict[seg_idx].append(self.memory_cluster_labels[seg_idx])

            maj_vote_labels.append(torch.mode(torch.tensor(self.base_scale_label_dict[seg_idx]))[0].item())
        return maj_vote_labels

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
            self._clear_memory(scale_idx)
        
        assert len(self.embs_array[scale_idx]) == \
               len(self.segment_raw_audio[scale_idx]) == \
               len(self.segment_indexes[scale_idx]) == \
               len(self.segment_range_ts[scale_idx])
        
        if self.use_temporal_label_major_vote:
            cluster_label_hyp = self._temporal_label_major_vote()
        else:
            cluster_label_hyp = self.memory_cluster_labels
        return cluster_label_hyp
    
    def _run_segmentation(
        self, 
        audio_buffer, 
        vad_timestamps, 
        segment_raw_audio, 
        segment_range_ts, 
        segment_indexes, 
        window, 
        shift
    ):
        """
        Remove the old segments that overlap with the new frame (self.frame_start)
        cursor_for_old_segments is pointing at the onset of the t_range popped most recently.

        Frame is in the middle of the buffer.

        |___Buffer___[   Frame   ]___Buffer___|

        """
        if self.buffer_start >= 0:
            # Check if this is the very first step
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
            
            # Collect the timeseries signal from the buffer
            sigs_list, sig_rangel_list, sig_indexes = get_segments_from_buffer(
                self.buffer_start,
                self.buffer_end,
                self.sample_rate,
                speech_labels_for_update,
                audio_buffer,
                segment_indexes,
                window,
                shift,
            )
            segment_raw_audio.extend(sigs_list)
            segment_range_ts.extend(sig_rangel_list)
            segment_indexes.extend(sig_indexes)

        if not len(segment_raw_audio) == len(segment_range_ts) == len(segment_indexes):
            raise ValueError("Segment information has a mismatch in length.")
        return segment_range_ts

    @torch.no_grad()
    def _run_embedding_extractor(self, audio_signal):
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
            torch_embs = self._run_embedding_extractor(audio_signal[stt_idx:end_idx])
            if embeddings is None:
                embeddings = torch_embs
            else:
                embeddings = torch.vstack((embeddings[:stt_idx, :], torch_embs))
        elif end_idx < stt_idx:
            embeddings = embeddings[:len(segment_ranges)]

        if len(segment_ranges) != embeddings.shape[0]: 
            raise ValueError("Segment ranges and embeddings shapes do not match.")
        return embeddings
    
    @timeit
    def _perform_online_clustering(
        self,
        uniq_embs_and_timestamps,
        oracle_num_speakers=None,
        cuda=False,
    ):
        device = torch.device("cuda") if cuda else torch.device("cpu")

        # Get base-scale (the highest index) information from uniq_embs_and_timestamps.
        embeddings_in_scales, timestamps_in_scales = split_input_data(
                embeddings_in_scales=uniq_embs_and_timestamps['embeddings'],
                timestamps_in_scales=uniq_embs_and_timestamps['timestamps'],
                multiscale_segment_counts=uniq_embs_and_timestamps['multiscale_segment_counts'],
        )
       
        _emb, self.scale_mapping_dict = getTempInterpolMultiScaleCosAffinityMatrix(
            multiscale_weights=uniq_embs_and_timestamps['multiscale_weights'],
            embeddings_in_scales=embeddings_in_scales,
            timestamps_in_scales=timestamps_in_scales,
            device=device
        )

        _emb = _emb.cpu().numpy()
        emb, reduced_labels, add_new = self._get_reduced_mat(_emb)

        emb = torch.tensor(emb).to(device)
        Y_clus = self.online_clus.forward(emb=emb,
                                          frame_index=self.frame_index,
                                          cuda=True,
                                          device=device,
                                         )

        Y_clus = Y_clus.cpu().numpy()
        total_cluster_labels = self.macth_labels(Y_clus, add_new)
        return total_cluster_labels

    def _get_interim_output(self, vad_ts):
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
    def diarize_step(self, audio_buffer, vad_timestamps):
        """
        See function `diarize()` in `ClusteringDiarizer` class.

        1. Segmentation
        2. Embedding Extraction
        3. Online Clustering & Counting
        4. Generate speaker labels

        Args:
            audio_buffer (np.ndarray):

            vad_timestamps (list):
                List containing VAD timestamps

        Returns:
            diar_hyp (list):
        """
        # In case buffer is not filled or there is no speech activity in the input
        if self.buffer_start < 0 or len(vad_timestamps) == 0:
            return self._get_interim_output(vad_timestamps)
        
        # Segmentation: (c.f. see `diarize` function in ClusteringDiarizer class)
        for scale_idx, (window, shift) in self.multiscale_args_dict['scale_dict'].items():
            
            # Step 1: Get subsegments for embedding extraction.
            segment_ranges = self._run_segmentation(
                audio_buffer=audio_buffer,
                vad_timestamps=vad_timestamps,
                segment_raw_audio=self.segment_raw_audio[scale_idx],
                segment_range_ts=self.segment_range_ts[scale_idx],
                segment_indexes=self.segment_indexes[scale_idx],
                window=window,
                shift=shift,
            )

            # Step 2-1: Extract speaker embeddings from the extracted subsegment timestamps.
            embeddings = self._extract_embeddings(
                audio_signal=self.segment_raw_audio[scale_idx],
                segment_ranges=self.segment_range_ts[scale_idx],
                indexes=self.segment_indexes[scale_idx],
                embeddings=self.embs_array[scale_idx],
            )
            
            # Step 2-2:Save the embeddings and segmentation timestamps in memory 
            self.embs_array[scale_idx] = embeddings

            self.multiscale_embeddings_and_timestamps[scale_idx] = [
                {self.uniq_id: embeddings},
                {self.uniq_id: segment_ranges},
            ]
        
        embs_and_timestamps = get_embs_and_timestamps(
            self.multiscale_embeddings_and_timestamps, self.multiscale_args_dict
        )
        
        # Step 3: Clustering: Perform an online version of clustering algorithm
        total_cluster_labels = self._perform_online_clustering(
            embs_and_timestamps[self.uniq_id],
            cuda=True,
        )
        
        for scale_idx, (window, shift) in self.multiscale_args_dict['scale_dict'].items():
            cluster_label_hyp = self.save_history_data(scale_idx, total_cluster_labels)
        
        # Step 4: Generate RTTM style diarization labels from segment ranges and cluster labels
        diar_hyp = generate_cluster_labels(self.memory_segment_ranges[self.base_scale_index], cluster_label_hyp)
        return diar_hyp
   
