# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import re
import copy
import math
import random
import logging
import itertools
from copy import deepcopy
import concurrent.futures
from cytoolz import groupby
from collections import defaultdict
from typing import Dict, Optional, Tuple, List

import numpy as np
import soundfile
from tqdm import tqdm
from scipy.stats import norm

import torch.utils.data
from lhotse.cut.set import mix
from lhotse.cut import CutSet, MixedCut, MonoCut, MixTrack
from lhotse import SupervisionSet, SupervisionSegment, dill_enabled, AudioSource, Recording
from lhotse.utils import uuid4

def find_first_nonzero(mat, max_cap_val=-1):
    # non zero values mask
    non_zero_mask = mat != 0
    # operations on the mask to find first nonzero values in the rows
    mask_max_values, mask_max_indices = torch.max(non_zero_mask, dim=1)
    # if the max-mask is zero, there is no nonzero value in the row
    mask_max_indices[mask_max_values == 0] = max_cap_val
    return mask_max_indices

def sort_probs_and_labels(self, labels, discrete=True, thres=0.5, return_inds=False, accum_frames=1):
    max_cap_val = labels.shape[1] + 1
    labels_discrete = labels.clone()
    if not discrete:
        labels_discrete[labels_discrete < thres] = 0
        labels_discrete[labels_discrete >= thres] = 1
    modi =torch.ones(labels.shape[1],labels.shape[1]).triu().to(labels.device)
    labels_accum = torch.matmul(labels_discrete.permute(0,2,1),modi).permute(0,2,1)
    labels_accum[labels_accum < accum_frames] = 0
    label_fz = self.find_first_nonzero(labels_accum, max_cap_val)
    label_fz[label_fz == -1] = max_cap_val
    sorted_inds = torch.sort(label_fz)[1]
    sorted_labels = labels.transpose(0,1)[:, torch.arange(labels.shape[0]).unsqueeze(1), sorted_inds].transpose(0, 1)
    if return_inds:
        return sorted_labels, sorted_inds
    else:
        return sorted_labels

def sort_targets_with_preds_ats(labels, preds, speaker_permutations, thres=0.5, tolerance=0):
    """
    Sorts labels and predictions to get optimal of all arrival-time ordered permutations
    """
    labels_discrete = labels.clone()
    labels_discrete[labels_discrete < thres] = 0
    labels_discrete[labels_discrete >= thres] = 1

    nonzero_ind = find_first_nonzero(labels_discrete, labels.shape[1])
    sorted_values = torch.sort(nonzero_ind)[0] #indices of first speech frame for arrival-time ordered permutation (batch_size, max_num_of_spks)

    perm_size = speaker_permutations.shape[0]
    permed_labels = labels_discrete[:, :, speaker_permutations] #all possible permutations of discrete labels (batch_size, num_frames, perm_size, max_num_of_spks)
    permed_nonzero_ind = find_first_nonzero(permed_labels, labels.shape[1]) #indices of first speech frame for all permutations (batch_size, perm_size, max_num_of_spks)
    perm_compare = torch.abs(sorted_values.unsqueeze(1) - permed_nonzero_ind) <= tolerance #comparison with first speech frame indices of the ATS permutation (batch_size, perm_size, max_num_of_spks)
    perm_mask = torch.all(perm_compare, dim=2).float() #binary mask for all permutation, 1 means permutation is arrival-time ordered (batch_size, perm_size)

    preds_rep = torch.unsqueeze(preds, 2).repeat(1, 1, perm_size, 1)
    match_score = torch.sum(permed_labels * preds_rep, axis=1).sum(axis=2)
    batch_best_perm = torch.argmax(match_score * perm_mask, axis=1) # non-ATS permutations are excluded by mask
    rep_speaker_permutations = speaker_permutations.repeat(batch_best_perm.shape[0],1) # (batch_size * perm_size, max_num_of_spks)
    global_inds_vec = torch.arange(0, perm_size*batch_best_perm.shape[0], perm_size).to(batch_best_perm.device) + batch_best_perm
    batch_perm_inds = rep_speaker_permutations[global_inds_vec.to(rep_speaker_permutations.device), :] # (batch_size, max_num_of_spks)
    max_score_permed_labels = torch.vstack([ labels[k, :, batch_perm_inds[k]].unsqueeze(0) for k in range(batch_perm_inds.shape[0])])
    return max_score_permed_labels

def get_pil_target(labels, preds, speaker_permutations):
    """
    Sorts labels and predictions to get optimal permutation
    """
    perm_size = speaker_permutations.shape[0]
    permed_labels = labels[:, :, speaker_permutations]
    preds_rep = torch.unsqueeze(preds, 2).repeat(1,1, speaker_permutations.shape[0],1)
    match_score = torch.sum(permed_labels * preds_rep, axis=1).sum(axis=2)
    batch_best_perm = torch.argmax(match_score, axis=1)
    rep_speaker_permutations = speaker_permutations.repeat(batch_best_perm.shape[0],1) # (batch_size * perm_size, max_num_of_spks)
    global_inds_vec = torch.arange(0, perm_size*batch_best_perm.shape[0], perm_size).to(batch_best_perm.device) + batch_best_perm
    batch_perm_inds = rep_speaker_permutations[global_inds_vec.to(rep_speaker_permutations.device), :] # (batch_size, max_num_of_spks)
    max_score_permed_labels = torch.vstack([ labels[k, :, batch_perm_inds[k]].unsqueeze(0) for k in range(batch_perm_inds.shape[0])])
    return max_score_permed_labels

def apply_spk_mapping(diar_preds: torch.Tensor, spk_mappings: torch.Tensor) -> torch.Tensor:
    """ 
    Applies a speaker mapping to diar predictions.

    Args:
        diar_preds (Tensor): The diar predictions tensor.   
            Dimension: (batch_size, num_frames, num_speakers)
        spk_mappings (Tensor): The speaker mappings tensor.
            Dimension: (batch_size, num_speakers)
    
    Returns:
        permuted_diar_preds (Tensor): The permuted diar predictions tensor with the given speaker mappings.
    """
    expanded_mappings = spk_mappings.unsqueeze(1).expand(-1, diar_preds.size(1), -1)
    permuted_diar_preds = torch.gather(diar_preds, 2, expanded_mappings)
    return permuted_diar_preds

def shuffle_spk_mapping(cuts: list, num_speakers: int, shuffle_spk_mapping: bool = False, pattern= r'<\|spltoken\d+\|>') -> Tuple[CutSet, torch.Tensor]:
    """ 
    Applies a shuffle mapping to speaker text labels in the cuts.
    Example:
        Original cut.text:
            "<|spltoken0|> we do shuffle <|spltoken1|> and map speakers <|spltoken0|> yes <|spltoken2|> we keep dimensions" 
        Speaker Mapping: [3, 0, 1, 2]
        Shuffled cut.text:
            "<|spltoken3|> we do shuffle <|spltoken0|> and map speakers <|spltoken3|> yes <|spltoken1|> we keep dimensions" 

    Args:
        cuts (List[MonoCut, MixedCut]): A list of Cut instances.
        num_speakers (int): The total number of speakers.
        shuffle_spk_mapping (bool): Whether to shuffle the speaker mappings.
        pattern (str): A regular expression pattern for speaker tokens.

    Returns:
        cuts (list): The updated CutSet with shuffled speaker mappings.
        spk_mappings (Tensor): 
            If shuffle_speaker_mapping is True, shuffled speaker mappings in batch.
            If shuffle_speaker_mapping is False, speaker mappings in batch is not permuted and returns torch.arange() values.
    """ 
    batch_size = len(cuts) 
    if shuffle_spk_mapping:
        permuted_indices = torch.rand(batch_size, num_speakers).argsort(dim=1)
        spk_mappings = torch.gather(torch.arange(num_speakers).repeat(batch_size, 1), 1, permuted_indices)
        str_pattern = pattern.replace("\\", '')
        left_str, right_str = str_pattern.split('d+')[0], str_pattern.split('d+')[1]
        for idx, cut in enumerate(cuts):
            word_list = []
            for word in deepcopy(cut.text).split(): 
                if len(re.findall(pattern, word)) > 0:
                    spk_token_int = int(word.replace(left_str,'').replace(right_str, ''))
                    new_spk = spk_mappings[idx][spk_token_int]
                    word_list.append(f'{left_str}{new_spk}{right_str}')
                else:
                    word_list.append(word)
            cuts[idx].supervisions[0].text = ' '.join(word_list)
    else:
        spk_mappings = torch.arange(num_speakers).unsqueeze(0).repeat(batch_size, 1)
    return cuts, spk_mappings 

def find_segments_from_rttm(
        recording_id: str, 
        rttms, 
        start_after: float, 
        end_before: float, 
        adjust_offset: bool=True, 
        tolerance: float=0.001):
    """ 
    Finds segments from the given rttm file.
    This function is designed to replace rttm

    Args:
        recording_id (str): The recording ID in string format.
        rttms (SupervisionSet): The SupervisionSet instance.
        start_after (float): The start time after which segments are selected.
        end_before (float): The end time before which segments are selected.
        adjust_offset (bool): Whether to adjust the offset of the segments.
        tolerance (float): The tolerance for time matching. 0.001 by default.
    
    Returns:
        segments (List[SupervisionSegment]): A list of SupervisionSegment instances.
    """
    segment_by_recording_id = rttms._segments_by_recording_id
    if segment_by_recording_id is None:
        from cytoolz import groupby
        segment_by_recording_id = groupby(lambda seg: seg.recording_id, rttms)

    return [
            # We only modify the offset - the duration remains the same, as we're only shifting the segment
            # relative to the Cut's start, and not truncating anything.
            segment.with_offset(-start_after) if adjust_offset else segment
            for segment in segment_by_recording_id.get(recording_id, [])
            if segment.start < end_before + tolerance
            and segment.end > start_after + tolerance
        ]

def speaker_to_target(
    a_cut,
    num_speakers: int = 4, 
    num_sample_per_mel_frame: int = 160, 
    num_mel_frame_per_asr_frame: int = 8, 
    spk_tar_all_zero: bool = False,
    boundary_segments: bool = False,
    soft_label: bool = False,
    ignore_num_spk_mismatch: bool = True,
    soft_thres: float = 0.5,
    ):
    '''
    Get rttm samples corresponding to one cut, generate speaker mask numpy.ndarray with shape (num_speaker, hidden_length)
    This function is needed for speaker diarization with ASR model trainings.

    Args:
        a_cut (MonoCut, MixedCut): Lhotse Cut instance which is MonoCut or MixedCut instance.
        num_speakers (int): max number of speakers for all cuts ("mask" dim0), 4 by default
        num_sample_per_mel_frame (int): number of sample per mel frame, sample_rate / 1000 * window_stride, 160 by default (10ms window stride)
        num_mel_frame_per_asr_frame (int): encoder subsampling_factor, 8 by default
        spk_tar_all_zero (Tensor): set to True gives all zero "mask"
        boundary_segments (bool): set to True to include segments containing the boundary of the cut, False by default for multi-speaker ASR training
        soft_label (bool): set to True to use soft label that enables values in [0, 1] range, False by default and leads to binary labels.
        ignore_num_spk_mismatch (bool): This is a temporary solution to handle speaker mismatch. Will be removed in the future.
    
    Returns:
        mask (Tensor): speaker mask with shape (num_speaker, hidden_lenght)
    '''
    # get cut-related segments from rttms
    # basename = os.path.basename(a_cut.rttm_filepath).replace('.rttm', '')
    if isinstance(a_cut, MixedCut):
        cut_list = [track.cut for track in a_cut.tracks if isinstance(track.cut, MonoCut)]
        offsets = [track.offset for track in a_cut.tracks if isinstance(track.cut, MonoCut)]
    elif isinstance(a_cut, MonoCut):
        cut_list = [a_cut]
        offsets = [0]
    else:
        raise ValueError(f"Unsupported cut type type{a_cut}: only MixedCut and MonoCut are supported")
    
    segments_total = []
    for i, cut in enumerate(cut_list):
        rttms = SupervisionSet.from_rttm(cut.rttm_filepath)
        if boundary_segments: # segments with seg_start < total_end and seg_end > total_start are included
            segments_iterator = find_segments_from_rttm(recording_id=cut.recording_id, rttms=rttms, start_after=cut.start, end_before=cut.end, tolerance=0.0)
        else: # segments with seg_start > total_start and seg_end < total_end are included
            segments_iterator = rttms.find(recording_id=cut.recording_id, start_after=cut.start, end_before=cut.end, adjust_offset=True)

        for seg in segments_iterator:
            if seg.start < 0:
                seg.duration += seg.start
                seg.start = 0
            if seg.end > cut.duration:
                seg.duration -= seg.end - cut.duration
            seg.start += offsets[i]
            segments_total.append(seg)
    
    # apply arrival time sorting to the existing segments
    segments_total.sort(key = lambda rttm_sup: rttm_sup.start)

    seen = set()
    seen_add = seen.add
    speaker_ats = [s.speaker for s in segments_total if not (s.speaker in seen or seen_add(s.speaker))]
     
    speaker_to_idx_map = {
            spk: idx
            for idx, spk in enumerate(speaker_ats)
    }
    if len(speaker_to_idx_map) > num_speakers and not ignore_num_spk_mismatch:  # raise error if number of speakers
        raise ValueError(f"Number of speakers {len(speaker_to_idx_map)} is larger than the maximum number of speakers {num_speakers}")
        
    # initialize mask matrices (num_speaker, encoder_hidden_len)
    feat_per_sec = int(a_cut.sampling_rate / num_sample_per_mel_frame) # 100 by default
    num_samples = get_hidden_length_from_sample_length(a_cut.num_samples, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
    if spk_tar_all_zero: 
        frame_mask = torch.zeros((num_samples, num_speakers))
    else:
        frame_mask = get_mask_from_segments(segments_total, a_cut, speaker_to_idx_map, num_speakers, feat_per_sec, ignore_num_spk_mismatch)
    soft_mask = get_soft_mask(frame_mask, num_samples, num_mel_frame_per_asr_frame)

    if soft_label:
        mask = soft_mask
    else:
        mask = (soft_mask > soft_thres).float()

    return mask

def get_mask_from_segments(segments: list, a_cut, speaker_to_idx_map: torch.Tensor, num_speakers: int =4, feat_per_sec: int=100, ignore_num_spk_mismatch: bool = False):
    """ 
    Generate mask matrix from segments list.
    This function is needed for speaker diarization with ASR model trainings.
    
    Args:
        segments: A list of Lhotse Supervision segments iterator.
        cut (MonoCut, MixedCut): Lhotse MonoCut or MixedCut instance.
        speaker_to_idx_map (dict): A dictionary mapping speaker names to indices.
        num_speakers (int): max number of speakers for all cuts ("mask" dim0), 4 by default
        feat_per_sec (int): number of frames per second, 100 by default, 0.01s frame rate
        ignore_num_spk_mismatch (bool): This is a temporary solution to handle speaker mismatch. Will be removed in the future.
    
    Returns:
        mask (Tensor): A numpy array of shape (num_speakers, encoder_hidden_len).
            Dimension: (num_speakers, num_frames)
    """
    # get targets with 0.01s frame rate
    num_samples = round(a_cut.duration * feat_per_sec) 
    mask = torch.zeros((num_samples, num_speakers))
    for rttm_sup in segments:
        speaker_idx = speaker_to_idx_map[rttm_sup.speaker]
        if speaker_idx >= num_speakers:
            if ignore_num_spk_mismatch:
                continue
            else:
                raise ValueError(f"Speaker Index {speaker_idx} exceeds the max index: {num_speakers-1}")
        stt = max(rttm_sup.start, 0)
        ent = min(rttm_sup.end, a_cut.duration)
        stf = int(stt * feat_per_sec)
        enf = int(ent * feat_per_sec)
        mask[stf:enf, speaker_idx] = 1.0
    return mask

def get_soft_mask(feat_level_target, num_samples, stride):
    """
    Get soft mask from feat_level_target with stride.
    This function is needed for speaker diarization with ASR model trainings.
    
    Args:
        feat_level_target (Tensor): A numpy array of shape (num_frames, num_speakers).
            Dimension: (num_frames, num_speakers)
        num_sample (int): The total number of samples.
        stride (int): The stride for the mask.
        """

    num_speakers = feat_level_target.shape[1]
    mask = torch.zeros(num_samples, num_speakers)

    for index in range(num_samples):
        if index == 0:
            seg_stt_feat = 0
        else:
            seg_stt_feat = stride * index - 1 - int(stride / 2)
        if index == num_samples - 1:
            seg_end_feat = feat_level_target.shape[0]
        else:
            seg_end_feat = stride * index - 1 + int(stride / 2)
        mask[index] = torch.mean(feat_level_target[seg_stt_feat:seg_end_feat+1, :], axis=0)
    return mask

def get_hidden_length_from_sample_length(
    num_samples: int, 
    num_sample_per_mel_frame: int = 160, 
    num_mel_frame_per_asr_frame: int = 8
) -> int:
    """ 
    Calculate the hidden length from the given number of samples.
    This function is needed for speaker diarization with ASR model trainings.

    This function computes the number of frames required for a given number of audio samples,
    considering the number of samples per mel frame and the number of mel frames per ASR frame.

    Parameters:
        num_samples (int): The total number of audio samples.
        num_sample_per_mel_frame (int, optional): The number of samples per mel frame. Default is 160.
        num_mel_frame_per_asr_frame (int, optional): The number of mel frames per ASR frame. Default is 8.

    Returns:
        hidden_length (int): The calculated hidden length in terms of the number of frames.
    """
    mel_frame_count = math.ceil((num_samples + 1) / num_sample_per_mel_frame)
    hidden_length = math.ceil(mel_frame_count / num_mel_frame_per_asr_frame)
    return int(hidden_length)

class ConcatenationMeetingSimulator():
    """
    This simulator concatenates the segments from different/same sessions to create a
    multi-speaker meeting. 
    """

    def __init__(
        self,
        intra_session_concat_prob: float|List[float] = [0, 1.0, 0.5, 0.2],
        data_type: str = "msasr",
        min_duration: float = 30.0,
        max_duration: float = 40.0,
        max_num_speakers: int = 4,
        speaker_count_distribution: List[float] = [0, 2, 3, 4],
        skip_long_segments: bool = True,
        valid_dataset_ids: List[str] = [],
    ):
        """
        :param intra_session_concat_prob: the probability of concatenating segments from the same
            session. [Default: 1]
        :param data_type: the type of data to simulate. Either 'msasr' or 'diar'. If 'msasr',
            the transcripts are included in the simulation,and the boundary segments are 
            not included. [Default: 'msasr']
        :param max_duration: the maximum duration of the simulated meeting. [Default: 40.0]
        """
        super().__init__()
        if isinstance(intra_session_concat_prob, float):
            self.intra_session_concat_prob = [intra_session_concat_prob] * (max_num_speakers)
        elif len(intra_session_concat_prob) == max_num_speakers:
            self.intra_session_concat_prob = intra_session_concat_prob
        else:
            raise ValueError(f"intra_session_concat_prob must be either a float or a list of floats, but got {intra_session_concat_prob}")
        if data_type not in ["msasr", "diar"]:
            raise ValueError("data_type must be either 'msasr' or 'diar', but got {data_type}")
        self.data_type = data_type
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_num_speakers = max_num_speakers
        self.speaker_count_distribution = speaker_count_distribution
        assert len(speaker_count_distribution) == max_num_speakers, f"Length of speaker_count_distribution {len(speaker_count_distribution)} must be equal to max_num_speakers {max_num_speakers}"

        if skip_long_segments:
            self.skip_duration = max_duration / 2
        else:
            self.skip_duration = max_duration

        self.valid_dataset_ids = valid_dataset_ids

    def fit(self, cuts) -> CutSet:
        """
        Read the manifest file and return a CutSet object. 
        Each line in the manifest file should be a JSON object representing a segment.
        """

        self.id2cut = {}
        self.sess2cut_ids = defaultdict(list)
        self.sess2spks = defaultdict(set)
        self.data2sess_ids = defaultdict(list)
        self.spk2cut_ids = defaultdict(list)
        self.data2num_spk2cut_ids = {}
        self.sess2num_spk2cut_ids = {}
        self.num_spk2cut_ids = {i+1:[] for i in range(self.max_num_speakers)}
        for i, cut in tqdm(enumerate(cuts), desc="Reading segments", ncols=100, total=len(cuts)):
            if cut.duration > self.skip_duration:
                continue
            if not hasattr(cut, 'dataset_id') or cut.dataset_id is None:
                continue
            if self.valid_dataset_ids and cut.dataset_id not in self.valid_dataset_ids:
                continue
            if cut.dataset_id not in self.data2num_spk2cut_ids:
                self.data2num_spk2cut_ids[cut.dataset_id] = defaultdict(list)
            if cut.recording_id not in self.sess2num_spk2cut_ids:
                self.sess2num_spk2cut_ids[cut.recording_id] = defaultdict(list)
            
            speakers = cut.global_speaker_ids
            if self.data_type == "msasr":
                speaker_tokens = set(re.findall(r'<\|spltoken\d+\|>', cut.text))
                if len(speakers) != len(speaker_tokens): 
                    # Lhotse automatically fixes the max duration of the cut, 
                    # resulting in the mismatch of the number of speakers 
                    # and speaker tokens for the last segment
                    # TODO: need to fix the issue in Lhotse that automatically fixes the max duration
                    continue
            for spk in speakers:
                self.spk2cut_ids[spk].append(cut.id)
            self.sess2spks[cut.recording_id] = self.sess2spks[cut.recording_id].union(speakers)
            
            self.id2cut[cut.id] = cut
            self.sess2cut_ids[cut.recording_id].append(cut.id)
            self.data2num_spk2cut_ids[cut.dataset_id][len(speakers)].append(cut.id)
            self.sess2num_spk2cut_ids[cut.recording_id][len(speakers)].append(cut.id)
            self.num_spk2cut_ids[len(speakers)].append(cut.id)
            if cut.recording_id not in self.data2sess_ids[cut.dataset_id]:
                self.data2sess_ids[cut.dataset_id].append(cut.recording_id)
                
        self.cut_ids = list(self.id2cut.keys())
        self.num_spk2sess_ids = groupby(lambda x: len(self.sess2spks[x]), self.sess2spks.keys())
        
        self.data2global_speaker = {
            dataset_id: True for dataset_id in self.data2sess_ids.keys()
        }        
            
    def _create_mixture(self, n_speakers: int, is_intra_session_concat=False) -> MixedCut:

        db_norm = norm.rvs(-32.05957708631966, 5.66648411405886) # mean and std from Fisher data
        
        if is_intra_session_concat:
            # intra-dataset and intra-session concatenation
            tracks, num_speakers = self.get_intra_session_tracks(n_speakers, db_norm=db_norm)

        else: 
            # intra-dataset but inter-session concatenation
            tracks, num_speakers = self.get_inter_session_tracks(n_speakers, db_norm=db_norm)

        cut = MixedCut(id='concat_' + '_'.join([track.cut.id for track in tracks]), tracks=tracks)
        if self.data_type == "msasr":
            cut = self.reorder_spk_mapping(cut)

        assert self.min_duration <= cut.duration <= self.max_duration, f"Total duration {cut.duration} is not within the range of min {self.min_duration} and max {self.max_duration}"
        assert n_speakers == num_speakers, f"Total number of speakers {cut.num_speakers} is not equal to the number of speakers {n_speakers}"

        return cut
    
    def get_intra_session_tracks(self, n_speakers: int=4, db_norm: float=-25) -> List[MixTrack]:
        """
        Get the tracks for the MixedCut object.
        """
        session_id = random.choice(self.num_spk2sess_ids[n_speakers])
        
        total_duration = 0.0
        total_spk_set = set()
        tracks = []
        while True:
            cut = self.id2cut[random.choice(self.sess2cut_ids[session_id])]
            tracks.append(MixTrack(cut=deepcopy(cut.normalize_loudness(target=db_norm, mix_first=False)), type=type(cut), offset=total_duration))
            total_spk_set = total_spk_set.union(cut.global_speaker_ids)
            total_duration += cut.duration

            # break condition
            if total_duration >= self.min_duration:
                if total_duration > self.max_duration: # exceed the maximum duration, starting over
                    total_duration = 0.0
                    total_spk_set = set()
                    tracks = []
                    session_id = random.choice(self.num_spk2sess_ids[n_speakers])
                if len(total_spk_set) == n_speakers: # meet the number of speakers and duration, break
                    break
                else:
                    total_duration = 0.0
                    total_spk_set = set()
                    tracks = []
                    session_id = random.choice(self.num_spk2sess_ids[n_speakers])
            
        return tracks, len(total_spk_set)

    def get_inter_session_tracks(self, n_speakers: int=4, db_norm: float=-25) -> List[MixTrack]:
        """
        Get the tracks for the MixedCut object.
        """
        sample_cut = self.id2cut[random.choice(self.cut_ids)]
        dataset_id = sample_cut.dataset_id
        n_spk_list = [n_spk for n_spk, cut_ids in self.data2num_spk2cut_ids[dataset_id].items() if len(cut_ids) > 0]
        sum_spk_list = set([i + j for i in n_spk_list for j in n_spk_list])

        if min(sum_spk_list) > n_speakers:
            raise ValueError(f"Cannot generate {n_speakers}-speaker inter session samples by concatenating two samples since the dataset {dataset_id} only have {','.join([str(i) for i in n_spk_list])} speakers.")

        n_spk_left = n_speakers
        total_duration = 0.0
        total_spk_set = set()
        tracks = []
        num_spk2cut_ids = self.data2num_spk2cut_ids[dataset_id]
        while True:
            #if n_spk_left == n_speakers: # for more speakers cases
            #    n_spk = random.choice([n_spk for n_spk in n_spk_list if n_spk < n_spk_left])
            if n_spk_left >= 2:
                n_spk = 2
            else:
                # n_spk = random.choice([n_spk for n_spk in n_spk_list if n_spk <= n_spk_left])
                n_spk = 1

            while True:
                cut = self.id2cut[random.choice(num_spk2cut_ids[n_spk])]
                spks = set(cut.global_speaker_ids)
                if not spks.intersection(total_spk_set):
                    break

            tracks.append(MixTrack(cut=deepcopy(cut.normalize_loudness(target=db_norm, mix_first=False)), type=type(cut), offset=total_duration))
            total_duration += cut.duration
            n_spk_left -= n_spk
            total_spk_set = total_spk_set.union(spks)

            # break condition
            
            if total_duration >= self.min_duration:
                if total_duration > self.max_duration or len(total_spk_set) < n_speakers: # exceed the maximum duration, starting over
                    total_duration = 0.0
                    n_spk_left = n_speakers
                    total_spk_set = set()
                    tracks = []
                if len(total_spk_set) == n_speakers: # meet the number of speakers and duration, break
                    break
            else:
                if len(total_spk_set) == n_speakers: # meet the number of speakers, but not the duration, starting over --- TODO: will try to find the segments that only contains those speakers
                    total_duration = 0.0
                    n_spk_left = n_speakers
                    total_spk_set = set()
                    tracks = []
                    
        return tracks, len(total_spk_set)
    
    def reorder_spk_mapping(self, cut: MixedCut, pattern=r'<\|spltoken\d+\|>') -> str:
        """
        Concatenate the texts of the input cuts.
        
        """
        global_spk_mapping = {}
        str_pattern = pattern.replace("\\", '')
        left_str, right_str = str_pattern.split('d+')
        for i, track in enumerate(cut.tracks):
            local_inverse_spk_mapping = {}
            local_spk_mapping = {}
            for speaker in track.cut.global_speaker_ids:
                if speaker not in global_spk_mapping:
                    global_spk_mapping[speaker] = len(global_spk_mapping)
                if speaker not in local_spk_mapping:
                    local_spk_mapping[speaker] = len(local_spk_mapping)
                    local_inverse_spk_mapping[len(local_inverse_spk_mapping)] = speaker
                    
            if i != 0:
                text = ''
                for word in track.cut.text.split(): 
                    if len(re.findall(pattern, word)) > 0:
                        local_spk_idx = int(word.replace(left_str,'').replace(right_str, ''))
                        spk = local_inverse_spk_mapping[local_spk_idx]
                        global_spk_idx = global_spk_mapping[spk]
                        text += f'{left_str}{global_spk_idx}{right_str}'
                    else:
                        text += ' ' + word
                track.cut.supervisions[0].text = text
                cut.supervisions[i].text = text
            else:
                cut.supervisions[0].text = track.cut.text
                # TODO: need to check the last speaker of last track and the first speaker of the current track 
                # if they are the same, we need to remove the the speaker token from the current track for segment-level
                # Do not need to remove the speaker token for word-level
            
        return cut
    
    def apply_speaker_distribution(self, num_meetings: int, speaker_count_distribution) -> Dict[int, int]:
        """
        Balance the speaker distribution for the simulated meetings.
        Args:
            num_meetings: The total number of simulated meetings.
            speaker_count_distribution: The speaker count distribution for the simulated meetings.
        For each number of speakers, calculate the number of meetings needed to balance the distribution.
        """

        total_spk = sum(speaker_count_distribution)
        num_speakers2num_meetings = {}
        for i_spk in range(self.max_num_speakers):
            num_speakers2num_meetings[i_spk+1] = round(num_meetings * speaker_count_distribution[i_spk] / total_spk)

        return num_speakers2num_meetings
        
    
    @dill_enabled(True)
    def simulate(self, 
        cuts: CutSet,
        num_meetings: int = 10000,
        seed: int = 0,
        num_jobs: int = 1,
    ) -> CutSet:
        random.seed(seed)

        self.fit(cuts)
        

        num_speakers2num_meetings = self.apply_speaker_distribution(num_meetings, self.speaker_count_distribution)
        logging.warn(f"Will be generating {(','.join([str(i) for i in num_speakers2num_meetings.values()]))} samples for {(','.join([str(i) for i in num_speakers2num_meetings.keys()]))} speakers given speaker count distribution of {str(self.speaker_count_distribution)}.")
        num_speakers2num_meetings[1] = 0 # skip 1-speaker samples
        logging.warn(f'But 1-speaker samples will be skipped. Will be generating {sum(num_speakers2num_meetings.values()) - num_speakers2num_meetings[1]} samples in total.')

        # Step 0: Calculate the number of intra-session and inter-session concatentation samples
        n_spks = [k for k, v in self.num_spk2cut_ids.items() if len(v) > 0]
        valid_sim_n_spks = set([i+j for i in n_spks for j in n_spks]) # valid number of speakers for inter-session samples
        n_spk2n_intra_mt, n_spk2n_inter_mt = {i+1:0 for i in range(self.max_num_speakers)}, {i+1:0 for i in range(self.max_num_speakers)}
        for n_spk, n_mt in num_speakers2num_meetings.items():
            logging.warn(f"=="*16 + f"{n_spk}-speaker" + "=="*16)
            if n_mt <= 0:
                logging.warning(f"No concatentation samples for {n_spk} speakers. Will skip simulation for {n_spk} speakers.")
                continue
            n_intra_mt = int(n_mt * self.intra_session_concat_prob[n_spk-1])
            n_inter_mt = n_mt - n_intra_mt
            if n_spk in self.num_spk2sess_ids:
                logging.warn(f"Will be genrating {n_intra_mt} {n_spk}-speaker intra-session concatentation samples.")
                n_spk2n_intra_mt[n_spk] = n_intra_mt
            else:
                logging.warning(f"Cannot generate {n_intra_mt} {n_spk}-speaker intra-session samples by concatenating two samples from the same session since we only have samples for {','.join([str(i) for i in n_spks])} speakers.")
                n_spk2n_intra_mt[n_spk] = 0
                n_inter_mt = n_mt
            if n_spk in valid_sim_n_spks:
                logging.warn(f"Will be genrating {n_inter_mt} {n_spk}-speaker inter-session concatentation samples.")
                n_spk2n_inter_mt[n_spk] = n_inter_mt
            else:
                logging.warning(f"Cannot generate {n_inter_mt} {n_spk}-speaker inter-session samples by concatenating two samples from different sessions since we only have samples for {','.join([str(i) for i in n_spks])} speakers.")
                if n_spk2n_intra_mt[n_spk] != 0:
                    n_spk2n_intra_mt[n_spk] = n_mt
                    logging.warn(f"Will be genrating {n_spk2n_intra_mt[n_spk]} {n_spk}-speaker intra-session concatentation samples instead.")
                else:
                    logging.warning(f"No samples for {n_spk} speakers. Will skip simulation for {n_spk} speakers.")
        logging.warn(f"""Will be generating {','.join([str(i) for i in n_spk2n_intra_mt.values()])} intra-session concatentation samples and {','.join([str(i) for i in n_spk2n_inter_mt.values()])} inter-session concatentation samples for {','.join([str(i+1) for i in range(self.max_num_speakers)])} speakers.""")
        # Step 1: intra-session
        num_intra_meetings = 0
        intra_mixtures = []
        logging.info(f"Simulating intra-session concatentation samples.")
        for n_spk, n_mt in n_spk2n_intra_mt.items():
            if n_mt <= 0:
                continue

            for i in tqdm(range(n_mt), desc=f"Simulating {n_spk}-speaker intra-session mixtures", ncols=128):
                intra_mixtures.append(self._create_mixture(n_speakers=n_spk, is_intra_session_concat=True))
            num_intra_meetings += n_mt
        logging.info(f"Finished simulating intra-session concatentation samples. Total number of intra-session concatentation samples: {num_intra_meetings}")
    
        # Steo 2: inter-session
        logging.info(f"Simulating inter-session concatentation samples.")
        
        num_inter_meetings = 0
        inter_mixtures = []
        for n_spk, n_mt in n_spk2n_inter_mt.items():
            if n_mt <= 0:
                continue
            
            for i in tqdm(range(n_mt), desc=f"Simulating {n_spk}-speaker inter-session mixtures", ncols=128):
                inter_mixtures.append(self._create_mixture(n_speakers=n_spk, is_intra_session_concat=False))
            num_inter_meetings += n_mt
        logging.info(f"Finished simulating inter-session concatentation samples. Total number of inter-session concatentation samples: {num_inter_meetings}")

        if num_inter_meetings + num_intra_meetings == 0:
            logging.warning(f"No samples are generated. Probably the duration of the segments is not within the range of min {self.min_duration//2} and max {self.max_duration//2}, or the speaker count distribution is not correctly set.")


        # Multi-processing gets slower, TODO
        # else:
        #     futures = []
        #     for n_spk, n_mt in num_speakers2num_meetings.items():
        #         tp = concurrent.futures.ProcessPoolExecutor(max_workers=num_jobs)
        #         futures.extend([tp.submit(self._create_mixture, n_spk) for _ in range(n_mt)])
        #     pbar = tqdm(total=num_meetings, desc=f"Simulating mixtures", unit="line", ncols=128) 
        #     count = 0
        #     for f in concurrent.futures.as_completed(futures):
        #         count += 1
        #         pbar.update()
        #         mixtures.append(f.result())
        #     tp.shutdown()
        #     pbar.close()

        return CutSet.from_cuts(intra_mixtures + inter_mixtures)
    

class MixMeetingSimulator():
    """
    This simulator Mix the segments from different/same sessions to create a
    multi-speaker meeting. 
    """

    def __init__(
        self,
        intra_session_mix_prob: float|List[float] = [0, 0, 0, 0],
        data_type: str = "msasr",
        min_duration: float = 80.0,
        max_duration: float = 100.0,
        max_num_speakers: int = 4,
        speaker_count_distribution: List[float] = [0, 0, 0.1, 4],
        valid_dataset_ids: List[str] = [],
    ):
        """
        :param intra_session_mix_prob: the probability of concatenating segments from the same
            session. [Default: 1]
        :param data_type: the type of data to simulate. Either 'msasr' or 'diar'. If 'msasr',
            the transcripts are included in the simulation,and the boundary segments are 
            not included. [Default: 'msasr']
        :param max_duration: the maximum duration of the simulated meeting. [Default: 40.0]
        """
        super().__init__()
        if isinstance(intra_session_mix_prob, float):
            self.intra_session_mix_prob = [intra_session_mix_prob] * (max_num_speakers)
        elif len(intra_session_mix_prob) == max_num_speakers:
            self.intra_session_mix_prob = intra_session_mix_prob
        else:
            raise ValueError(f"intra_session_mix_prob must be either a float or a list of floats, but got {intra_session_mix_prob}")
        if data_type not in ["msasr", "diar"]:
            raise ValueError("data_type must be either 'msasr' or 'diar', but got {data_type}")
        self.data_type = data_type
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_num_speakers = max_num_speakers
        self.speaker_count_distribution = speaker_count_distribution
        self.valid_dataset_ids = valid_dataset_ids
        assert len(speaker_count_distribution) == max_num_speakers, f"Length of speaker_count_distribution {len(speaker_count_distribution)} must be equal to max_num_speakers {max_num_speakers}"

    def fit(self, cuts) -> CutSet:
        """
        Read the manifest file and return a CutSet object. 
        Each line in the manifest file should be a JSON object representing a segment.
        """

        self.id2cut = {}
        self.sess2cut_ids = defaultdict(list)
        self.sess2spks = defaultdict(set)
        self.data2sess_ids = defaultdict(list)
        self.spk2cut_ids = defaultdict(list)
        self.data2num_spk2cut_ids = {}
        self.sess2num_spk2cut_ids = {}
        self.num_spk2cut_ids = {i+1:[] for i in range(self.max_num_speakers)}
        for i, cut in tqdm(enumerate(cuts), desc="Reading segments", ncols=100, total=len(cuts)):
            if not self.min_duration <= cut.duration <= self.max_duration:
                continue
            if not hasattr(cut, 'dataset_id') or cut.dataset_id is None:
                continue
            if self.valid_dataset_ids and cut.dataset_id not in self.valid_dataset_ids:
                continue
            if cut.dataset_id not in self.data2num_spk2cut_ids:
                self.data2num_spk2cut_ids[cut.dataset_id] = defaultdict(list)
            if cut.recording_id not in self.sess2num_spk2cut_ids:
                self.sess2num_spk2cut_ids[cut.recording_id] = defaultdict(list)
            
            speakers = cut.global_speaker_ids
            if self.data_type == "msasr":
                speaker_tokens = set(re.findall(r'<\|spltoken\d+\|>', cut.text))
                if len(speakers) != len(speaker_tokens): 
                    # Lhotse automatically fixes the max duration of the cut, 
                    # resulting in the mismatch of the number of speakers 
                    # and speaker tokens for the last segment
                    # TODO: need to fix the issue in Lhotse that automatically fixes the max duration
                    continue
            for spk in speakers:
                self.spk2cut_ids[spk].append(cut.id)
            self.sess2spks[cut.recording_id] = self.sess2spks[cut.recording_id].union(speakers)
            
            self.id2cut[cut.id] = cut
            self.sess2cut_ids[cut.recording_id].append(cut.id)
            self.data2num_spk2cut_ids[cut.dataset_id][len(speakers)].append(cut.id)
            self.sess2num_spk2cut_ids[cut.recording_id][len(speakers)].append(cut.id)
            self.num_spk2cut_ids[len(speakers)].append(cut.id)
            if cut.recording_id not in self.data2sess_ids[cut.dataset_id]:
                self.data2sess_ids[cut.dataset_id].append(cut.recording_id)
                
        self.cut_ids = list(self.id2cut.keys())
        self.num_spk2sess_ids = groupby(lambda x: len(self.sess2spks[x]), self.sess2spks.keys())
        
        self.data2global_speaker = {
            dataset_id: True for dataset_id in self.data2sess_ids.keys()
        }        
            
    def _create_mixture(self, n_speakers: int, is_intra_session_concat=False) -> MixedCut:

        db_norm = norm.rvs(-32.05957708631966, 5.66648411405886) # mean and std from Fisher data
        
        if is_intra_session_concat:
            # intra-dataset and intra-session concatenation
            tracks, num_speakers = self.get_intra_session_tracks(n_speakers, db_norm=db_norm)

        else: 
            # intra-dataset but inter-session concatenation
            tracks, num_speakers = self.get_inter_session_tracks(n_speakers, db_norm=db_norm)

        cut = MixedCut(id='mix_' + '_'.join([track.cut.id for track in tracks]), tracks=tracks)
        if self.data_type == "msasr":
            cut = self.reorder_spk_mapping(cut)

        assert self.min_duration <= cut.duration <= self.max_duration, f"Total duration {cut.duration} is not within the range of min {self.min_duration} and max {self.max_duration}"
        assert n_speakers == num_speakers, f"Total number of speakers {cut.num_speakers} is not equal to the number of speakers {n_speakers}"

        return cut
    
    def get_intra_session_tracks(self, n_speakers: int=4, db_norm: float=-25) -> List[MixTrack]:
        """
        Get the tracks for the MixedCut object.
        """
        session_id = random.choice(self.num_spk2sess_ids[n_speakers])
        
        total_spk_set = set()
        tracks = []
        while True:
            cut = self.id2cut[random.choice(self.sess2cut_ids[session_id])]
            tracks.append(MixTrack(cut=deepcopy(cut.normalize_loudness(target=db_norm, mix_first=False)), type=type(cut), offset=0))
            total_spk_set = total_spk_set.union(cut.global_speaker_ids)
            total_duration = max(total_duration, cut.duration)

            # break condition
            if total_duration >= self.min_duration:
                if total_duration > self.max_duration: # exceed the maximum duration, starting over
                    total_duration = 0.0
                    total_spk_set = set()
                    tracks = []
                    session_id = random.choice(self.num_spk2sess_ids[n_speakers])
                if len(total_spk_set) == n_speakers: # meet the number of speakers and duration, break
                    break
                else:
                    total_duration = 0.0
                    total_spk_set = set()
                    tracks = []
                    session_id = random.choice(self.num_spk2sess_ids[n_speakers])
            
        return tracks, len(total_spk_set)

    def get_inter_session_tracks(self, n_speakers: int=4, db_norm: float=-25) -> List[MixTrack]:
        """
        Get the tracks for the MixedCut object.
        """
        sample_cut = self.id2cut[random.choice(self.cut_ids)]
        dataset_id = sample_cut.dataset_id
        n_spk_list = [n_spk for n_spk, cut_ids in self.data2num_spk2cut_ids[dataset_id].items() if len(cut_ids) > 0]
        sum_spk_list = set([i + j for i in n_spk_list for j in n_spk_list])

        if min(sum_spk_list) > n_speakers:
            raise ValueError(f"Cannot generate {n_speakers}-speaker inter session samples by concatenating two samples since the dataset {dataset_id} only have {','.join([str(i) for i in n_spk_list])} speakers.")

        n_spk_left = n_speakers
        total_duration = 0.0
        total_spk_set = set()
        tracks = []
        num_spk2cut_ids = self.data2num_spk2cut_ids[dataset_id]
        while True:
            if n_spk_left >= 2:
                n_spk = 2
            else:
                # n_spk = random.choice([n_spk for n_spk in n_spk_list if n_spk <= n_spk_left])
                n_spk = 1

            while True:
                cut = self.id2cut[random.choice(num_spk2cut_ids[n_spk])]
                spks = set(cut.global_speaker_ids)
                if not spks.intersection(total_spk_set):
                    break

            tracks.append(MixTrack(cut=deepcopy(cut.normalize_loudness(target=db_norm, mix_first=False)), type=type(cut), offset=0))
            total_duration = max(total_duration, cut.duration)
            n_spk_left -= n_spk
            total_spk_set = total_spk_set.union(spks)

            # break condition
            
            if total_duration >= self.min_duration:
                if total_duration > self.max_duration or len(tracks) > 2: # exceed the maximum duration, starting over
                    total_duration = 0.0
                    n_spk_left = n_speakers
                    total_spk_set = set()
                    tracks = []
                if len(total_spk_set) == n_speakers: # meet the number of speakers and duration, break
                    break
            else:
                if len(total_spk_set) == n_speakers: # meet the number of speakers, but not the duration, starting over --- TODO: will try to find the segments that only contains those speakers
                    total_duration = 0.0
                    n_spk_left = n_speakers
                    total_spk_set = set()
                    tracks = []
                    
        return tracks, len(total_spk_set)
    
    def reorder_spk_mapping(self, cut: MixedCut, pattern=r'<\|spltoken\d+\|>') -> str:
        """
        Concatenate the texts of the input cuts.
        
        """
        global_spk_mapping = {}
        str_pattern = pattern.replace("\\", '')
        left_str, right_str = str_pattern.split('d+')
        for i, track in enumerate(cut.tracks):
            local_inverse_spk_mapping = {}
            local_spk_mapping = {}
            for speaker in track.cut.global_speaker_ids:
                if speaker not in global_spk_mapping:
                    global_spk_mapping[speaker] = len(global_spk_mapping)
                if speaker not in local_spk_mapping:
                    local_spk_mapping[speaker] = len(local_spk_mapping)
                    local_inverse_spk_mapping[len(local_inverse_spk_mapping)] = speaker
                    
            if i != 0:
                text = ''
                for word in track.cut.text.split(): 
                    if len(re.findall(pattern, word)) > 0:
                        local_spk_idx = int(word.replace(left_str,'').replace(right_str, ''))
                        spk = local_inverse_spk_mapping[local_spk_idx]
                        global_spk_idx = global_spk_mapping[spk]
                        text += f'{left_str}{global_spk_idx}{right_str}'
                    else:
                        text += ' ' + word
                track.cut.supervisions[0].text = text
                cut.supervisions[i].text = text
            else:
                cut.supervisions[0].text = track.cut.text
                # TODO: need to check the last speaker of last track and the first speaker of the current track 
                # if they are the same, we need to remove the the speaker token from the current track for segment-level
                # Do not need to remove the speaker token for word-level
            
        return cut
    
    def apply_speaker_distribution(self, num_meetings: int, speaker_count_distribution) -> Dict[int, int]:
        """
        Balance the speaker distribution for the simulated meetings.
        Args:
            num_meetings: The total number of simulated meetings.
            speaker_count_distribution: The speaker count distribution for the simulated meetings.
        For each number of speakers, calculate the number of meetings needed to balance the distribution.
        """

        total_spk = sum(speaker_count_distribution)
        num_speakers2num_meetings = {}
        for i_spk in range(self.max_num_speakers):
            num_speakers2num_meetings[i_spk+1] = round(num_meetings * speaker_count_distribution[i_spk] / total_spk)

        return num_speakers2num_meetings
        
    
    @dill_enabled(True)
    def simulate(self, 
        cuts: CutSet,
        num_meetings: int = 10000,
        seed: int = 0,
        num_jobs: int = 1,
    ) -> CutSet:
        random.seed(seed)

        self.fit(cuts)

        num_speakers2num_meetings = self.apply_speaker_distribution(num_meetings, self.speaker_count_distribution)
        logging.warn(f"Will be generating {(','.join([str(i) for i in num_speakers2num_meetings.values()]))} samples for {(','.join([str(i) for i in num_speakers2num_meetings.keys()]))} speakers given speaker count distribution of {str(self.speaker_count_distribution)}.")
        num_speakers2num_meetings[1] = 0 # skip 1-speaker samples
        logging.warn(f'But 1-speaker samples will be skipped. Will be generating {sum(num_speakers2num_meetings.values()) - num_speakers2num_meetings[1]} samples in total.')

        # Step 0: Calculate the number of intra-session and inter-session concatentation samples
        n_spks = [k for k, v in self.num_spk2cut_ids.items() if len(v) > 0]
        valid_sim_n_spks = set([i+j for i in n_spks for j in n_spks]) # valid number of speakers for inter-session samples
        n_spk2n_intra_mt, n_spk2n_inter_mt = {i+1:0 for i in range(self.max_num_speakers)}, {i+1:0 for i in range(self.max_num_speakers)}
        for n_spk, n_mt in num_speakers2num_meetings.items():
            logging.warn(f"=="*16 + f"{n_spk}-speaker" + "=="*16)
            if n_mt <= 0:
                logging.warning(f"No intra-session concatentation samples for {n_spk} speakers. Will skip simulation for {n_spk} speakers.")
                continue
            n_intra_mt = int(n_mt * self.intra_session_mix_prob[n_spk-1])
            n_inter_mt = n_mt - n_intra_mt
            if n_spk in self.num_spk2sess_ids:
                logging.warn(f"Will be genrating {n_intra_mt} {n_spk}-speaker intra-session concatentation samples.")
                n_spk2n_intra_mt[n_spk] = n_intra_mt
            else:
                logging.warning(f"Cannot generate {n_intra_mt} {n_spk}-speaker intra-session samples by concatenating two samples from the same session since we only have samples for {','.join([str(i) for i in n_spks])} speakers.")
                n_spk2n_intra_mt[n_spk] = 0
                n_inter_mt = n_mt
            if n_spk in valid_sim_n_spks:
                logging.warn(f"Will be genrating {n_inter_mt} {n_spk}-speaker inter-session concatentation samples.")
                n_spk2n_inter_mt[n_spk] = n_inter_mt
            else:
                logging.warning(f"Cannot generate {n_inter_mt} {n_spk}-speaker inter-session samples by concatenating two samples from different sessions since we only have samples for {','.join([str(i) for i in n_spks])} speakers.")
                if n_spk2n_intra_mt[n_spk] != 0:
                    n_spk2n_intra_mt[n_spk] = n_mt
                    logging.warn(f"Will be genrating {n_spk2n_intra_mt[n_spk]} {n_spk}-speaker intra-session concatentation samples instead.")
                else:
                    logging.warning(f"No samples for {n_spk} speakers. Will skip simulation for {n_spk} speakers.")
        logging.warn(f"""Will be generating {','.join([str(i) for i in n_spk2n_intra_mt.values()])} intra-session concatentation samples and {','.join([str(i) for i in n_spk2n_inter_mt.values()])} inter-session concatentation samples for {','.join([str(i+1) for i in range(self.max_num_speakers)])} speakers.""")
        # Step 1: intra-session
        num_intra_meetings = 0
        intra_mixtures = []
        logging.info(f"Simulating intra-session concatentation samples.")
        for n_spk, n_mt in n_spk2n_intra_mt.items():
            if n_mt <= 0:
                continue

            for i in tqdm(range(n_mt), desc=f"Simulating {n_spk}-speaker intra-session mixtures", ncols=128):
                intra_mixtures.append(self._create_mixture(n_speakers=n_spk, is_intra_session_concat=True))
            num_intra_meetings += n_mt
        logging.info(f"Finished simulating intra-session concatentation samples. Total number of intra-session concatentation samples: {num_intra_meetings}")
    
        # Steo 2: inter-session
        logging.info(f"Simulating inter-session concatentation samples.")
        
        num_inter_meetings = 0
        inter_mixtures = []
        for n_spk, n_mt in n_spk2n_inter_mt.items():
            if n_mt <= 0:
                continue
            
            for i in tqdm(range(n_mt), desc=f"Simulating {n_spk}-speaker inter-session mixtures", ncols=128):
                inter_mixtures.append(self._create_mixture(n_speakers=n_spk, is_intra_session_concat=False))
            num_inter_meetings += n_mt
        logging.info(f"Finished simulating inter-session concatentation samples. Total number of inter-session concatentation samples: {num_inter_meetings}")

        if num_inter_meetings + num_intra_meetings == 0:
            logging.warning(f"No samples are generated. Probably the duration of the segments is not within the range of min {self.min_duration} and max {self.max_duration}, or the speaker count distribution is not correctly set.")

        return CutSet.from_cuts(intra_mixtures + inter_mixtures)

class LibriSpeechMixSimulator():

    def __init__(
        self,
        min_duration: float = 80.0,
        max_duration: float = 100.0,
        n_mix_speakers: List[int] = [1, 2, 3],
        speaker_count_distribution: List[float] = [1, 1, 1],
    ):
        """
        :param min_duration: the minimum duration of the simulated meeting. [Default: 80.0]
        :param max_duration: the maximum duration of the simulated meeting. [Default: 100.0]
        """
        super().__init__()
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.n_mix_speakers = n_mix_speakers
        self.speaker_count_distribution = speaker_count_distribution
        assert len(speaker_count_distribution) == len(n_mix_speakers), f"Length of speaker_count_distribution {len(speaker_count_distribution)} must be equal to max_num_speakers {len(n_mix_speakers)}"

    def fit(self, cuts) -> CutSet:
        pass

    def simulate(self, 
        cuts: CutSet,
        num_meetings: int = 10000,
        seed: int = 0,
        num_jobs: int = 1,
    ) -> CutSet:
        random.seed(seed)

        cut_set = []
        for n_speakers, n_mt in zip(self.n_mix_speakers, self.speaker_count_distribution):
            if n_mt <= 0:
                continue
            for i in tqdm(range(n_mt), desc=f"Simulating {n_speakers}-speaker mixtures", ncols=128):
                cut_set.append(self._create_mixture(n_speakers=n_speakers))
        return CutSet.from_cuts(cut_set)

class LibriSpeechMixGenerator():
    def __init__(self):
        pass

    def generate(self, cuts):
        cut_set = []
        for cut in tqdm(cuts):
            offsets = cut.delays
            durations = cut.durations
            wavs = cut.wavs
            texts = cut.texts
            speakers = cut.speakers

            tracks = []
            for i, (offset, duration, wav, text, speaker) in enumerate(zip(offsets, durations, wavs, texts, speakers)):
                wav_dur = soundfile.info(wav).duration
                wav_samples = soundfile.info(wav).frames
                custom = {
                    'speaker': speaker,
                    'text': text,
                }
                cut_1spk = MonoCut(
                    id=wav.split('/')[-1].replace('.wav', ''),
                    start=0,
                    duration=duration,
                    channel=0,
                    supervisions=[],
                    recording=Recording(
                        id=wav.split('/')[-1].replace('.wav', ''),
                        sources=[
                            AudioSource(
                                type='file',
                                channels=[0],
                                source=wav
                            )
                        ],
                        sampling_rate=16000, 
                        num_samples=wav_samples,
                        duration=wav_dur
                    ),
                    custom=custom
                )

                tracks.append(MixTrack(cut=cut_1spk, type=type(cut_1spk), offset=offset))
            sup = SupervisionSegment(
                id=cut.id,
                recording_id=cut.recording_id,
                start=0,
                duration=offset+wav_dur,
                text=cut.text,
            )
            tracks[0].cut.supervisions.append(sup)
            cut_multi_spk = MixedCut(id=cut.id, tracks=tracks)
            
            cut_set.append(cut_multi_spk)
        
        return CutSet.from_cuts(cut_set)