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
import io
import os
from collections import Counter, OrderedDict
from statistics import mode
from typing import Dict, List, Optional, Union

import torch
import webdataset as wd

from nemo.collections.asr.data.audio_to_text import expand_audio_filepaths
from nemo.collections.asr.parts.preprocessing.segment import available_formats as valid_sf_formats
from nemo.collections.asr.parts.utils.nmesc_clustering import get_argmin_mat
from nemo.collections.common.parts.preprocessing import collections
from nemo.collections.common.parts.preprocessing.collections import DiarizationSpeechLabel
from nemo.core.classes import Dataset, IterableDataset
from nemo.core.neural_types import (
    AudioSignal,
    EncodedRepresentation,
    LabelsType,
    LengthsType,
    NeuralType,
    RegressionValuesType,
)
from nemo.utils import logging

# List of valid file formats (prioritized by order of importance)
VALID_FILE_FORMATS = ';'.join(['wav', 'mp3', 'flac'] + [fmt.lower() for fmt in valid_sf_formats.keys()])


def repeat_signal(signal, sig_len, required_length):
    """repeat signal to make short signal to have required_length
    Args:
        signal (FloatTensor): input signal
        sig_len (LongTensor): length of input signal
        required_length(float) : length of generated signal
    Returns:
        signal (FloatTensor): generated signal of required_length by repeating itself.
    """
    repeat = int(required_length // sig_len)
    rem = int(required_length % sig_len)
    sub = signal[-rem:] if rem > 0 else torch.tensor([])
    rep_sig = torch.cat(repeat * [signal])
    signal = torch.cat((rep_sig, sub))
    return signal


def get_scale_mapping_list(uniq_timestamps):
    """
    Call get_argmin_mat function to find the index of the non-base-scale segment that is closest to the 
    given base-scale segment. For each scale and each segment, a base-scale segment is assigned.

    Args:
        uniq_timestamps: (Dict)
            The dictionary containing embeddings, timestamps and multiscale weights.
            If uniq_timestamps contains only one scale, single scale diarization is performed.

    Returns:
        scale_mapping_argmat (torch.tensor):

            The element at the m-th row and the n-th column of the scale mapping matrix indicates the (m+1)-th scale
            segment index which has the closest center distance with (n+1)-th segment in the base scale.

            Example:
                scale_mapping_argmat[2][101] = 85

            In the above example, it means that 86-th segment in the 3rd scale (python index is 2) is mapped with
            102-th segment in the base scale. Thus, the longer segments bound to have more repeating numbers since
            multiple base scale segments (since the base scale has the shortest length) fall into the range of the
            longer segments. At the same time, each row contains N numbers of indices where N is number of
            segments in the base-scale (i.e., the finest scale).
    """
    uniq_scale_dict = uniq_timestamps['scale_dict']
    scale_mapping_argmat = [[] for _ in range(len(uniq_scale_dict.keys()))]

    session_scale_mapping_dict = get_argmin_mat(uniq_scale_dict)
    for scale_idx in sorted(uniq_scale_dict.keys()):
        scale_mapping_argmat[scale_idx] = torch.tensor(session_scale_mapping_dict[scale_idx])
    scale_mapping_argmat = torch.stack(scale_mapping_argmat)
    return scale_mapping_argmat


def normalize(signal):
    """normalize signal
    Args:
        signal(FloatTensor): signal to be normalized.
    """
    signal_minusmean = signal - signal.mean()
    return signal_minusmean / signal_minusmean.abs().max()


def count_occurence(manifest_file_id):
    """Count number of wav files in Dict manifest_file_id. Use for _TarredAudioToLabelDataset.
    Args:
        manifest_file_id (Dict): Dict of files and their corresponding id. {'A-sub0' : 1, ..., 'S-sub10':100}
    Returns:
        count (Dict): Dict of wav files {'A' : 2, ..., 'S':10}
    """
    count = dict()
    for i in manifest_file_id:
        audio_filename = i.split("-sub")[0]
        count[audio_filename] = count.get(audio_filename, 0) + 1
    return count


def _speech_collate_fn(batch, pad_id):
    """collate batch of audio sig, audio len, tokens, tokens len
    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
               LongTensor):  A tuple of tuples of signal, signal lengths,
               encoded tokens, and encoded tokens length.  This collate func
               assumes the signals are 1d torch tensors (i.e. mono audio).
    """
    _, audio_lengths, _, tokens_lengths = zip(*batch)
    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()
    max_tokens_len = max(tokens_lengths).item()

    audio_signal, tokens = [], []
    for sig, sig_len, tokens_i, tokens_i_len in batch:
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)
        tokens_i_len = tokens_i_len.item()
        if tokens_i_len < max_tokens_len:
            pad = (0, max_tokens_len - tokens_i_len)
            tokens_i = torch.nn.functional.pad(tokens_i, pad, value=pad_id)
        tokens.append(tokens_i)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None
    tokens = torch.stack(tokens)
    tokens_lengths = torch.stack(tokens_lengths)

    return audio_signal, audio_lengths, tokens, tokens_lengths


def _fixed_seq_collate_fn(self, batch):
    """collate batch of audio sig, audio len, tokens, tokens len
        Args:
            batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
                LongTensor):  A tuple of tuples of signal, signal lengths,
                encoded tokens, and encoded tokens length.  This collate func
                assumes the signals are 1d torch tensors (i.e. mono audio).
        """
    _, audio_lengths, _, tokens_lengths = zip(*batch)

    has_audio = audio_lengths[0] is not None
    fixed_length = int(max(audio_lengths))

    audio_signal, tokens, new_audio_lengths = [], [], []
    for sig, sig_len, tokens_i, _ in batch:
        if has_audio:
            sig_len = sig_len.item()
            chunck_len = sig_len - fixed_length

            if chunck_len < 0:
                repeat = fixed_length // sig_len
                rem = fixed_length % sig_len
                sub = sig[-rem:] if rem > 0 else torch.tensor([])
                rep_sig = torch.cat(repeat * [sig])
                sig = torch.cat((rep_sig, sub))
            new_audio_lengths.append(torch.tensor(fixed_length))

            audio_signal.append(sig)

        tokens.append(tokens_i)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(new_audio_lengths)
    else:
        audio_signal, audio_lengths = None, None
    tokens = torch.stack(tokens)
    tokens_lengths = torch.stack(tokens_lengths)

    return audio_signal, audio_lengths, tokens, tokens_lengths


def _vad_frame_seq_collate_fn(self, batch):
    """collate batch of audio sig, audio len, tokens, tokens len
    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
            LongTensor):  A tuple of tuples of signal, signal lengths,
            encoded tokens, and encoded tokens length.  This collate func
            assumes the signals are 1d torch tensors (i.e. mono audio).
            batch size equals to 1.
    """
    slice_length = int(self.featurizer.sample_rate * self.window_length_in_sec)
    _, audio_lengths, _, tokens_lengths = zip(*batch)
    slice_length = min(slice_length, max(audio_lengths))
    shift = int(self.featurizer.sample_rate * self.shift_length_in_sec)
    has_audio = audio_lengths[0] is not None

    audio_signal, num_slices, tokens, audio_lengths = [], [], [], []

    append_len_start = slice_length // 2
    append_len_end = slice_length - slice_length // 2
    for sig, sig_len, tokens_i, _ in batch:
        if self.normalize_audio:
            sig = normalize(sig)
        start = torch.zeros(append_len_start)
        end = torch.zeros(append_len_end)
        sig = torch.cat((start, sig, end))
        sig_len += slice_length

        if has_audio:
            slices = torch.div(sig_len - slice_length, shift, rounding_mode='trunc')
            for slice_id in range(slices):
                start_idx = slice_id * shift
                end_idx = start_idx + slice_length
                signal = sig[start_idx:end_idx]
                audio_signal.append(signal)

            num_slices.append(slices)
            tokens.extend([tokens_i] * slices)
            audio_lengths.extend([slice_length] * slices)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.tensor(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None

    tokens = torch.stack(tokens)
    tokens_lengths = torch.tensor(num_slices)
    return audio_signal, audio_lengths, tokens, tokens_lengths


def _extract_seg_info_from_rttm(self, uniq_id, rttm_lines, target_spks=None):
    """
    Get RTTM lines containing speaker labels, start time and end time. target_spks contains two targeted
    speaker indices for creating the groundtruth label file. Only speakers in target_spks variable will be
    included in the output lists.

    Args:
        uniq_id (str):
            Unique file ID that refers to an input audio file and corresponding RTTM (Annotation) file.
        rttm_lines (list):
            List containing RTTM lines in str format.

    Returns:
        rttm_tup (tuple):
            Tuple containing lists of start time, end time and speaker labels.

    """
    stt_list, end_list, speaker_list, bi_ch_infer_spks = [], [], [], []
    if target_spks:
        label_scale_idx = max(self.emb_dict.keys())
        mapping_dict = self.emb_dict[label_scale_idx][uniq_id]['mapping']
        inv_map = {v: k for k, v in mapping_dict.items()}

        for spk_idx in target_spks[0]:
            spk_str = f'speaker_{spk_idx}'
            if spk_str in inv_map:
                bi_ch_infer_spks.append(inv_map[spk_str])
    for line in rttm_lines:
        rttm = line.strip().split()
        start, end, speaker = self.s2n(rttm[3]), self.s2n(rttm[4]) + self.s2n(rttm[3]), rttm[7]
        if target_spks is None or speaker in bi_ch_infer_spks:
            end_list.append(end)
            stt_list.append(start)
            speaker_list.append(speaker)
    rttm_tup = (stt_list, end_list, speaker_list)
    return rttm_tup


def _assign_frame_level_spk_vector(self, uniq_id, rttm_timestamps, target_spks, min_spks=2):
    """
    Create a multi-dimensional vector sequence containing speaker timestamp information in RTTM.
    The unit-length is the frame shift length of the acoustic feature. The feature-level annotations
    fr_level_target will later be converted to base-segment level diarization label.

    Args:
        uniq_id (str):
            Unique file ID that refers to an input audio file and corresponding RTTM (Annotation) file.
        rttm_timestamps (list):
            List containing start and end time for each speaker segment label.
            stt_list, end_list and speaker_list are contained.
        target_spks (tuple):
            Speaker indices that are generated from combinations. If there are only one or two speakers,
            only a single target_spks variable is generated.

    Returns:
        fr_level_target (torch.tensor):
            Tensor containing label for each feature level frame.
    """
    stt_list, end_list, speaker_list = rttm_timestamps
    if len(speaker_list) == 0:
        return None
    else:
        sorted_speakers = sorted(list(set(speaker_list)))
        total_fr_len = int(max(end_list) * self.decim)
        spk_num = max(len(sorted_speakers), min_spks)
        if spk_num > self.max_spks:
            raise ValueError(
                f"Number of speaker {spk_num} should be less than or equal to self.max_num_speakers {self.max_num_speakers}"
            )
        speaker_mapping_dict = {rttm_key: x_int for x_int, rttm_key in enumerate(sorted_speakers)}
        fr_level_target = torch.zeros(total_fr_len, spk_num)

        # If RTTM is not provided, then there is no speaker mapping dict in target_spks.
        # Thus, return a zero-filled tensor as a placeholder.
        if target_spks and target_spks[1] is None:
            return fr_level_target
        for count, (stt, end, spk_rttm_key) in enumerate(zip(stt_list, end_list, speaker_list)):
            stt, end = round(stt, self.round_digits), round(end, self.round_digits)
            spk = speaker_mapping_dict[spk_rttm_key]
            stt_fr, end_fr = int(round(stt, 2) * self.fr_per_sec), int(round(end, self.round_digits) * self.fr_per_sec)
            if target_spks is None:
                fr_level_target[stt_fr:end_fr, spk] = 1
            else:
                if spk in target_spks[0]:
                    idx = target_spks[0].index(spk)
                    fr_level_target[stt_fr:end_fr, idx] = 1

        return fr_level_target


def _get_diar_target_labels(self, uniq_id, fr_level_target, ms_ts_dict):
    """
    Generate a hard-label (0 or 1) for each base-scale segment. soft_label_thres is a threshold for determining
    how much overlap we require for labeling a segment-level label. Note that label_vec varialbe is not a one-hot encoded vector.
    label_vec is a multidimensional hard-label that can contain annotations for indicating overlapp of speech.

    Args:
        uniq_id (str):
            Unique file ID that refers to an input audio file and corresponding RTTM (Annotation) file.
        fr_level_target (torch.tensor):
            Tensor containing label for each feature-level frame.
        ms_ts_dict (Dict):
            Dictionary containing timestamps and speaker embedding sequence for each scale.

    Returns:
        seg_target (torch.tensor):
            Tensor containing binary speaker labels for base-scale segments.
        base_clus_label (torch.tensor):
            Representative label for each segment in terms of each speaker's time in the segment.
    """
    seg_target_list, base_clus_label = [], []
    subseg_time_stamp_list = ms_ts_dict[uniq_id]["scale_dict"][self.scale_n - 1]["time_stamps"]
    for line in subseg_time_stamp_list:
        line_split = line.split()
        seg_stt, seg_end = float(line_split[0]), float(line_split[1])
        seg_stt_fr, seg_end_fr = int(seg_stt * self.fr_per_sec), int(seg_end * self.fr_per_sec)
        soft_label_vec = torch.sum(fr_level_target[seg_stt_fr:seg_end_fr, :], axis=0) / (seg_end_fr - seg_stt_fr)
        label_int = torch.argmax(soft_label_vec)
        label_vec = (soft_label_vec > self.soft_label_thres).float()
        seg_target_list.append(label_vec.detach())
        base_clus_label.append(label_int.detach())
    seg_target = torch.stack(seg_target_list)
    base_clus_label = torch.stack(base_clus_label)
    return seg_target, base_clus_label


class _AudioLabelDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio files,
    labels, and durations and offsets(in seconds). Each new line is a
    different sample. Example below:
    and their target labels. JSON files should be of the following format::
        {"audio_filepath": "/path/to/audio_wav_0.wav", "duration": time_in_sec_0, "label": \
target_label_0, "offset": offset_in_sec_0}
        ...
        {"audio_filepath": "/path/to/audio_wav_n.wav", "duration": time_in_sec_n, "label": \
target_label_n, "offset": offset_in_sec_n}
    Args:
        manifest_filepath (str): Dataset parameter. Path to JSON containing data.
        labels (list): Dataset parameter. List of target classes that can be output by the speaker recognition model.
        featurizer
        min_duration (float): Dataset parameter. All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        trim (bool): Whether to use trim silence from beginning and end of audio signal using librosa.effects.trim().
            Defaults to False.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
        """

        output_types = {
            'audio_signal': NeuralType(
                ('B', 'T'),
                AudioSignal(freq=self._sample_rate)
                if self is not None and hasattr(self, '_sample_rate')
                else AudioSignal(),
            ),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

        if self.is_regression_task:
            output_types.update(
                {
                    'targets': NeuralType(tuple('B'), RegressionValuesType()),
                    'targets_length': NeuralType(tuple('B'), LengthsType()),
                }
            )
        else:

            output_types.update(
                {'label': NeuralType(tuple('B'), LabelsType()), 'label_length': NeuralType(tuple('B'), LengthsType()),}
            )

        return output_types

    def __init__(
        self,
        *,
        manifest_filepath: str,
        labels: List[str],
        featurizer,
        min_duration: Optional[float] = 0.1,
        max_duration: Optional[float] = None,
        trim: bool = False,
        is_regression_task: bool = False,
    ):
        super().__init__()
        self.collection = collections.ASRSpeechLabel(
            manifests_files=manifest_filepath.split(','),
            min_duration=min_duration,
            max_duration=max_duration,
            is_regression_task=is_regression_task,
        )

        self.featurizer = featurizer
        self.trim = trim
        self.is_regression_task = is_regression_task

        if not is_regression_task:
            self.labels = labels if labels else self.collection.uniq_labels
            self.num_classes = len(self.labels) if self.labels is not None else 1
            self.label2id, self.id2label = {}, {}
            for label_id, label in enumerate(self.labels):
                self.label2id[label] = label_id
                self.id2label[label_id] = label

            for idx in range(len(self.labels[:5])):
                logging.debug(" label id {} and its mapped label {}".format(idx, self.id2label[idx]))

        else:
            self.labels = []
            self.num_classes = 1

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, index):
        sample = self.collection[index]

        offset = sample.offset

        if offset is None:
            offset = 0

        features = self.featurizer.process(sample.audio_file, offset=offset, duration=sample.duration, trim=self.trim)
        f, fl = features, torch.tensor(features.shape[0]).long()

        if not self.is_regression_task:
            t = torch.tensor(self.label2id[sample.label]).long()
        else:
            t = torch.tensor(sample.label).float()

        tl = torch.tensor(1).long()  # For compatibility with collate_fn used later

        return f, fl, t, tl


# Ported from https://github.com/NVIDIA/OpenSeq2Seq/blob/master/open_seq2seq/data/speech2text/speech_commands.py
class AudioToClassificationLabelDataset(_AudioLabelDataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, command class, and durations (in seconds). Each new line is a
    different sample. Example below:
    {"audio_filepath": "/path/to/audio_wav_0.wav", "duration": time_in_sec_0, "label": \
        target_label_0, "offset": offset_in_sec_0}
    ...
    {"audio_filepath": "/path/to/audio_wav_n.wav", "duration": time_in_sec_n, "label": \
        target_label_n, "offset": offset_in_sec_n}
    Args:
        manifest_filepath: Path to manifest json as described above. Can
            be comma-separated paths.
        labels (Optional[list]): String containing all the possible labels to map to
            if None then automatically picks from ASRSpeechLabel collection.
        featurizer: Initialized featurizer class that converts paths of
            audio to feature tensors
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include
            in dataset
        trim: Boolean flag whether to trim the audio
    """

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch, pad_id=0)


class AudioToSpeechLabelDataset(_AudioLabelDataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, command class, and durations (in seconds). Each new line is a
    different sample. Example below:
    {"audio_filepath": "/path/to/audio_wav_0.wav", "duration": time_in_sec_0, "label": \
        target_label_0, "offset": offset_in_sec_0}
    ...
    {"audio_filepath": "/path/to/audio_wav_n.wav", "duration": time_in_sec_n, "label": \
        target_label_n, "offset": offset_in_sec_n}
    Args:
        manifest_filepath (str): Path to manifest json as described above. Can
            be comma-separated paths.
        labels (Optional[list]): String containing all the possible labels to map to
            if None then automatically picks from ASRSpeechLabel collection.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        trim (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        window_length_in_sec (float): length of window/slice (in seconds)
            Use this for speaker recognition and VAD tasks.
        shift_length_in_sec (float): amount of shift of window for generating the frame for VAD task in a batch
            Use this for VAD task during inference.
        normalize_audio (bool): Whether to normalize audio signal.
            Defaults to False.
        is_regression_task (bool): Whether the dataset is for a regression task instead of classification
    """

    def __init__(
        self,
        *,
        manifest_filepath: str,
        labels: List[str],
        featurizer,
        min_duration: Optional[float] = 0.1,
        max_duration: Optional[float] = None,
        trim: bool = False,
        window_length_in_sec: Optional[float] = 8,
        shift_length_in_sec: Optional[float] = 1,
        normalize_audio: bool = False,
        is_regression_task: bool = False,
    ):
        self.window_length_in_sec = window_length_in_sec
        self.shift_length_in_sec = shift_length_in_sec
        self.normalize_audio = normalize_audio

        logging.debug("Window/slice length considered for collate func is {}".format(self.window_length_in_sec))
        logging.debug("Shift length considered for collate func is {}".format(self.shift_length_in_sec))

        super().__init__(
            manifest_filepath=manifest_filepath,
            labels=labels,
            featurizer=featurizer,
            min_duration=min_duration,
            max_duration=max_duration,
            trim=trim,
            is_regression_task=is_regression_task,
        )

    def fixed_seq_collate_fn(self, batch):
        return _fixed_seq_collate_fn(self, batch)

    def vad_frame_seq_collate_fn(self, batch):
        return _vad_frame_seq_collate_fn(self, batch)


class _TarredAudioLabelDataset(IterableDataset):
    """
    A similar Dataset to the AudioLabelDataSet, but which loads tarred audio files.

    Accepts a single comma-separated JSON manifest file (in the same style as for the AudioToSpeechLabelDataset),
    as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
    contain the information for one audio file, including at least the label and name of the audio
    file within the tarball.

    Valid formats for the audio_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].

    Note: For brace expansion in (1), there may be cases where `{x..y}` syntax cannot be used due to shell interference.
    This occurs most commonly inside SLURM scripts. Therefore we provide a few equivalent replacements.
    Supported opening braces - { <=> (, [, < and the special tag _OP_.
    Supported closing braces - } <=> ), ], > and the special tag _CL_.
    For SLURM based tasks, we suggest the use of the special tags for ease of use.

    See the documentation for more information about accepted data and input formats.

    If using multiple processes the number of shards should be divisible by the number of workers to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    In addition, if using mutiprocessing, each shard MUST HAVE THE SAME NUMBER OF ENTRIES after filtering
    is applied. We currently do not check for this, but your program may hang if the shards are uneven!

    Notice that a few arguments are different from the AudioLabelDataSet; for example, shuffle (bool) has been
    replaced by shuffle_n (int).

    Additionally, please note that the len() of this DataLayer is assumed to be the length of the manifest
    after filtering. An incorrect manifest length may lead to some DataLoader issues down the line.

    Args:
        audio_tar_filepaths: Either a list of audio tarball filepaths, or a
            string (can be brace-expandable).
        manifest_filepath (str): Path to the manifest.
        labels (list): Dataset parameter.
            List of target classes that can be output by the speaker recognition model.
        featurizer
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        trim(bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        window_length_in_sec (float): length of slice/window (in seconds) # Pass this only for speaker recognition and VAD task
        shift_length_in_sec (float): amount of shift of window for generating the frame for VAD task. in a batch # Pass this only for VAD task during inference.
        normalize_audio (bool): Whether to normalize audio signal. Defaults to False.
        shard_strategy (str): Tarred dataset shard distribution strategy chosen as a str value during ddp.
            -   `scatter`: The default shard strategy applied by WebDataset, where each node gets
                a unique set of shards, which are permanently pre-allocated and never changed at runtime.
            -   `replicate`: Optional shard strategy, where each node gets all of the set of shards
                available in the tarred dataset, which are permanently pre-allocated and never changed at runtime.
                The benefit of replication is that it allows each node to sample data points from the entire
                dataset independently of other nodes, and reduces dependence on value of `shuffle_n`.

                Note: Replicated strategy allows every node to sample the entire set of available tarfiles,
                and therefore more than one node may sample the same tarfile, and even sample the same
                data points! As such, there is no assured guarantee that all samples in the dataset will be
                sampled at least once during 1 epoch.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 0.
        is_regression_task (bool): Whether it is a regression task. Defualts to False.
    """

    def __init__(
        self,
        *,
        audio_tar_filepaths: Union[str, List[str]],
        manifest_filepath: Union[str, List[str]],
        labels: List[str],
        featurizer,
        shuffle_n: int = 0,
        min_duration: Optional[float] = 0.1,
        max_duration: Optional[float] = None,
        trim: bool = False,
        shard_strategy: str = "scatter",
        global_rank: int = 0,
        world_size: int = 0,
        is_regression_task: bool = False,
    ):
        self.collection = collections.ASRSpeechLabel(
            manifests_files=manifest_filepath,
            min_duration=min_duration,
            max_duration=max_duration,
            index_by_file_id=True,  # Must set this so the manifest lines can be indexed by file ID
        )

        self.file_occurence = count_occurence(self.collection.mapping)

        self.featurizer = featurizer
        self.trim = trim

        self.labels = labels if labels else self.collection.uniq_labels
        self.num_classes = len(self.labels)

        self.label2id, self.id2label = {}, {}
        for label_id, label in enumerate(self.labels):
            self.label2id[label] = label_id
            self.id2label[label_id] = label

        for idx in range(len(self.labels[:5])):
            logging.debug(" label id {} and its mapped label {}".format(idx, self.id2label[idx]))

        audio_tar_filepaths = expand_audio_filepaths(
            audio_tar_filepaths=audio_tar_filepaths,
            shard_strategy=shard_strategy,
            world_size=world_size,
            global_rank=global_rank,
        )
        # Put together WebDataset
        self._dataset = wd.WebDataset(urls=audio_tar_filepaths, nodesplitter=None)

        if shuffle_n > 0:
            self._dataset = self._dataset.shuffle(shuffle_n)
        else:
            logging.info("WebDataset will not shuffle files within the tar files.")

        self._dataset = (
            self._dataset.rename(audio=VALID_FILE_FORMATS, key='__key__')
            .to_tuple('audio', 'key')
            .pipe(self._filter)
            .map(f=self._build_sample)
        )

    def _filter(self, iterator):
        """This function is used to remove samples that have been filtered out by ASRSpeechLabel already.
        Otherwise, we would get a KeyError as _build_sample attempts to find the manifest entry for a sample
        that was filtered out (e.g. for duration).
        Note that if using multi-GPU training, filtering may lead to an imbalance in samples in each shard,
        which may make your code hang as one process will finish before the other.
        """

        class TarredAudioFilter:
            def __init__(self, collection, file_occurence):
                self.iterator = iterator
                self.collection = collection
                self.file_occurence = file_occurence
                self._iterable = self._internal_generator()

            def __iter__(self):
                self._iterable = self._internal_generator()
                return self

            def __next__(self):
                try:
                    values = next(self._iterable)
                except StopIteration:
                    # reset generator
                    self._iterable = self._internal_generator()
                    values = next(self._iterable)

                return values

            def _internal_generator(self):
                """
                WebDataset requires an Iterator, but we require an iterable that yields 1-or-more
                values per value inside self.iterator.

                Therefore wrap the iterator with a generator function that will yield 1-or-more
                values per sample in the iterator.
                """
                for _, tup in enumerate(self.iterator):
                    audio_bytes, audio_filename = tup

                    file_id, _ = os.path.splitext(os.path.basename(audio_filename))
                    if audio_filename in self.file_occurence:
                        for j in range(0, self.file_occurence[file_id]):
                            if j == 0:
                                audio_filename = file_id
                            else:
                                audio_filename = file_id + "-sub" + str(j)
                            yield audio_bytes, audio_filename

        return TarredAudioFilter(self.collection, self.file_occurence)

    def _build_sample(self, tup):
        """Builds the training sample by combining the data from the WebDataset with the manifest info.
        """
        audio_bytes, audio_filename = tup

        # Grab manifest entry from self.collection
        file_id, _ = os.path.splitext(os.path.basename(audio_filename))

        manifest_idx = self.collection.mapping[file_id]
        manifest_entry = self.collection[manifest_idx]

        offset = manifest_entry.offset
        if offset is None:
            offset = 0

        # Convert audio bytes to IO stream for processing (for SoundFile to read)
        audio_filestream = io.BytesIO(audio_bytes)
        features = self.featurizer.process(
            audio_filestream, offset=offset, duration=manifest_entry.duration, trim=self.trim,
        )

        audio_filestream.close()

        # Audio features
        f, fl = features, torch.tensor(features.shape[0]).long()

        t = self.label2id[manifest_entry.label]
        tl = 1  # For compatibility with collate_fn used later

        return f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

    def __iter__(self):
        return self._dataset.__iter__()

    def __len__(self):
        return len(self.collection)


class TarredAudioToClassificationLabelDataset(_TarredAudioLabelDataset):
    """
    A similar Dataset to the AudioToClassificationLabelDataset, but which loads tarred audio files.

    Accepts a single comma-separated JSON manifest file (in the same style as for the AudioToClassificationLabelDataset),
    as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
    contain the information for one audio file, including at least the transcript and name of the audio
    file within the tarball.

    Valid formats for the audio_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].

    See the WebDataset documentation for more information about accepted data and input formats.

    If using multiple processes the number of shards should be divisible by the number of workers to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    In addition, if using mutiprocessing, each shard MUST HAVE THE SAME NUMBER OF ENTRIES after filtering
    is applied. We currently do not check for this, but your program may hang if the shards are uneven!

    Notice that a few arguments are different from the AudioToBPEDataset; for example, shuffle (bool) has been
    replaced by shuffle_n (int).

    Additionally, please note that the len() of this DataLayer is assumed to be the length of the manifest
    after filtering. An incorrect manifest length may lead to some DataLoader issues down the line.

    Args:
        audio_tar_filepaths: Either a list of audio tarball filepaths, or a
            string (can be brace-expandable).
        manifest_filepath (str): Path to the manifest.
        labels (list): Dataset parameter.
            List of target classes that can be output by the speaker recognition model.
        featurizer
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        trim(bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        shard_strategy (str): Tarred dataset shard distribution strategy chosen as a str value during ddp.
            -   `scatter`: The default shard strategy applied by WebDataset, where each node gets
                a unique set of shards, which are permanently pre-allocated and never changed at runtime.
            -   `replicate`: Optional shard strategy, where each node gets all of the set of shards
                available in the tarred dataset, which are permanently pre-allocated and never changed at runtime.
                The benefit of replication is that it allows each node to sample data points from the entire
                dataset independently of other nodes, and reduces dependence on value of `shuffle_n`.

                Note: Replicated strategy allows every node to sample the entire set of available tarfiles,
                and therefore more than one node may sample the same tarfile, and even sample the same
                data points! As such, there is no assured guarantee that all samples in the dataset will be
                sampled at least once during 1 epoch.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 0.
        is_regression_task (bool): Whether it is a regression task. Defualts to False.
    """

    # self.labels = labels if labels else self.collection.uniq_labels
    # self.num_commands = len(self.labels)

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch, pad_id=0)


class TarredAudioToSpeechLabelDataset(_TarredAudioLabelDataset):
    """
    A similar Dataset to the AudioToSpeechLabelDataset, but which loads tarred audio files.

    Accepts a single comma-separated JSON manifest file (in the same style as for the AudioToSpeechLabelDataset),
    as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
    contain the information for one audio file, including at least the transcript and name of the audio
    file within the tarball.

    Valid formats for the audio_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].

    See the WebDataset documentation for more information about accepted data and input formats.

    If using multiple processes the number of shards should be divisible by the number of workers to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    In addition, if using mutiprocessing, each shard MUST HAVE THE SAME NUMBER OF ENTRIES after filtering
    is applied. We currently do not check for this, but your program may hang if the shards are uneven!

    Notice that a few arguments are different from the AudioToBPEDataset; for example, shuffle (bool) has been
    replaced by shuffle_n (int).

    Additionally, please note that the len() of this DataLayer is assumed to be the length of the manifest
    after filtering. An incorrect manifest length may lead to some DataLoader issues down the line.

    Args:
        audio_tar_filepaths: Either a list of audio tarball filepaths, or a
            string (can be brace-expandable).
        manifest_filepath (str): Path to the manifest.
        labels (list): Dataset parameter.
            List of target classes that can be output by the speaker recognition model.
        featurizer
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        trim(bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        window_length_in_sec (float): time length of window/slice (in seconds) # Pass this only for speaker recognition and VAD task
        shift_length_in_sec (float): amount of shift of window for generating the frame for VAD task. in a batch # Pass this only for VAD task during inference.
        normalize_audio (bool): Whether to normalize audio signal. Defaults to False.
        shard_strategy (str): Tarred dataset shard distribution strategy chosen as a str value during ddp.
            -   `scatter`: The default shard strategy applied by WebDataset, where each node gets
                a unique set of shards, which are permanently pre-allocated and never changed at runtime.
            -   `replicate`: Optional shard strategy, where each node gets all of the set of shards
                available in the tarred dataset, which are permanently pre-allocated and never changed at runtime.
                The benefit of replication is that it allows each node to sample data points from the entire
                dataset independently of other nodes, and reduces dependence on value of `shuffle_n`.

                Note: Replicated strategy allows every node to sample the entire set of available tarfiles,
                and therefore more than one node may sample the same tarfile, and even sample the same
                data points! As such, there is no assured guarantee that all samples in the dataset will be
                sampled at least once during 1 epoch.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 0.
    """

    def __init__(
        self,
        *,
        audio_tar_filepaths: Union[str, List[str]],
        manifest_filepath: Union[str, List[str]],
        labels: List[str],
        featurizer,
        shuffle_n: int = 0,
        min_duration: Optional[float] = 0.1,
        max_duration: Optional[float] = None,
        trim: bool = False,
        window_length_in_sec: Optional[float] = 8,
        shift_length_in_sec: Optional[float] = 1,
        normalize_audio: bool = False,
        shard_strategy: str = "scatter",
        global_rank: int = 0,
        world_size: int = 0,
    ):
        logging.info("Window/slice length considered for collate func is {}".format(window_length_in_sec))
        logging.info("Shift length considered for collate func is {}".format(shift_length_in_sec))
        self.window_length_in_sec = window_length_in_sec
        self.shift_length_in_sec = shift_length_in_sec
        self.normalize_audio = normalize_audio

        super().__init__(
            audio_tar_filepaths=audio_tar_filepaths,
            manifest_filepath=manifest_filepath,
            labels=labels,
            featurizer=featurizer,
            shuffle_n=shuffle_n,
            min_duration=min_duration,
            max_duration=max_duration,
            trim=trim,
            shard_strategy=shard_strategy,
            global_rank=global_rank,
            world_size=world_size,
        )

    def fixed_seq_collate_fn(self, batch):
        return _fixed_seq_collate_fn(self, batch)

    def sliced_seq_collate_fn(self, batch):
        return _sliced_seq_collate_fn(self, batch)

    def vad_frame_seq_collate_fn(self, batch):
        return _vad_frame_seq_collate_fn(self, batch)


class _AudioDiarTrainDataset(Dataset):
    """
    Dataset class that loads a json file containing paths to audio files,
    RTTM files and number of speakers. This Dataset class is designed for
    training or fine-tuning speaker embedding extractor and diarization decoder
    at the same time.

    Example:
    {"audio_filepath": "/path/to/audio_wav_0.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_0.rttm}
    ...
    {"audio_filepath": "/path/to/audio_wav_n.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_n.rttm}

    Args:
        manifest_filepath (str):
            Path to input manifest json files.
        clus_label_dict (Dict):
            Segment-level speaker labels from Clustering results.
        soft_label_thres (float):
            Threshold that determines the label of each segment based on RTTM file information.
        featurizer:
            Featurizer instance for generating features from the raw waveform.
        window_stride (float):
            Window stride for acoustic feature. This value is used for calculating the numbers of feature-level frames.
        emb_batch_size (int):
            Number of embedding vectors that are trained with attached computational graphs.
        max_spks (int):
            Integer value that limits the number of speakers for the model that is being trained.
        bi_ch_infer (bool):
            This variable should be True if dataloader is created for an inference task.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
        """
        output_types = {
            "features": NeuralType(('B', 'T'), AudioSignal()),
            "feature_length": NeuralType(('B'), LengthsType()),
            "ms_seg_timestamps": NeuralType(('B', 'C', 'T', 'D'), LengthsType()),
            "ms_seg_counts": NeuralType(('B', 'C'), LengthsType()),
            "clus_label_index": NeuralType(('B', 'T'), LengthsType()),
            "scale_mapping": NeuralType(('B', 'C', 'T'), LengthsType()),
            "targets": NeuralType(('B', 'T', 'C'), ProbsType()),
        }

        return output_types

    def __init__(
        self,
        *,
        manifest_filepath: str,
        multiscale_args_dict: str,
        ms_ts_dict: Dict,
        soft_label_thres: float,
        featurizer,
        window_stride,
        emb_batch_size,
        max_spks: int,
        bi_ch_infer: bool,
        random_flip: bool = True,
    ):
        super().__init__()
        self.collection = DiarizationSpeechLabel(
            manifests_files=manifest_filepath.split(','),
            emb_dict=None,
            clus_label_dict=None,
            max_spks=max_spks,
            bi_ch_infer=bi_ch_infer,
        )
        self.featurizer = featurizer
        self.multiscale_args_dict = multiscale_args_dict
        self.ms_ts_dict = ms_ts_dict
        self.round_digits = 2
        self.decim = 10 ** self.round_digits
        self.fr_per_sec = 100
        self.soft_label_thres = soft_label_thres
        self.bi_ch_infer = bi_ch_infer
        self.max_spks = 2
        self.feat_per_sec = int(1 / window_stride)
        self.emb_batch_size = emb_batch_size
        self.random_flip = random_flip

    def __len__(self):
        return len(self.collection)

    def s2n(self, x):
        """Convert string to float then round the number."""
        return round(float(x), self.round_digits)

    def assign_labels_to_longer_segs(self, uniq_id, base_scale_clus_label):
        """
        Assign the generated speaker labels from the base scale (the finest scale) to the longer scales.
        This process is needed to get the cluster labels for each scale. The cluster labels are needed to
        calculate the cluster-average speaker embedding for each scale.

        Args:
            uniq_id (str):
                Unique sample ID for training.
            base_scale_clus_label (torch.tensor):
                Tensor varialbe containing the speaker labels for the base-scale segments.
        Returns:
            per_scale_clus_label (torch.tensor):
                Tensor variable containing the speaker labels for each segment in each scale.
                Note that the total length of the speaker label sequence differs over scale since
                each scale has a different number of segments for the same session.

            scale_mapping (torch.tensor):
                Matrix containing the segment indices of each scale. scale_mapping is necessary for reshaping the
                multiscale embeddings to form an input matrix for the MSDD model.
        """
        per_scale_clus_label = []
        self.scale_n = len(self.ms_ts_dict[uniq_id]['scale_dict'])
        uniq_scale_mapping = get_scale_mapping_list(self.ms_ts_dict[uniq_id])
        for scale_index in range(self.scale_n):
            new_clus_label = []
            max_index = max(uniq_scale_mapping[scale_index])
            for seg_idx in range(max_index + 1):
                if seg_idx in uniq_scale_mapping[scale_index]:
                    seg_clus_label = mode(base_scale_clus_label[uniq_scale_mapping[scale_index] == seg_idx])
                else:
                    seg_clus_label = 0 if len(new_clus_label) == 0 else new_clus_label[-1]
                new_clus_label.append(seg_clus_label)
            per_scale_clus_label.extend(new_clus_label)
        per_scale_clus_label = torch.tensor(per_scale_clus_label)
        return per_scale_clus_label, uniq_scale_mapping

    def get_diar_target_labels(self, uniq_id, fr_level_target):
        """
        Convert frame-level diarization target varialbe into segment-level target variable. Since the granularity is reduced
        from frame level (10ms) to segment level (100ms~500ms), we need a threshold value, soft_label_thres, which determines
        the label of each segment based on the overlap between a segment range (start and end time) and the frame-level target variable.

        Args:
            uniq_id (str):
                Unique file ID that refers to an input audio file and corresponding RTTM (Annotation) file.
            fr_level_target (torch.tensor):
                Tensor containing label for each feature-level frame.
        Returns:
            seg_target (torch.tensor):
                Tensor containing binary speaker labels for base-scale segments.
            base_clus_label (torch.tensor):
                Representative speaker label for each segment. This variable only has one speaker label for each base-scale segment.

        """
        seg_target_list, base_clus_label = [], []
        self.scale_n = len(self.ms_ts_dict[uniq_id]['scale_dict'])
        subseg_time_stamp_list = self.ms_ts_dict[uniq_id]["scale_dict"][self.scale_n - 1]["time_stamps"]
        for line in subseg_time_stamp_list:
            line_split = line.split()
            seg_stt, seg_end = float(line_split[0]), float(line_split[1])
            seg_stt_fr, seg_end_fr = int(seg_stt * self.fr_per_sec), int(seg_end * self.fr_per_sec)
            soft_label_vec = torch.sum(fr_level_target[seg_stt_fr:seg_end_fr, :], axis=0) / (seg_end_fr - seg_stt_fr)
            label_int = torch.argmax(soft_label_vec)
            label_vec = (soft_label_vec > self.soft_label_thres).float()
            seg_target_list.append(label_vec.detach())
            base_clus_label.append(label_int.detach())
        seg_target = torch.stack(seg_target_list)
        base_clus_label = torch.stack(base_clus_label)
        return seg_target, base_clus_label

    def parse_rttm_for_ms_targets(self, sample, target_spks=None):
        """
        Generate target tensor variable by extracting groundtruth diarization labels from an RTTM file.
        This function converts (start, end, speaker_id) format into base-scale (the finest scale) segment level
        diarization label in a matrix form.

        Example of seg_target:
            [[0., 1.], [0., 1.], [1., 1.], [1., 0.], [1., 0.], ..., [0., 1.]]

        Args:
            sample:
                DiarizationSpeechLabel instance containing the following variables.

                audio_file (str):
                    Path of the input audio file (raw waveform).
                offset (str):
                    Offset of the input audio file provided in the input manifest file.
                duration (float):
                    Duration of the input audio file provided in the input manifest file.
                rttm_file (str):
                    Path of the groundtruth diarization annotation file (RTTM format).

        Returns:
            clus_label_index (torch.tensor):
                Groundtruth Clustering label (cluster index for each segment) from RTTM files for training purpose.
            seg_target  (torch.tensor):
                Tensor varialble containing hard-labels of speaker activity in each base-scale segment.
            scale_mapping (torch.tensor):
                Matrix containing the segment indices of each scale. scale_mapping is necessary for reshaping the
                multiscale embeddings to form an input matrix for the MSDD model.

        """
        rttm_lines = open(sample.rttm_file).readlines()
        uniq_id = self.get_uniq_id_with_range(sample)
        rttm_timestamps = self.extract_seg_info_from_rttm(uniq_id, rttm_lines)
        fr_level_target = self.assign_frame_level_spk_vector(uniq_id, rttm_timestamps, target_spks=None)
        seg_target, base_clus_label = self.get_diar_target_labels(uniq_id, fr_level_target)
        clus_label_index, scale_mapping = self.assign_labels_to_longer_segs(uniq_id, base_clus_label)
        return clus_label_index, seg_target, scale_mapping

    @staticmethod
    def read_rttm_file(rttm_path):
        return open(rttm_path).readlines()

    @staticmethod
    def get_uniq_id_with_range(sample, deci=3):
        """
        Generate unique training sample ID from unique file ID, offset and duration. The start-end time added
        unique ID is required for identifying the sample since multiple short audio samples are generated from a single
        audio file. The start time and end time of the audio stream uses millisecond units if deci=3.

        Args:
            sample:
                DiarizationSpeechLabel instance from collections.
        Returns:
            uniq_id (str):
                Unique sample ID which includes start and end time of the audio stream.
                Example: abc1001_3122_6458

        """
        bare_uniq_id = sample.rttm_file.split('/')[-1].split('.rttm')[0]
        offset = str(int(round(sample.offset, deci) * pow(10, deci)))
        endtime = str(int(round(sample.offset + sample.duration, deci) * pow(10, deci)))
        uniq_id = f"{bare_uniq_id}_{offset}_{endtime}"
        return uniq_id

    def getRepeatedList(self, mapping_argmat, score_mat_size):
        """
        Count the numbers in the mapping dictionary and create lists that contain repeated indices to be
        used for creating the repeated affinity matrix for fusing the affinity values.
        """
        count_dict = dict(Counter(mapping_argmat))
        repeat_list = []
        for k in range(score_mat_size):
            if k in count_dict:
                repeat_list.append(count_dict[k])
            else:
                repeat_list.append(0)
        return repeat_list

    def get_ms_seg_timestamps(self, sample):
        """
        Get start and end time of segments in each scale.

        Args:
            sample:
                DiarizationSpeechLabel instance from preprocessing.collections
        Returns:
            ms_seg_timestamps (torch.tensor):
                Tensor containing Multiscale segment timestamps.
            ms_seg_counts (torch.tensor):
                The number of segments for each scale. This information is used for reshaping embedding batch
                during forward propagation.
        """
        uniq_id = self.get_uniq_id_with_range(sample)
        ms_seg_timestamps_list = []
        max_seq_len = len(self.ms_ts_dict[uniq_id]["scale_dict"][self.scale_n - 1]["time_stamps"])
        ms_seg_counts = [0 for _ in range(self.scale_n)]
        for scale_idx in range(self.scale_n):
            scale_ts_list = []
            for k, line in enumerate(self.ms_ts_dict[uniq_id]["scale_dict"][scale_idx]["time_stamps"]):
                line_split = line.split()
                seg_stt, seg_end = float(line_split[0]), float(line_split[1])
                stt, end = (
                    int((seg_stt - sample.offset) * self.feat_per_sec),
                    int((seg_end - sample.offset) * self.feat_per_sec),
                )
                scale_ts_list.append(torch.tensor([stt, end]).detach())
            ms_seg_counts[scale_idx] = len(self.ms_ts_dict[uniq_id]["scale_dict"][scale_idx]["time_stamps"])
            scale_ts = torch.stack(scale_ts_list)
            scale_ts_padded = torch.cat([scale_ts, torch.zeros(max_seq_len - len(scale_ts_list), 2)], dim=0)
            ms_seg_timestamps_list.append(scale_ts_padded.detach())
        ms_seg_timestamps = torch.stack(ms_seg_timestamps_list)
        ms_seg_counts = torch.tensor(ms_seg_counts)
        return ms_seg_timestamps, ms_seg_counts

    def __getitem__(self, index):
        sample = self.collection[index]
        if sample.offset is None:
            sample.offset = 0
        clus_label_index, targets, scale_mapping = self.parse_rttm_for_ms_targets(sample)
        features = self.featurizer.process(sample.audio_file, offset=sample.offset, duration=sample.duration)
        feature_length = torch.tensor(features.shape[0]).long()
        ms_seg_timestamps, ms_seg_counts = self.get_ms_seg_timestamps(sample)
        if self.random_flip:
            torch.manual_seed(index)
            flip = torch.randperm(self.max_spks)
            clus_label_index, targets = flip[clus_label_index], targets[:, flip]
        return features, feature_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, targets


class _AudioMSDDDataset(Dataset):
    """
    Dataset class that loads a json file containing paths to audio files,
    rttm files and number of speakers. This Dataset class is built for diarization inference and
    evaluation. Speaker embedding sequences, segment timestamps, cluster-average speaker embeddings
    are loaded from memory and fed into dataloader.

    Example:
    {"audio_filepath": "/path/to/audio_wav_0.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_0.rttm}
    ...
    {"audio_filepath": "/path/to/audio_wav_n.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_n.rttm}

    Args:
        manifest_filepath (str):
             Path to input manifest json files.
        emb_dict (Dict):
            Dictionary containing cluster-average embeddings and speaker mapping information.
        emb_seq (Dict):
            Dictionary containing multiscale speaker embedding sequence, scale mapping and corresponding segment timestamps.
        clus_label_dict (Dict):
            Segment-level speaker labels from clustering results.
        soft_label_thres (float):
            A threshold that determines the label of each segment based on RTTM file information.
        featurizer:
            Featurizer instance for generating features from raw waveform.
        max_spks (int):
            Integer value that limits the number of speakers.
        bi_ch_infer (bool):
            This variable should be True if dataloader is created for an inference task.
        use_single_scale_clus (bool):
            Use only one scale for clustering instead of using multiple scales of embeddings for clustering.
        seq_eval_mode (bool):
            If True, F1 score will be calculated for each speaker pair as in the validation accuracy in training mode.
        bi_ch_infer (bool):
            If True, this Dataset class stays in inference mode.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
        """
        output_types = OrderedDict(
            {
                "ms_emb_seq": NeuralType(('B', 'T', 'C', 'D'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
                "ms_avg_embs": NeuralType(('B', 'C', 'D', 'C'), EncodedRepresentation()),
                "targets": NeuralType(('B', 'T', 'C'), ProbsType()),
            }
        )
        return output_types

    def __init__(
        self,
        *,
        manifest_filepath: str,
        emb_dict: Dict,
        emb_seq: Dict,
        clus_label_dict: Dict,
        soft_label_thres: float,
        max_spks: int,
        seq_eval_mode: bool,
        use_single_scale_clus: bool,
        bi_ch_infer: bool,
    ):
        super().__init__()
        self.collection = DiarizationSpeechLabel(
            manifests_files=manifest_filepath.split(','),
            emb_dict=emb_dict,
            clus_label_dict=clus_label_dict,
            max_spks=max_spks,
            seq_eval_mode=seq_eval_mode,
            bi_ch_infer=bi_ch_infer,
        )
        self.emb_dict = emb_dict
        self.emb_seq = emb_seq
        self.clus_label_dict = clus_label_dict
        self.round_digits = 2
        self.decim = 10 ** self.round_digits
        self.fr_per_sec = 100
        self.soft_label_thres = soft_label_thres
        self.bi_ch_infer = bi_ch_infer
        self.max_spks = max_spks
        self.use_single_scale_clus = use_single_scale_clus
        self.seq_eval_mode = seq_eval_mode

    def __len__(self):
        return len(self.collection)

    def s2n(self, x):
        return round(float(x), self.round_digits)

    def parse_rttm_multiscale(self, sample, target_spks=None):
        """
        Generate target tensor variable by extracting groundtruth diarization labels from an RTTM file.
        This function converts (start, end, speaker_id) format into base-scale (the finest scale) segment level
        diarization label in a matrix form.

        Example of seg_target:
            [[0., 1.], [0., 1.], [1., 1.], [1., 0.], [1., 0.], ..., [0., 1.]]

        Args:
            sample:
                DiarizationSpeechLabel instance containing the following variables.

                audio_file (str):
                    Path of the input audio file (raw waveform).
                offset (str):
                    Offset of the input audio file provided in the input manifest file.
                duration (float):
                    Duration of the input audio file provided in the input manifest file.
                rttm_file (str):
                    Path of the groundtruth diarization annotation file (RTTM format).
                target_spks (tuple):
                    Two Indices of targeted speakers for evaluation.
                    Example: (2, 3)
                sess_spk_dict (Dict):
                    Mapping between RTTM speakers and speaker labels in the clustering result.
                clus_spk_digits (tuple):
                    Tuple containing all the speaker indices from the clustering result.
                    Example: (0, 1, 2, 3)
                rttm_spkr_digits (tuple):
                    Tuple containing all the speaker indices in the RTTM file.
                    Example: (0, 1, 2)

        Returns:
            seg_target (torch.tensor):
                Tensor varialble containing hard-labels of speaker activity in each base-scale segment.
        """
        rttm_lines = open(sample.rttm_file).readlines()
        uniq_id = sample.rttm_file.split('/')[-1].split('.rttm')[0]
        rttm_timestamps = self.extract_seg_info_from_rttm(uniq_id, rttm_lines, target_spks)
        fr_level_target = self.assign_frame_level_spk_vector(uniq_id, rttm_timestamps, target_spks)
        seg_target = self.get_diar_target_labels_from_fr_target(uniq_id, fr_level_target)
        return seg_target

    def get_diar_target_labels_from_fr_target(self, uniq_id, fr_level_target):
        """

        """
        if fr_level_target is None:
            return None
        else:
            seg_target_list = []
            for (seg_stt, seg_end, label_int) in self.clus_label_dict[uniq_id]:
                seg_stt_fr, seg_end_fr = int(seg_stt * self.fr_per_sec), int(seg_end * self.fr_per_sec)
                soft_label_vec = torch.sum(fr_level_target[seg_stt_fr:seg_end_fr, :], axis=0) / (
                    seg_end_fr - seg_stt_fr
                )
                label_vec = (soft_label_vec > self.soft_label_thres).int()
                seg_target_list.append(label_vec)
            seg_target = torch.stack(seg_target_list)
            return seg_target

    def getRepeatedList(self, mapping_argmat, score_mat_size):
        """
        Count the numbers in the mapping dictionary and create lists that contain
        repeated indices to be used for creating the repeated affinity matrix for
        fusing the affinity values.
        """
        count_dict = dict(Counter(mapping_argmat))
        repeat_list = []
        for k in range(score_mat_size):
            if k in count_dict:
                repeat_list.append(count_dict[k])
            else:
                repeat_list.append(0)
        return repeat_list

    def __getitem__(self, index):
        sample = self.collection[index]
        if sample.offset is None:
            sample.offset = 0

        # uniq_id = _get_uniq_id_from_rttm(sample.rttm_file)
        uniq_id = sample.rttm_file.split('/')[-1].split('.rttm')[0]
        scale_n = len(self.emb_dict.keys())
        _avg_embs = torch.stack([self.emb_dict[scale_index][uniq_id]['avg_embs'] for scale_index in range(scale_n)])

        if self.bi_ch_infer:
            avg_embs = _avg_embs[:, :, self.collection[index].target_spks]
        else:
            avg_embs = _avg_embs

        if avg_embs.shape[2] > self.max_spks:
            raise ValueError(
                f" avg_embs.shape[2] {avg_embs.shape[2]} should be less than or equal to self.max_num_speakers {self.max_spks}"
            )

        feats = []
        for scale_index in range(scale_n):
            repeat_mat = self.emb_seq["session_scale_mapping"][uniq_id][scale_index]
            feats.append(self.emb_seq[scale_index][uniq_id][repeat_mat, :])
        feats_out = torch.stack(feats).permute(1, 0, 2)
        feats_len = feats_out.shape[0]

        if self.seq_eval_mode:
            targets = self.parse_rttm_multiscale(sample, self.collection[index].target_spks)
        else:
            targets = torch.zeros(feats_len, 2).float()

        return feats_out, feats_len, targets, avg_embs


def _diar_train_collate_fn(self, batch):
    """
    Collate batch of variables that are needed for raw waveform to diarization label training.
    The following variables are included in training/validation batch:

    Args:
        batch (tuple):
            Batch tuple containing the variables for the diarization training.
    Returns:
        features (torch.tensor):
            Raw waveform samples (time series) loaded from the audio_filepath in the input manifest file.
        feature lengths (time series sample length):
            A list of lengths of the raw waveform samples.
        ms_seg_timestamps (torch.tensor):
            Matrix containing the start time and end time (timestamps) for each segment and each scale.
            ms_seg_timestamps is needed for extracting acoustic features from raw waveforms.
        ms_seg_counts (torch.tensor):
            Matrix containing The number of segments for each scale. ms_seg_counts is necessary for reshaping
            the input matrix for the MSDD model.
        clus_label_index (torch.tensor):
            Groundtruth Clustering label (cluster index for each segment) from RTTM files for training purpose.
            clus_label_index is necessary for calculating cluster-average embedding.
        scale_mapping (torch.tensor):
            Matrix containing the segment indices of each scale. scale_mapping is necessary for reshaping the
            multiscale embeddings to form an input matrix for the MSDD model.
        targets (torch.tensor):
            Groundtruth Speaker label for the given input embedding sequence.
    """
    packed_batch = list(zip(*batch))
    features, feature_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, targets = packed_batch
    features_list, feature_length_list = [], []
    ms_seg_timestamps_list, ms_seg_counts_list, scale_clus_label_list, scale_mapping_list, targets_list = (
        [],
        [],
        [],
        [],
        [],
    )

    max_raw_feat_len = max([x.shape[0] for x in features])
    max_target_len = max([x.shape[0] for x in targets])
    max_total_seg_len = max([x.shape[0] for x in clus_label_index])

    for feat, feat_len, ms_seg_ts, ms_seg_ct, scale_clus, scl_map, tgt in batch:
        seq_len = tgt.shape[0]
        pad_feat = (0, max_raw_feat_len - feat_len)
        pad_tgt = (0, 0, 0, max_target_len - seq_len)
        pad_sm = (0, max_target_len - seq_len)
        pad_ts = (0, 0, 0, max_target_len - seq_len)
        pad_sc = (0, max_total_seg_len - scale_clus.shape[0])
        padded_feat = torch.nn.functional.pad(feat, pad_feat)
        padded_tgt = torch.nn.functional.pad(tgt, pad_tgt)
        padded_sm = torch.nn.functional.pad(scl_map, pad_sm)
        padded_ms_seg_ts = torch.nn.functional.pad(ms_seg_ts, pad_ts)
        padded_scale_clus = torch.nn.functional.pad(scale_clus, pad_sc)

        features_list.append(padded_feat)
        feature_length_list.append(feat_len.clone().detach())
        ms_seg_timestamps_list.append(padded_ms_seg_ts)
        ms_seg_counts_list.append(ms_seg_ct.clone().detach())
        scale_clus_label_list.append(padded_scale_clus)
        scale_mapping_list.append(padded_sm)
        targets_list.append(padded_tgt)

    features = torch.stack(features_list)
    feature_length = torch.stack(feature_length_list)
    ms_seg_timestamps = torch.stack(ms_seg_timestamps_list)
    clus_label_index = torch.stack(scale_clus_label_list)
    ms_seg_counts = torch.stack(ms_seg_counts_list)
    scale_mapping = torch.stack(scale_mapping_list)
    targets = torch.stack(targets_list)
    return features, feature_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, targets


def _msdd_collate_fn(self, batch):
    """
    Collate batch of feats (speaker embeddings), feature lengths, target label sequences and cluster-average embeddings.
    Args:
        batch (tuple):
            Batch tuple containing feats, feats_len, targets and ms_avg_embs.
    Returns:
        feats (torch.tensor):
            Collated speaker embedding with unified length.
        feats_len (torch.tensor):
            The actual length of each embedding sequence without zero padding.
        targets (torch.tensor):
            Groundtruth Speaker label for the given input embedding sequence.
        ms_avg_embs (torch.tensor):
            Cluster-average speaker embedding vectors.
    """

    packed_batch = list(zip(*batch))
    feats, feats_len, targets, ms_avg_embs = packed_batch
    feats_list, flen_list, targets_list, ms_avg_embs_list = [], [], [], []
    max_audio_len = max(feats_len)
    max_target_len = max([x.shape[0] for x in targets])

    for feature, feat_len, target, ivector in batch:
        flen_list.append(feat_len)
        ms_avg_embs_list.append(ivector)
        if feat_len < max_audio_len:
            pad_a = (0, 0, 0, 0, 0, max_audio_len - feat_len)
            pad_t = (0, 0, 0, max_target_len - target.shape[0])
            padded_feature = torch.nn.functional.pad(feature, pad_a)
            padded_target = torch.nn.functional.pad(target, pad_t)
            feats_list.append(padded_feature)
            targets_list.append(padded_target)
        else:
            targets_list.append(target.clone().detach())
            feats_list.append(feature.clone().detach())

    feats = torch.stack(feats_list)
    feats_len = torch.tensor(flen_list)
    targets = torch.stack(targets_list)
    ms_avg_embs = torch.stack(ms_avg_embs_list)
    return feats, feats_len, targets, ms_avg_embs


class AudioToSpeechDiarTrainDataset(_AudioDiarTrainDataset):
    """
    Dataset class that loads a json file containing paths to audio files,
    rttm files and number of speakers. This Dataset class is designed for
    training or fine-tuning speaker embedding extractor and diarization decoder
    at the same time.

    Example:
    {"audio_filepath": "/path/to/audio_wav_0.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_0.rttm}
    ...
    {"audio_filepath": "/path/to/audio_wav_n.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_n.rttm}

    Args:
        manifest_filepath (str):
            Path to input manifest json files.
        clus_label_dict (Dict):
            Segment-level speaker labels from Clustering results.
        soft_label_thres (float):
            A threshold that determines the label of each segment based on RTTM file information.
        featurizer:
            Featurizer instance for generating features from the raw waveform.
        window_stride (float):
            Window stride for acoustic feature. This value is used for calculating the numbers of feature-level frames.
        emb_batch_size (int):
            Number of embedding vectors that are trained with attached computational graphs.
        max_spks (int):
            Integer value that limits the number of speakers for the model that is being trained.
        bi_ch_infer (bool):
            This variable should be True if dataloader is created for an inference task.
    """

    def __init__(
        self,
        *,
        manifest_filepath: str,
        multiscale_args_dict: Dict,
        ms_ts_dict: Dict,
        soft_label_thres: float,
        featurizer,
        window_stride,
        emb_batch_size,
        max_spks: int,
        bi_ch_infer: bool,
    ):
        super().__init__(
            manifest_filepath=manifest_filepath,
            multiscale_args_dict=multiscale_args_dict,
            ms_ts_dict=ms_ts_dict,
            soft_label_thres=soft_label_thres,
            featurizer=featurizer,
            window_stride=window_stride,
            emb_batch_size=emb_batch_size,
            max_spks=max_spks,
            bi_ch_infer=bi_ch_infer,
        )

    def diar_train_collate_fn(self, batch):
        return _diar_train_collate_fn(self, batch)

    def extract_seg_info_from_rttm(self, uniq_id, rttm_lines, target_spks=None):
        return _extract_seg_info_from_rttm(self, uniq_id, rttm_lines, target_spks)

    def assign_frame_level_spk_vector(self, uniq_id, rttm_timestamps, target_spks):
        return _assign_frame_level_spk_vector(self, uniq_id, rttm_timestamps, target_spks)


class AudioToSpeechMSDDDataset(_AudioMSDDDataset):
    """
    Dataset class that loads a json file containing paths to audio files,
    rttm files and number of speakers. The created labels are used for diarization inference.

    Example:
    {"audio_filepath": "/path/to/audio_wav_0.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_0.rttm}
    ...
    {"audio_filepath": "/path/to/audio_wav_n.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_n.rttm}

    Args:
        manifest_filepath (str):
            Path to input manifest json files.
        emb_dict (Dict):
            Dictionary containing cluster-average embeddings and speaker mapping information.
        emb_seq (Dict):
            Dictionary containing multiscale speaker embedding sequence, scale mapping and corresponding segment timestamps.
        clus_label_dict (Dict):
            Segment-level speaker labels from clustering results.
        soft_label_thres (float):
            Threshold that determines speaker labels of segments depending on the overlap with groundtruth speaker timestamps.
        featurizer:
            Featurizer instance for generating features from raw waveform.
        max_spks (int):
            Integer value that limits the number of speakers.
        bi_ch_infer (bool):
            This variable should be True if dataloader is created for an inference task.
        use_single_scale_clus (bool):
            Use only one scale for clustering instead of using multiple scales of embeddings for clustering.
        seq_eval_mode (bool):
            If True, F1 score will be calculated for each speaker pair as in the validation accuracy in training mode.
        bi_ch_infer (bool):
            If True, this Dataset class operates in inference mode. In inference mode, a set of speakers in the input audio
            is split into multiple pairs of speakers and speaker tuples (e.g. 3 speakers: [(0,1), (1,2), (2,3)]) and then
            fed into the diarization system to merge the individual results.
    """

    def __init__(
        self,
        *,
        manifest_filepath: str,
        emb_dict: Dict,
        emb_seq: Dict,
        clus_label_dict: Dict,
        soft_label_thres: float,
        max_spks: int,
        use_single_scale_clus: bool,
        seq_eval_mode: bool,
        bi_ch_infer: bool,
    ):
        super().__init__(
            manifest_filepath=manifest_filepath,
            emb_dict=emb_dict,
            emb_seq=emb_seq,
            clus_label_dict=clus_label_dict,
            soft_label_thres=soft_label_thres,
            max_spks=max_spks,
            use_single_scale_clus=use_single_scale_clus,
            seq_eval_mode=seq_eval_mode,
            bi_ch_infer=bi_ch_infer,
        )

    def msdd_collate_fn(self, batch):
        return _msdd_collate_fn(self, batch)

    def extract_seg_info_from_rttm(self, uniq_id, rttm_lines, target_spks):
        return _extract_seg_info_from_rttm(self, uniq_id, rttm_lines, target_spks)

    def assign_frame_level_spk_vector(self, uniq_id, rttm_timestamps, target_spks):
        return _assign_frame_level_spk_vector(self, uniq_id, rttm_timestamps, target_spks)
