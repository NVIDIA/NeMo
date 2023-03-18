# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import beta, gamma
from tqdm import tqdm

from nemo.collections.asr.parts.preprocessing.perturb import AudioAugmentor
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.utils import logging


def clamp_min_list(target_list: List[float], min_val: float) -> List[float]:
    """
    Clamp numbers in the given list with `min_val`.
    Args:
        target_list (list):
            List containing floating point numbers
        min_val (float):
            Desired minimum value to clamp the numbers in `target_list`

    Returns:
        (list) List containing clamped numbers
    """
    return [max(x, min_val) for x in target_list]


def clamp_max_list(target_list: List[float], max_val: float) -> List[float]:
    """
    Clamp numbers in the given list with `max_val`.
    Args:
        target_list (list):
            List containing floating point numbers
        min_val (float):
            Desired maximum value to clamp the numbers in `target_list`

    Returns:
        (list) List containing clamped numbers
    """
    return [min(x, max_val) for x in target_list]


def get_cleaned_base_path(self, output_dir: str) -> str:
    """
    Delete output directory if it exists or throw warning.

    Args:
        output_dir (str): Path to output directory

    Returns:
        basepath (str): Path to base-path directory for writing output files
    """
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        if self._params.data_simulator.outputs.overwrite_output:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.mkdir(output_dir)
        else:
            raise Exception("Output directory is nonempty and overwrite_output = false")
    elif not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # only add root if paths are relative
    if not os.path.isabs(output_dir):
        ROOT = os.getcwd()
        basepath = os.path.join(ROOT, output_dir)
    else:
        basepath = output_dir
    return basepath


def binary_search_alignments(
    inds: List[int], max_audio_read_sec: float, min_alignment_count: int, audio_manifest: dict
) -> int:
    """
    Binary search to find the index of the alignment that satisfies the maximum audio read duration, 
    `max_audio_read_sec`. This is used to avoid reading the short audio files.
    NOTE: `offset_max` should be at least 1 to avoid feeding max=0 to random sampling function.

    Args:
        inds (list): List of indices to search from
        audio_manifest (dict): Dictionary containing the audio file's alignments

    Returns:
        offset_max (int) Index of the alignment that satisfies the maximum audio read duration
    """
    left, right = 0, len(inds) - min_alignment_count
    while left < right:
        mid = left + (right - left) // 2
        dur_left = audio_manifest['alignments'][-1 * min_alignment_count] - audio_manifest['alignments'][inds[mid]]
        if dur_left < max_audio_read_sec:
            right = mid - 1
        elif dur_left > max_audio_read_sec:
            left = mid + 1
        else:
            break
    offset_max = max(left + (right - left) // 2, 1)
    return offset_max


def get_subset_of_audio_manifest(
    audio_manifest: dict, offset_index: int, max_audio_read_sec: float, min_alignment_count: int,
) -> dict:
    """
    Get a subset of `audio_manifest` for faster audio-file reading.

    Args:
        audio_manifest (dict): Audio manifest dictionary.
            keys: 'offset', 'duration', 'alignments', 'words'
        offset_index (int): Index of the offset.

    Returns:
        audio_manifest (dict): Subset of `audio_manifest` is returned for `words` and `alignments` keys.
    """
    alignment_array = np.array(audio_manifest['alignments'])
    alignment_array_pr = np.array(alignment_array[offset_index:]) - alignment_array[offset_index]
    subset_alignments = alignment_array_pr[alignment_array_pr < max_audio_read_sec]
    if len(subset_alignments) < min_alignment_count:
        # Cases where the word next to the offset is longer than the max_audio_read_sec.
        logging.warning(
            f"subset_alignments of {audio_manifest['audio_filepath']} \n"
            f"has subset alignment length:{len(subset_alignments)} at offset_index:{offset_index}, "
            f"word:{audio_manifest['words'][offset_index:offset_index+min_alignment_count]}, "
            f"alignments:{alignment_array_pr[:min_alignment_count]} which is longer than _max_audio_read_sec:{[0, max_audio_read_sec]}."
            " Truncating the alignements."
        )
        # Attach the `_max_audio_read_sec` to the `subset_alignments` to truncate the alignment timestamp.
        subset_alignments = np.concatenate([subset_alignments, np.array([max_audio_read_sec])])
    audio_manifest['offset'], audio_manifest['duration'] = (
        alignment_array[offset_index],
        subset_alignments[-1] - subset_alignments[0],
    )
    audio_manifest['alignments'] = subset_alignments.tolist()
    audio_manifest['words'] = audio_manifest['words'][offset_index : offset_index + len(subset_alignments)]
    return audio_manifest


def read_audio_from_buffer(
    audio_manifest: dict,
    buffer_dict: dict,
    offset_index: int,
    device: torch.device,
    max_audio_read_sec: float = 2.5,
    min_alignment_count: int = 2,
    read_subset: bool = True,
) -> Tuple[torch.Tensor, int, dict]:
    """
    Read from the provided file path while maintaining a hash-table that saves loading time.
    Also, this function only reads a subset of the audio file if `read_subset` is True for faster audio-file reading.

    Args:
        audio_manifest (dict): Audio manifest dictionary.
            keys: 'audio_filepath', 'duration', 'alignments', 'words'
        buffer_dict (dict): Hash-table that saves loaded audio files.
        offset_index (int): Index of the offset for the audio file.
        read_subset (bool): If True, read a subset of the audio file.
                            To control the length of the audio file, use data_simulator.session_params.max_audio_read_sec.
                            Note that using large value (greater than 3~4 sec) for `max_audio_read_sec` will slow down the generation process.
                            If False, read the entire audio file.

    Returns:
        audio_file (torch.Tensor): Time-series audio data in a tensor.
        sr (int): Sample rate of the audio file.
        audio_manifest (dict): (modified) audio manifest dictionary.
    """
    audio_file_id = f"{audio_manifest['audio_filepath']}#{offset_index}"
    if audio_file_id in buffer_dict:
        audio_file, sr, audio_manifest = buffer_dict[audio_file_id]
    else:
        if read_subset:
            audio_manifest = get_subset_of_audio_manifest(
                audio_manifest=audio_manifest,
                offset_index=offset_index,
                max_audio_read_sec=max_audio_read_sec,
                min_alignment_count=min_alignment_count,
            )
            segment = AudioSegment.from_file(
                audio_file=audio_manifest['audio_filepath'],
                offset=audio_manifest['offset'],
                duration=audio_manifest['duration'],
            )
        else:
            segment = AudioSegment.from_file(audio_file=audio_manifest['audio_filepath'])
        audio_file, sr = torch.from_numpy(segment.samples).to(device), segment.sample_rate
        if read_subset and segment.duration < (audio_manifest['alignments'][-1] - audio_manifest['alignments'][0]):
            audio_manifest['alignments'][-1] = min(segment.duration, audio_manifest['alignments'][-1])
        if audio_file.ndim > 1:
            audio_file = torch.mean(audio_file, 1, False).to(device)
        buffer_dict[audio_file_id] = (audio_file, sr, audio_manifest)
    return audio_file, sr, audio_manifest


def perturb_audio(audio: torch.Tensor, sr: int, augmentor: AudioAugmentor, device: torch.device) -> torch.Tensor:
    """
    Perturb the audio (segment or session) using audio augmentor.

    Args:
        audio (torch.Tensor): Time-series signal of the segment
        sr (int): Sample rate of the original audio file
        augmentor (AudioAugmentor): Audio augmentor to use

    Returns:
        audio (torch.Tensor): Perturbed audio (time-series signal) of the segment
    """
    if augmentor is None:
        return audio
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    audio_segment = AudioSegment(audio, sample_rate=sr)
    augmentor.perturb(audio_segment)
    audio_segment = torch.from_numpy(audio_segment.samples).to(device)
    return audio_segment


def get_random_offset_index(
    audio_manifest: dict,
    audio_read_buffer_dict: dict,
    offset_min: int = 0,
    max_audio_read_sec: float = 2.5,
    min_alignment_count: int = 2,
) -> int:
    """
    Get an index for randomly accessing the silence in alignment timestamps. 

    Args:
        audio_manifest (dict): Audio manifest dictionary.
            keys: 'audio_filepath', 'duration', 'alignments', 'words'

    Returns: 
        (int): Random offset index smaller than `offset_count`.
    """
    if len(audio_manifest['alignments']) <= min_alignment_count:
        raise ValueError(
            f"Audio file {audio_manifest['audio_filepath']} has less than {min_alignment_count} alignment timestamps."
        )
    index_file_id = f"{audio_manifest['audio_filepath']}#index"

    # Avoid multiple indexings of the same audio file by using a hash-table.
    if index_file_id in audio_read_buffer_dict:
        (sil_inds, offset_max) = audio_read_buffer_dict[index_file_id]
    else:
        # Find all silence indices
        sil_inds = np.where((np.array(audio_manifest['words']) == '') == True)[0]
        if audio_manifest['alignments'][-1] - audio_manifest['alignments'][0] < max_audio_read_sec:
            # The total duration is already short, therefore skip range search.
            offset_max = 1
        else:
            # Find the range that satisfies `max_audio_read_sec` duration.
            offset_max = binary_search_alignments(
                inds=sil_inds,
                max_audio_read_sec=max_audio_read_sec,
                min_alignment_count=min_alignment_count,
                audio_manifest=audio_manifest,
            )
        audio_read_buffer_dict[index_file_id] = (sil_inds, offset_max)

    # If the audio file is shorter than the max_audio_read_sec, then we don't need to read a subset of the audio file.
    if (
        len(sil_inds) <= min_alignment_count
        or (audio_manifest['alignments'][-1] - audio_manifest['alignments'][0]) < max_audio_read_sec
    ):
        return offset_min
    else:
        offset_index = np.random.randint(offset_min, offset_max)
        return sil_inds[offset_index]


def get_speaker_ids(sess_idx, speaker_samples, permutated_speaker_inds) -> List[str]:
    """
    Randomly select speaker IDs from the loaded manifest file.

    Returns:
        speaker_ids (list): List of speaker IDs
    """
    all_speaker_ids = list(speaker_samples.keys())
    idx_list = permutated_speaker_inds[sess_idx, :]
    speaker_ids = [all_speaker_ids[i] for i in idx_list]
    return speaker_ids


def build_speaker_samples_map(manifest) -> Dict:
    """
    Build a dictionary for mapping speaker ID to their list of samples

    Returns:
        speaker_samples (Dict[list]):
            Dictionary mapping speaker ID to their list of samples
    """
    speaker_samples = defaultdict(list)
    logging.info("Building speaker to samples map...")
    for sample in tqdm(manifest, total=len(manifest)):
        speaker_id = sample['speaker_id']
        speaker_samples[speaker_id].append(sample)
    return speaker_samples


def sample_noise_manifest(noise_manifest: Dict, num_noise_files: int) -> list:
    """
    Sample noise manifest to a specified count `num_noise_files` for the current simulated audio session.

    Args:
        noise_manifest (list): 
            List of noise source samples to be sampled from.

    Returns:
        sampled_noise_manifest (list):
            List of noise samples to be used for the current session.
    """
    num_noise_files = min(len(noise_manifest), num_noise_files)
    sampled_noise_manifest = []
    if num_noise_files > 0:
        selected_noise_ids = np.random.choice(range(len(noise_manifest)), num_noise_files, replace=False)
        for k in selected_noise_ids:
            sampled_noise_manifest.append(noise_manifest[k])
    return sampled_noise_manifest


def read_noise_manifest(add_bg, background_manifest):
    """
    Read the noise manifest file and sample the noise manifest.

    Returns:
        noise_manifest (list): List of the entire noise source samples.
    """
    noise_manifest = []
    if add_bg is True:
        if background_manifest is not None:
            background_manifest_list = background_manifest
            if isinstance(background_manifest_list, str):
                background_manifest_list = [background_manifest_list]
            for background_manifest in background_manifest_list:
                if os.path.exists(background_manifest):
                    noise_manifest += read_manifest(background_manifest)
                else:
                    raise FileNotFoundError(f"Noise manifest file: {background_manifest} file not found.")
        else:
            raise FileNotFoundError(
                f"Noise manifest file is null. Please provide a valid noise manifest file/list if add_bg=True."
            )
    return noise_manifest


def get_speaker_samples(speaker_ids: List[str], speaker_samples: list) -> Dict[str, list]:
    """
    Get a list of the samples for each of the specified speakers.

    Args:
        speaker_ids (list): LibriSpeech speaker IDs for each speaker in the current session.
    
    Returns:
        speaker_wav_align_map (dict): Dictionary containing speaker IDs and their corresponding wav filepath and alignments.
    """
    speaker_wav_align_map = defaultdict(list)
    for sid in speaker_ids:
        speaker_wav_align_map[sid] = speaker_samples[sid]
    return speaker_wav_align_map


def load_speaker_sample(
    speaker_wav_align_map: List[dict], speaker_ids: List[str], speaker_turn: int, output_precision: int
) -> str:
    """
    Load a sample for the selected speaker ID.
    The first alignment and word must be silence that determines the start of the alignments.

    Args:
        speaker_wav_align_map (dict): Dictionary containing speaker IDs and their corresponding wav filepath and alignments.
        speaker_ids (list): LibriSpeech speaker IDs for each speaker in the current session.
        speaker_turn (int): Current speaker turn.
    
    Returns:
        file_path (str): Path to the desired audio file
    """
    speaker_id = speaker_ids[speaker_turn]
    file_id = np.random.randint(0, max(len(speaker_wav_align_map[str(speaker_id)]) - 1, 1))
    file_dict = speaker_wav_align_map[str(speaker_id)][file_id]

    # Check if the alignment file has at least 2 words.
    if len(file_dict['alignments']) < 2:
        raise ValueError(
            f"Alignment file {file_dict['audio_filepath']} has an inappropriate length of {len(file_dict['alignments'])} < 2."
        )

    # Check whether the first word is silence and insert a silence token if the first token is not silence.
    if file_dict['words'][0] != "":
        file_dict['words'].insert(0, "")
        file_dict['alignments'].insert(0, 1 / (10 ** output_precision))
    file_dict = copy.deepcopy(file_dict)
    return file_dict


def get_split_points_in_alignments(
    words: List[str],
    alignments: List[float],
    split_buffer: float,
    sr: int,
    sentence_audio_len: int,
    new_start: float = 0,
):
    """
    Collect split points in the alignment based on silence.
    Silence is defined as a blank symbol between two words that is longer than 2 * split_buffer.

    Args:
        words (List[str]): List of words in the sentence.
        alignments (List[float]): List of alignment timestamps in the sentence.
        split_buffer (float): Buffer length in seconds.
        sr (int): Sample rate of the audio.
        sentence_audio_len (int): Length of the sentence audio in samples.
        new_start (float): Start of the sentence audio in seconds.

    Returns:
        splits (List[List[int]]): List of integer split points in the sentence audio.
    """
    splits = []
    for i in range(len(words)):
        if words[i] == "" and i != 0 and i != len(words) - 1:
            silence_length = alignments[i] - alignments[i - 1]
            if silence_length > 2 * split_buffer:  # split utterance on silence
                new_end = alignments[i - 1] + split_buffer
                splits.append(
                    [int(new_start * sr), int(new_end * sr),]
                )
                new_start = alignments[i] - split_buffer
    # The last split point should be added
    splits.append([int(new_start * sr), sentence_audio_len])
    return splits


def per_speaker_normalize(
    sentence_audio: torch.Tensor, splits: List[List[int]], speaker_turn: int, volume: List[float], device: torch.device
) -> torch.Tensor:
    """
    Normalize time-series audio signal per speaker.

    Args:
        sentence_audio (torch.Tensor): Time-series audio signal.
        splits (List[List[int]]): List of integer split points in the sentence audio.
        speaker_turn (int): Speaker ID of the current speaker.

    Returns:
        sentence_audio (torch.Tensor): Normalized time-series audio signal.
    """
    split_length = torch.tensor(0).to(device).double()
    split_sum = torch.tensor(0).to(device).double()
    for split in splits:
        split_length += len(sentence_audio[split[0] : split[1]])
        split_sum += torch.sum(sentence_audio[split[0] : split[1]] ** 2)
    average_rms = torch.sqrt(split_sum * 1.0 / split_length)
    sentence_audio = sentence_audio / (1.0 * average_rms) * volume[speaker_turn]
    return sentence_audio


def get_session_silence_mean(mean: float, var: float = 0.0) -> float:
    """
    Get the target mean silence for current session using re-parameterized Beta distribution.
    The following constraints are applied to make a > 0 and b > 0:

        0 < mean_silence < 1
        0 < mean_silence_var < mean_silence * (1 - mean_silence)

    Args:
        silence_mean (float): 
            Target mean silence for the current session
    """
    if var > 0:
        a = mean ** 2 * (1 - mean) / var - mean
        b = mean * (1 - mean) ** 2 / var - (1 - mean)
        if a < 0 or b < 0:
            raise ValueError(
                f"Beta(a, b), a = {a:.3f} and b = {b:.3f} should be both greater than 0. "
                f"Invalid `mean_silence_var` value {var} for sampling from Beta distribution. "
                f"`mean_silence_var` should be less than `mean_silence * (1 - mean_silence)`. "
                f"Please check `mean_silence_var` and try again."
            )
        silence_mean = beta(a, b).rvs()
    else:
        silence_mean = mean
    return silence_mean


def get_session_overlap_mean(mean: float, var: float = 0.0) -> float:
    """
    Get the target mean overlap for current session using re-parameterized Beta distribution.
    The following constraints are applied to make a > 0 and b > 0:

        0 < mean_overlap < 1
        0 < mean_overlap_var < mean_overlap * (1 - mean_overlap)

    Returns:
        overlap_mean (float):
            Target mean overlap for the current session
    """
    if var > 0:
        a = mean ** 2 * (1 - mean) / var - mean
        b = mean * (1 - mean) ** 2 / var - (1 - mean)
        if a < 0 or b < 0:
            raise ValueError(
                f"Beta(a, b), a = {a:.3f} and b = {b:.3f} should be both greater than 0. "
                f"Invalid `mean_overlap_var` value {var} for sampling from Beta distribution. "
                f"`mean_overlap_var` should be less than `mean_overlap * (1 - mean_overlap)`. "
                f"Please check `mean_overlap_var` and try again."
            )
        overlap_mean = beta(a, b).rvs()
    else:
        overlap_mean = mean
    return overlap_mean


def sample_from_overlap_model(
    non_silence_len_samples: int,
    sess_overlap_mean: float,
    running_overlap_len_samples: int,
    per_overlap_min_len: int,
    per_overlap_max_len: int,
    missing_overlap: int,
    per_overlap_var: float,
    add_missing_overlap: bool,
) -> int:
    """
    Sample from the overlap model to determine the amount of overlap between segments.
    Gamma distribution is employed for modeling  the highly skewed distribution of overlap length distribution.
    When we add an overlap occurrence, we want to meet the desired overlap ratio defined by `sess_overlap_mean`.
    Let `overlap_mean` be the desired overlap amount, then the mean and variance of the gamma distribution is given by:

        sess_overlap_mean = (overlap_mean + running_overlap_len_samples) / (overlap_mean + non_silence_len_samples)

    The above equation is setting `overlap_mean` to yield the desired overlap ratio `sess_overlap_mean`. 
    We use the above `overlap_mean` value to sample overlap-length for each overlap occurrence.
    
    Args:
        non_silence_len_samples (int): 
            The total amount of non-silence (speech) region regardless of overlap status

    Returns:
        desired_overlap_amount (int): 
            Amount of overlap between segments (in terms of number of samples).
    """
    overlap_mean = ((sess_overlap_mean * non_silence_len_samples) - running_overlap_len_samples) / (
        1 - sess_overlap_mean
    )
    overlap_mean = max(per_overlap_min_len, min(max(0, overlap_mean), per_overlap_max_len))
    if add_missing_overlap:
        overlap_mean += missing_overlap

    if overlap_mean > 0:
        overlap_var = per_overlap_var

        desired_overlap_amount = (
            int(gamma(a=overlap_mean ** 2 / overlap_var, scale=overlap_var / overlap_mean).rvs())
            if overlap_var > 0
            else int(overlap_mean)
        )
        desired_overlap_amount = max(per_overlap_min_len, min(desired_overlap_amount, per_overlap_max_len))
    else:
        desired_overlap_amount = 0

    return desired_overlap_amount


def sample_from_silence_model(
    running_len_samples: int,
    session_len_samples: int,
    sess_silence_mean: float,
    running_silence_len_samples: int,
    per_silence_min_len: int,
    per_silence_max_len: int,
    per_silence_var: float,
) -> int:
    """
    Sample from the silence model to determine the amount of silence to add between sentences.
    Gamma distribution is employed for modeling the highly skewed distribution of silence length distribution.
    When we add silence between sentences, we want to ensure that the proportion of silence meets the `sess_silence_mean`.
    Thus, we employ the following formula to determine the amount of silence to add:

        running_ratio = running_len_samples / session_len_samples
        silence_mean = (session_len_samples*(sess_silence_mean) - running_silence_len_samples) * running_ratio.

    `running_ratio` is the proportion of the created session compared to the targeted total session length.

    Args:
        running_len_samples (int): 
            Running length of the session (in terms of number of samples).
        session_len_samples (int):
            Targeted total session length (in terms of number of samples).

    Returns:
        silence_amount (int): Amount of silence to add between sentences (in terms of number of samples).
    """
    running_ratio = running_len_samples / session_len_samples
    silence_mean = (session_len_samples * (sess_silence_mean) - running_silence_len_samples) * running_ratio
    silence_mean = max(per_silence_min_len, min(silence_mean, per_silence_max_len))
    if silence_mean > 0:
        silence_var = per_silence_var
        silence_amount = (
            int(gamma(a=(silence_mean ** 2) / silence_var, scale=silence_var / silence_mean).rvs())
            if silence_var > 0
            else int(silence_mean)
        )
        silence_amount = max(per_silence_min_len, min(silence_amount, per_silence_max_len))
    else:
        silence_amount = 0

    return silence_amount


class AnnotationGenerator(object):
    def __init__(self, cfg):
        self._params = cfg

    def _create_new_rttm_entry(
        self, words: List[str], alignments: List[float], start: int, end: int, speaker_id: int
    ) -> List[str]:
        """
        Create new RTTM entries (to write to output rttm file)

        Args:
            start (int): Current start of the audio file being inserted.
            end (int): End of the audio file being inserted.
            speaker_id (int): LibriSpeech speaker ID for the current entry.
        
        Returns:
            rttm_list (list): List of rttm entries
        """
        rttm_list = []
        new_start = start
        # look for split locations
        for i in range(len(words)):
            if words[i] == "" and i != 0 and i != len(words) - 1:
                silence_length = alignments[i] - alignments[i - 1]
                if (
                    silence_length > 2 * self._params.data_simulator.session_params.split_buffer
                ):  # split utterance on silence
                    new_end = start + alignments[i - 1] + self._params.data_simulator.session_params.split_buffer
                    t_stt = float(round(new_start, self._params.data_simulator.outputs.output_precision))
                    t_end = float(round(new_end, self._params.data_simulator.outputs.output_precision))
                    rttm_list.append(f"{t_stt} {t_end} {speaker_id}")
                    new_start = start + alignments[i] - self._params.data_simulator.session_params.split_buffer

        t_stt = float(round(new_start, self._params.data_simulator.outputs.output_precision))
        t_end = float(round(end, self._params.data_simulator.outputs.output_precision))
        rttm_list.append(f"{t_stt} {t_end} {speaker_id}")
        return rttm_list

    def _create_new_json_entry(
        self,
        text: List[str],
        wav_filename: str,
        start: int,
        length: int,
        speaker_id: int,
        rttm_filepath: str,
        ctm_filepath: str,
    ) -> dict:
        """
        Create new JSON entries (to write to output json file).

        Args:
            wav_filename (str): Output wav filepath.
            start (int): Current start of the audio file being inserted.
            length (int): Length of the audio file being inserted.
            speaker_id (int): LibriSpeech speaker ID for the current entry.
            rttm_filepath (str): Output rttm filepath.
            ctm_filepath (str): Output ctm filepath.
        
        Returns:
            dict (dict): JSON entry
        """
        start = float(round(start, self._params.data_simulator.outputs.output_precision))
        length = float(round(length, self._params.data_simulator.outputs.output_precision))
        meta = {
            "audio_filepath": wav_filename,
            "offset": start,
            "duration": length,
            "label": speaker_id,
            "text": text,
            "num_speakers": self._params.data_simulator.session_config.num_speakers,
            "rttm_filepath": rttm_filepath,
            "ctm_filepath": ctm_filepath,
            "uem_filepath": None,
        }
        return meta

    def _create_new_ctm_entry(
        self, words: List[str], alignments: List[float], session_name: str, speaker_id: int, start: int
    ) -> List[str]:
        """
        Create new CTM entry (to write to output ctm file)

        Args:
            session_name (str): Current session name.
            start (int): Current start of the audio file being inserted.
            speaker_id (int): LibriSpeech speaker ID for the current entry.
        
        Returns:
            arr (list): List of ctm entries
        """
        arr = []
        start = float(round(start, self._params.data_simulator.outputs.output_precision))
        for i in range(len(words)):
            word = words[i]
            if (
                word != ""
            ):  # note that using the current alignments the first word is always empty, so there is no error from indexing the array with i-1
                prev_align = 0 if i == 0 else alignments[i - 1]
                align1 = float(round(prev_align + start, self._params.data_simulator.outputs.output_precision))
                align2 = float(
                    round(alignments[i] - prev_align, self._params.data_simulator.outputs.output_precision,)
                )
                text = f"{session_name} {speaker_id} {align1} {align2} {word} 0\n"
                arr.append((align1, text))
        return arr
