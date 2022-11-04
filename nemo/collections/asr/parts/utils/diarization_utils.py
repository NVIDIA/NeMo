# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import csv
import itertools
import json
import os
from collections import OrderedDict as od
from datetime import datetime
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.parts.utils.speaker_utils import (
    audio_rttm_map,
    get_uniqname_from_filepath,
    labels_to_rttmfile,
    rttm_to_labels,
    write_rttm2manifest,
)
from nemo.utils import logging

try:
    import arpa

    ARPA = True
except ImportError:
    ARPA = False

__all__ = ['OfflineDiarWithASR']


def dump_json_to_file(file_path: str, session_trans_dict: dict):
    """
    Write a json file from the session_trans_dict dictionary.

    Args:
        file_path (str):
            Target filepath where json file is saved
        session_trans_dict (dict):
            Dictionary containing transcript, speaker labels and timestamps
    """
    with open(file_path, "w") as outfile:
        json.dump(session_trans_dict, outfile, indent=4)


def write_txt(w_path: str, val: str):
    """
    Write a text file from the string input.

    Args:
        w_path (str):
            Target path for saving a file
        val (str):
            String variable to be written
    """
    with open(w_path, "w") as output:
        output.write(val + '\n')


def get_per_spk_ref_transcripts(ctm_file_path: str) -> Tuple[List[str], str]:
    """
    Save the reference transcripts separately by its speaker identity.

    Args:
        ctm_file_path (str):
            Filepath to the reference CTM files.

    Returns:
        spk_reference (list):
            List containing the reference transcripts for each speaker.

            Example:
            >>> spk_reference = ["hi how are you well that's nice", "i'm good yeah how is your sister"]

        mix_reference (str):
            Reference transcript from CTM file. This transcript has word sequence in temporal order.

            Example:
            >>> mix_reference = ["hi how are you i'm good well that's nice yeah how is your sister"]
    """
    mix_reference, per_spk_ref_trans_dict = [], {}
    ctm_content = open(ctm_file_path).readlines()
    for ctm_line in ctm_content:
        ctm_split = ctm_line.split()
        spk = ctm_split[1]
        if spk not in per_spk_ref_trans_dict:
            per_spk_ref_trans_dict[spk] = []
        per_spk_ref_trans_dict[spk].append(ctm_split[4])
        mix_reference.append(ctm_split[4])
    spk_reference = [" ".join(word_list) for word_list in per_spk_ref_trans_dict.values()]
    mix_reference = " ".join(mix_reference)
    return spk_reference, mix_reference


def get_per_spk_hyp_transcripts(word_dict_seq_list: List[Dict[str, float]]) -> Tuple[List[str], str]:
    """
    Save the hypothesis transcripts separately according to its speaker identity.

    Args:
        word_dict_seq_list (list):
            List containing words and corresponding word timestamps in dictionary format.

    Returns:
        spk_hypothesis (list):
            Dictionary containing the hypothesis transcript for each speaker. A list containing the sequence
            of words is assigned for each speaker.

            Example:
            >>> spk_hypothesis= ["hi how are you well that's nice", "i'm good yeah how is your sister"]

        mix_hypothesis (str):
            Hypothesis transcript from ASR output. This transcript has word sequence in temporal order.

            Example:
            >>> mix_hypothesis = ["hi how are you i'm good well that's nice yeah how is your sister"]
    """
    mix_hypothesis, per_spk_hyp_trans_dict = [], {}
    for word_dict in word_dict_seq_list:
        spk = word_dict['speaker_label']
        if spk not in per_spk_hyp_trans_dict:
            per_spk_hyp_trans_dict[spk] = []
        per_spk_hyp_trans_dict[spk].append(word_dict['word'])
        mix_hypothesis.append(word_dict['word'])

    # Create a list containing string formatted transcript
    spk_hypothesis = [" ".join(word_list) for word_list in per_spk_hyp_trans_dict.values()]
    mix_hypothesis = " ".join(mix_hypothesis)
    return spk_hypothesis, mix_hypothesis


def concat_perm_word_error_rate(
    spk_hypothesis: List[str], spk_reference: List[str]
) -> Tuple[float, str, str]:
    """
    Calculate a session-level concatenated minimum-permutation word error rate (cpWER) value. cpWER is
    a scoring method that can evaluate speaker diarization and speech recognition performance at the same time.
    cpWER is calculated by going through the following steps.

    1. Concatenate all utterances of each speaker for both reference and hypothesis files.
    2. Compute the WER between the reference and all possible speaker permutations of the hypothesis.
    3. Pick the lowest WER among them (this is assumed to be the best permutation: `min_perm_hyp_trans`).

    cpWER was proposed in the following article:
        CHiME-6 Challenge: Tackling Multispeaker Speech Recognition for Unsegmented Recordings
        https://arxiv.org/pdf/2004.09249.pdf

    Implementation:
        - Brute force permutation method for calculating cpWER has a time complexity of `O(n!)`.
        - To reduce the computational burden, linear sum assignment (LSA) algorithm is applied
          (also known as Hungarian algorithm) to find the permutation that leads to the lowest WER.
        - In this implementation, instead of calculating all WER values for all permutation of hypotheses,
          we only calculate WER values of (estimated number of speakers) x (reference number of speakers)
          combinations with `O(n^2)`) time complexity and then select the permutation that yields the lowest
          WER based on LSA algorithm.
        - LSA algorithm has `O(n^3)` time complexity in the worst case.
        - Note that we discard redundant speakers when the number of the estimated speakers are larger than
          the number of reference speakers. Thus, the words belong to the redundant speakers are considered
          as deletions during WER calculation.

    Args:
        spk_hypothesis (list):
            List containing the hypothesis transcript for each speaker. A list containing the sequence
            of words is assigned for each speaker.

            Example:
            >>> spk_hypothesis = ["hey how are you we that's nice", "i'm good yes hi is your sister"]

        spk_reference (list):
            List containing the reference transcript for each speaker. A list containing the sequence
            of words is assigned for each speaker.

            Example:
            >>> spk_reference = ["hi how are you well that's nice", "i'm good yeah how is your sister"]

    Returns:
        cpWER (float):
            cpWER value for the given session.
        min_perm_hyp_trans (str):
            Hypothesis transcript containing the permutation that minimizes WER. Words are separated by spaces.
        ref_trans (str):
            Reference transcript in an arbitrary permutation. Words are separated by spaces.
    """
    ref_word_list = []

    # Concatenate the reference transcripts into a string variable
    ref_trans = " ".join(spk_reference)

    # Get all pairs of (estimated num of spks) x (reference num of spks) combinations
    hyp_ref_pair = [spk_hypothesis, spk_reference]
    all_pairs = list(itertools.product(*hyp_ref_pair))

    # Calculate WER for each speaker in hypothesis with reference
    # There are (number of hyp speakers) x (number of ref speakers) combinations
    lsa_wer_list = []
    for (spk_hyp_trans, spk_ref_trans) in all_pairs:
        spk_wer = word_error_rate(hypotheses=[spk_hyp_trans], references=[spk_ref_trans])
        lsa_wer_list.append(spk_wer)

    # Make a cost matrix and calculate a linear sum assignment on the cost matrix.
    # Row is hypothesis index and column is reference index
    cost_wer = np.array(lsa_wer_list).reshape([len(spk_hypothesis), len(spk_reference)])
    row_hyp_ind, col_ref_ind = linear_sum_assignment(cost_wer)

    # In case where hypothesis has more speakers, add words from residual speakers
    hyp_permed = [spk_hypothesis[k] for k in row_hyp_ind]
    min_perm_hyp_trans = " ".join(hyp_permed)

    # Calculate a WER value from the permutation that yields the lowest WER.
    cpWER = word_error_rate(hypotheses=[min_perm_hyp_trans], references=[ref_trans])
    return cpWER, min_perm_hyp_trans, ref_trans


def calculate_total_WER(
    word_seq_lists: List[List[Dict[str, Union[str, float]]]], ctm_file_list: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Launcher function for `calculate_session_cpWER`. Calculate session-level cpWER and average cpWER.
    For detailed information about cpWER, see docstrings of `concat_perm_word_error_rate` function.

    As opposed to cpWER, mixWER is the regular WER value where the hypothesis transcript contains
    words in temporal order regardless of the speakers. mixWER value can be different from cpWER value,
    depending on the speaker diarization results.

    Args:
        word_seq_lists (list):
            List containing multiple lists of dictionaries containing word, word timestamp and speaker label.
            The dictionary has the following keys: `word`, `start_time`, `end_time` and `speaker_label`.

            Example:
            >>> word_dict_seq_list = \
            >>> [ [{'word': 'right', 'start_time': 0.0, 'end_time': 0.04, 'speaker_label': 'speaker_0'},  
                   {'word': 'and', 'start_time': 0.64, 'end_time': 0.68, 'speaker_label': 'speaker_1'},
                   ...],
                  [ {'word': 'hey', 'start_time': 1.84, 'end_time': 2.10, 'speaker_label': 'speaker_0'},  
                  [ {'word': 'but', 'start_time': 2.14, 'end_time': 2.32, 'speaker_label': 'speaker_0'},  
                 ...], 
                  ...
                ]

        ctm_file_list (list):
            List containing filepaths of CTM files.

    Returns:
        wer_results (dict):
            Dictionary containing session level cpWER and WER values.
    """
    wer_results = {'total': {}}
    hyps_spk, refs_spk = [], []
    hyps_mix, refs_mix = [], []

    for k, (word_dict_seq_list, ctm_file_path) in enumerate(zip(word_seq_lists, ctm_file_list)):
        spk_hypothesis, mix_hypotheses = get_per_spk_hyp_transcripts(word_dict_seq_list)
        spk_reference, mix_reference = get_per_spk_ref_transcripts(ctm_file_path)

        # Calculate cpWER and mixWER using the above results
        cpWER, hyp_trans, ref_trans = concat_perm_word_error_rate(spk_hypothesis, spk_reference)
        mixWER = word_error_rate(hypotheses=[mix_hypotheses], references=[mix_reference])

        # Save session-level cpWER and mixWER values
        uniq_id = get_uniqname_from_filepath(ctm_file_path)
        wer_results[uniq_id] = {}
        wer_results[uniq_id]['cpWER'] = cpWER
        wer_results[uniq_id]['mixWER'] = mixWER

        hyps_spk.append(hyp_trans)
        refs_spk.append(ref_trans)
        hyps_mix.append(mix_hypotheses)
        refs_mix.append(mix_reference)

    # Take an average of cpWER and regular WER value on all sessions
    wer_results['total']['average_cpWER'] = word_error_rate(hypotheses=hyps_spk, references=refs_spk)
    wer_results['total']['average_mixWER'] = word_error_rate(hypotheses=hyps_mix, references=refs_mix)
    return wer_results


class OfflineDiarWithASR:
    """
    A class designed for performing ASR and diarization together.

    Attributes:
        cfg_diarizer (OmegaConf):
            Hydra config for diarizer key
        params (OmegaConf):
            Parameters config in diarizer.asr
        ctc_decoder_params (OmegaConf)
            Hydra config for beam search decoder
        realigning_lm_params (OmegaConf):
            Hydra config for realigning language model
        manifest_filepath (str):
            Path to the input manifest path
        nonspeech_threshold (float):
            Threshold for VAD logits that are used for creating speech segments
        fix_word_ts_with_VAD (bool):
            Choose whether to fix word timestamps by using VAD results
        root_path (str):
            Path to the folder where diarization results are saved
        vad_threshold_for_word_ts (float):
            Threshold used for compensating word timestamps with VAD output
        max_word_ts_length_in_sec (float):
            Maximum limit for the duration of each word timestamp
        word_ts_anchor_offset (float):
            Offset for word timestamps from ASR decoders
        run_ASR:
            Placeholder variable for an ASR launcher function
        realigning_lm:
            Placeholder variable for a loaded ARPA Language model
        ctm_exists (bool):
            Boolean that indicates whether all files have the corresponding reference CTM file
        frame_VAD (dict):
            Dictionary containing frame-level VAD logits
        AUDIO_RTTM_MAP:
            Dictionary containing the input manifest information
        color_palette (dict):
            Dictionary containing the ANSI color escape codes for each speaker label (speaker index)
    """

    def __init__(self, cfg_diarizer):
        self.cfg_diarizer = cfg_diarizer
        self.params = cfg_diarizer.asr.parameters
        self.ctc_decoder_params = cfg_diarizer.asr.ctc_decoder_parameters
        self.realigning_lm_params = cfg_diarizer.asr.realigning_lm_parameters
        self.manifest_filepath = cfg_diarizer.manifest_filepath
        self.nonspeech_threshold = self.params.asr_based_vad_threshold
        self.fix_word_ts_with_VAD = self.params.fix_word_ts_with_VAD
        self.root_path = cfg_diarizer.out_dir

        self.vad_threshold_for_word_ts = 0.7
        self.max_word_ts_length_in_sec = 0.6
        self.word_ts_anchor_offset = 0.0
        self.run_ASR = None
        self.realigning_lm = None
        self.ctm_exists = False
        self.frame_VAD = {}

        self.make_file_lists()

        self.color_palette = {
            'speaker_0': '\033[1;32m',
            'speaker_1': '\033[1;34m',
            'speaker_2': '\033[1;30m',
            'speaker_3': '\033[1;31m',
            'speaker_4': '\033[1;35m',
            'speaker_5': '\033[1;36m',
            'speaker_6': '\033[1;37m',
            'speaker_7': '\033[1;30m',
            'speaker_8': '\033[1;33m',
            'speaker_9': '\033[0;34m',
            'white': '\033[0;37m',
        }

        self.csv_columns = [
            'uniq_id',
            'DER',
            'CER',
            'FA',
            'MISS',
            'est_n_spk',
            'ref_n_spk',
            'cpWER',
            'mixWER',
            'mapping',
        ]

    def make_file_lists(self):
        """
        Create lists containing the filepaths of audio clips and CTM files.
        """
        self.AUDIO_RTTM_MAP = audio_rttm_map(self.manifest_filepath)
        self.audio_file_list = [value['audio_filepath'] for _, value in self.AUDIO_RTTM_MAP.items()]

        self.ctm_file_list = []
        for k, audio_file_path in enumerate(self.audio_file_list):
            uniq_id = get_uniqname_from_filepath(audio_file_path)
            if (
                'ctm_filepath' in self.AUDIO_RTTM_MAP[uniq_id]
                and self.AUDIO_RTTM_MAP[uniq_id]['ctm_filepath'] is not None
                and uniq_id in self.AUDIO_RTTM_MAP[uniq_id]['ctm_filepath']
            ):
                self.ctm_file_list.append(self.AUDIO_RTTM_MAP[uniq_id]['ctm_filepath'])

        # check if all unique IDs have CTM files
        if len(self.audio_file_list) == len(self.ctm_file_list):
            self.ctm_exists = True

    def _load_realigning_LM(self):
        """
        Load ARPA language model for realigning speaker labels for words.
        """
        self.N_range = (
            self.realigning_lm_params['min_number_of_words'],
            self.realigning_lm_params['max_number_of_words'],
        )
        self.stt_end_tokens = ['</s>', '<s>']
        logging.info(f"Loading LM for realigning: {self.realigning_lm_params['arpa_language_model']}")
        return arpa.loadf(self.realigning_lm_params['arpa_language_model'])[0]

    def _save_VAD_labels_list(self, word_ts_dict: Dict[str, Dict[str, List[float]]]):
        """
        Take the non_speech labels from logit output. The logit output is obtained from
        `run_ASR` function.

        Args:
            word_ts_dict (dict):
                Dictionary containing word timestamps.
        """
        self.VAD_RTTM_MAP = {}
        for idx, (uniq_id, word_timestamps) in enumerate(word_ts_dict.items()):
            speech_labels_float = self._get_speech_labels_from_decoded_prediction(word_timestamps)
            speech_labels = self._get_str_speech_labels(speech_labels_float)
            output_path = os.path.join(self.root_path, 'pred_rttms')
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            filename = labels_to_rttmfile(speech_labels, uniq_id, output_path)
            self.VAD_RTTM_MAP[uniq_id] = {'audio_filepath': self.audio_file_list[idx], 'rttm_filepath': filename}

    def _get_speech_labels_from_decoded_prediction(self, input_word_ts: List[float]) -> List[float]:
        """
        Extract speech labels from the ASR output (decoded predictions)

        Args:
            input_word_ts (list):
                List containing word timestamps.

        Returns:
            word_ts (list):
                The ranges of the speech segments, which are merged ranges of input_word_ts.
        """
        speech_labels = []
        word_ts = copy.deepcopy(input_word_ts)
        if word_ts == []:
            return speech_labels
        else:
            count = len(word_ts) - 1
            while count > 0:
                if len(word_ts) > 1:
                    if word_ts[count][0] - word_ts[count - 1][1] <= self.nonspeech_threshold:
                        trangeB = word_ts.pop(count)
                        trangeA = word_ts.pop(count - 1)
                        word_ts.insert(count - 1, [trangeA[0], trangeB[1]])
                count -= 1
        return word_ts

    def run_diarization(self, diar_model_config, word_timestamps) -> Dict[str, List[str]]:
        """
        Launch the diarization process using the given VAD timestamp (oracle_manifest).

        Args:
            diar_model_config (OmegaConf):
                Hydra configurations for speaker diarization
            word_and_timestamps (list):
                List containing words and word timestamps

        Returns:
            diar_hyp (dict):
                A dictionary containing rttm results which are indexed by a unique ID.
            score Tuple[pyannote object, dict]:
                A tuple containing pyannote metric instance and mapping dictionary between
                speakers in hypotheses and speakers in reference RTTM files.
        """

        if diar_model_config.diarizer.asr.parameters.asr_based_vad:
            self._save_VAD_labels_list(word_timestamps)
            oracle_manifest = os.path.join(self.root_path, 'asr_vad_manifest.json')
            oracle_manifest = write_rttm2manifest(self.VAD_RTTM_MAP, oracle_manifest)
            diar_model_config.diarizer.vad.model_path = None
            diar_model_config.diarizer.vad.external_vad_manifest = oracle_manifest

        diar_model = ClusteringDiarizer(cfg=diar_model_config)
        score = diar_model.diarize()
        if diar_model_config.diarizer.vad.model_path is not None and not diar_model_config.diarizer.oracle_vad:
            self._get_frame_level_VAD(
                vad_processing_dir=diar_model.vad_pred_dir,
                smoothing_type=diar_model_config.diarizer.vad.parameters.smoothing,
            )

        diar_hyp = {}
        for k, audio_file_path in enumerate(self.audio_file_list):
            uniq_id = get_uniqname_from_filepath(audio_file_path)
            pred_rttm = os.path.join(self.root_path, 'pred_rttms', uniq_id + '.rttm')
            diar_hyp[uniq_id] = rttm_to_labels(pred_rttm)
        return diar_hyp, score

    def _get_frame_level_VAD(self, vad_processing_dir, smoothing_type=False):
        """
        Read frame-level VAD outputs.

        Args:
            vad_processing_dir (str):
                Path to the directory where the VAD results are saved.
            smoothing_type (bool or str): [False, median, mean]
                type of smoothing applied softmax logits to smooth the predictions.
        """
        if isinstance(smoothing_type, bool) and not smoothing_type:
            ext_type = 'frame'
        else:
            ext_type = smoothing_type

        for uniq_id in self.AUDIO_RTTM_MAP:
            frame_vad = os.path.join(vad_processing_dir, uniq_id + '.' + ext_type)
            frame_vad_float_list = []
            with open(frame_vad, 'r') as fp:
                for line in fp.readlines():
                    frame_vad_float_list.append(float(line.strip()))
            self.frame_VAD[uniq_id] = frame_vad_float_list

    def gather_eval_results(
        self, metric, mapping_dict: Dict[str, str], trans_info_dict: Dict[str, Dict[str, float]], decimals: int = 4
    ) -> Dict[str, Dict[str, float]]:
        """
        Gather diarization evaluation results from pyannote DiarizationErrorRate metric object.

        Args:
            metric (DiarizationErrorRate metric):
                DiarizationErrorRate metric pyannote object
            trans_info_dict (dict):
                Dictionary containing word timestamps, speaker labels and words from all sessions.
                Each session is indexed by unique ID as a key.
            mapping_dict (dict):
                Dictionary containing speaker mapping labels for each audio file with key as unique name
            decimals (int):
                The number of rounding decimals for DER value

        Returns:
            der_results (dict):
                Dictionary containing scores for each audio file along with aggregated results
        """
        results = metric.results_
        der_results = {}
        count_correct_spk_counting = 0
        for result in results:
            key, score = result
            pred_rttm = os.path.join(self.root_path, 'pred_rttms', key + '.rttm')
            pred_labels = rttm_to_labels(pred_rttm)

            ref_rttm = self.AUDIO_RTTM_MAP[key]['rttm_filepath']
            ref_labels = rttm_to_labels(ref_rttm)
            ref_n_spk = self.get_num_of_spk_from_labels(ref_labels)
            est_n_spk = self.get_num_of_spk_from_labels(pred_labels)

            if self.cfg_diarizer['oracle_vad']:
                score['missed detection'] = 0
                score['false alarm'] = 0

            _DER, _CER, _FA, _MISS = (
                (score['confusion'] + score['false alarm'] + score['missed detection']) / score['total'],
                score['confusion'] / score['total'],
                score['false alarm'] / score['total'],
                score['missed detection'] / score['total'],
            )

            der_results[key] = {
                "DER": round(_DER, decimals),
                "CER": round(_CER, decimals),
                "FA": round(_FA, decimals),
                "MISS": round(_MISS, decimals),
                "est_n_spk": est_n_spk,
                "ref_n_spk": ref_n_spk,
                "mapping": mapping_dict[key],
            }
            count_correct_spk_counting += int(est_n_spk == ref_n_spk)

        DER, CER, FA, MISS = (
            abs(metric),
            metric['confusion'] / metric['total'],
            metric['false alarm'] / metric['total'],
            metric['missed detection'] / metric['total'],
        )
        der_results["total"] = {
            "DER": DER,
            "CER": CER,
            "FA": FA,
            "MISS": MISS,
            "spk_counting_acc": count_correct_spk_counting / len(metric.results_),
        }

        return der_results

    def _get_the_closest_silence_start(
        self, vad_index_word_end: float, vad_frames: np.ndarray, offset: int = 10
    ) -> float:
        """
        Find the closest silence frame from the given starting position.

        Args:
            vad_index_word_end (float):
                The timestamp of the end of the current word.
            vad_frames (numpy.array):
                The numpy array containing  frame-level VAD probability.
            params (dict):
                Contains the parameters for diarization and ASR decoding.

        Returns:
            cursor (float):
                A timestamp of the earliest start of a silence region from
                the given time point, vad_index_word_end.
        """

        cursor = vad_index_word_end + offset
        limit = int(100 * self.max_word_ts_length_in_sec + vad_index_word_end)
        while cursor < len(vad_frames):
            if vad_frames[cursor] < self.vad_threshold_for_word_ts:
                break
            else:
                cursor += 1
                if cursor > limit:
                    break
        cursor = min(len(vad_frames) - 1, cursor)
        cursor = round(cursor / 100.0, 2)
        return cursor

    def _compensate_word_ts_list(
        self, audio_file_list: List[str], word_ts_dict: Dict[str, List[float]],
    ) -> Dict[str, List[List[float]]]:
        """
        Compensate the word timestamps based on the VAD output.
        The length of each word is capped by self.max_word_ts_length_in_sec.

        Args:
            audio_file_list (list):
                List containing audio file paths.
            word_ts_dict (dict):
                Dictionary containing timestamps of words.

        Returns:
            enhanced_word_ts_dict (dict):
                Dictionary containing the enhanced word timestamp values indexed by unique-IDs.
        """
        enhanced_word_ts_dict = {}
        for idx, (uniq_id, word_ts_seq_list) in enumerate(word_ts_dict.items()):
            N = len(word_ts_seq_list)
            enhanced_word_ts_buffer = []
            for k, word_ts in enumerate(word_ts_seq_list):
                if k < N - 1:
                    word_len = round(word_ts[1] - word_ts[0], 2)
                    len_to_next_word = round(word_ts_seq_list[k + 1][0] - word_ts[0] - 0.01, 2)
                    if uniq_id in self.frame_VAD:
                        vad_index_word_end = int(100 * word_ts[1])
                        closest_sil_stt = self._get_the_closest_silence_start(
                            vad_index_word_end, self.frame_VAD[uniq_id]
                        )
                        vad_est_len = round(closest_sil_stt - word_ts[0], 2)
                    else:
                        vad_est_len = len_to_next_word
                    min_candidate = min(vad_est_len, len_to_next_word)
                    fixed_word_len = max(min(self.max_word_ts_length_in_sec, min_candidate), word_len)
                    enhanced_word_ts_buffer.append([word_ts[0], word_ts[0] + fixed_word_len])
                else:
                    enhanced_word_ts_buffer.append([word_ts[0], word_ts[1]])

            enhanced_word_ts_dict[uniq_id] = enhanced_word_ts_buffer
        return enhanced_word_ts_dict

    def get_transcript_with_speaker_labels(
        self, diar_hyp: Dict[str, List[str]], word_hyp: Dict[str, List[str]], word_ts_hyp: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Match the diarization result with the ASR output.
        The words and the timestamps for the corresponding words are matched in a for loop.

        Args:
            diar_hyp (dict):
                Dictionary of the Diarization output labels in str. Indexed by unique IDs.

                Example:
                >>>  diar_hyp['my_audio_01'] = ['0.0 4.375 speaker_1', '4.375 5.125 speaker_0', ...]

            word_hyp (dict):
                Dictionary of words from ASR inference. Indexed by unique IDs.

                Example:
                >>> word_hyp['my_audio_01'] = ['hi', 'how', 'are', ...]

            word_ts_hyp (dict):
                Dictionary containing the start time and the end time of each word.
                Indexed by unique IDs.

                Example:
                >>> word_ts_hyp['my_audio_01'] = [[0.0, 0.04], [0.64, 0.68], [0.84, 0.88], ...]

        Returns:
            trans_info_dict (dict):
                Dictionary containing word timestamps, speaker labels and words from all sessions.
                Each session is indexed by a unique ID.
        """
        trans_info_dict = {}
        if self.fix_word_ts_with_VAD:
            if self.frame_VAD == {}:
                logging.info(
                    f"VAD timestamps are not provided. Fixing word timestamps without VAD. Please check the hydra configurations."
                )
            word_ts_refined = self._compensate_word_ts_list(self.audio_file_list, word_ts_hyp)
        else:
            word_ts_refined = word_ts_hyp

        if self.realigning_lm_params['arpa_language_model']:
            if not ARPA:
                raise ImportError(
                    'LM for realigning is provided but arpa is not installed. Install arpa using PyPI: pip install arpa'
                )
            else:
                self.realigning_lm = self._load_realigning_LM()

        for k, audio_file_path in enumerate(self.audio_file_list):
            uniq_id = get_uniqname_from_filepath(audio_file_path)
            word_dict_seq_list = self._get_word_dict_seq_list(
                uniq_id, diar_hyp, word_hyp, word_ts_hyp, word_ts_refined
            )
            if self.realigning_lm:
                word_dict_seq_list = self.realign_words_with_lm(word_dict_seq_list)
            trans_info_dict = self._make_json_output(uniq_id, diar_hyp, word_dict_seq_list, trans_info_dict)
        logging.info(f"Diarization with ASR output files are saved in: {self.root_path}/pred_rttms")
        return trans_info_dict

    def _get_word_dict_seq_list(
        self,
        uniq_id: str,
        diar_hyp: Dict[str, List[str]],
        word_hyp: Dict[str, List[str]],
        word_ts_hyp: Dict[str, List[float]],
        word_ts_refined: Dict[str, List[float]],
    ) -> Dict[str, Dict[str, str]]:
        """
        Save the hypothesis words and speaker labels to a dictionary variable for future use.

        Args:
            uniq_id (str):
                A unique ID (key) that identifies each input audio file.
            diar_hyp (dict):
                Dictionary of the Diarization output labels in str. Indexed by unique IDs.

                Example:
                >>>  diar_hyp['my_audio_01'] = ['0.0 4.375 speaker_1', '4.375 5.125 speaker_0', ...]

            word_hyp (dict):
                Dictionary of words from ASR inference. Indexed by unique IDs.

                Example:
                >>> word_hyp['my_audio_01'] = ['hi', 'how', 'are', ...]

            word_ts_hyp (dict):
                Dictionary containing the start time and the end time of each word.
                Indexed by unique IDs.

                Example:
                >>> word_ts_hyp['my_audio_01'] = [[0.0, 0.04], [0.64, 0.68], [0.84, 0.88], ...]
            
            word_ts_refined (dict):
                Dictionary containing the refined word timestamps based on hypothesis word timestamps.
                Indexed by unique IDs.

                Example:
                >>> word_ts_refined['my_audio_01'] = [[0.0, 0.60], [0.64, 0.80], [0.84, 0.92], ...]

        Returns:
            word_dict_seq_list (list):
                List containing word by word dictionary containing word, timestamps and speaker labels.

                Example:
                >>> [{'word': 'right', 'start_time': 0.0, 'end_time': 0.04, 'speaker_label': 'speaker_0'},  
                     {'word': 'and', 'start_time': 0.64, 'end_time': 0.68, 'speaker_label': 'speaker_1'},  
                     {'word': 'i', 'start_time': 0.84, 'end_time': 0.88, 'speaker_label': 'speaker_1'},  
                     ...]
        """
        words, labels = word_hyp[uniq_id], diar_hyp[uniq_id]
        start_point, end_point, speaker = labels[0].split()
        word_pos, idx = 0, 0
        word_dict_seq_list = []
        for j, word_ts_stt_end in enumerate(word_ts_hyp[uniq_id]):
            word_pos = self._get_word_timestamp_anchor(word_ts_stt_end)
            if word_pos > float(end_point):
                idx += 1
                idx = min(idx, len(labels) - 1)
                start_point, end_point, speaker = labels[idx].split()
            refined_word_ts_stt_end = word_ts_refined[uniq_id][j]
            stt_sec, end_sec = round(refined_word_ts_stt_end[0], 2), round(refined_word_ts_stt_end[1], 2)
            word_dict_seq_list.append(
                {'word': words[j], 'start_time': stt_sec, 'end_time': end_sec, 'speaker_label': speaker}
            )
        return word_dict_seq_list

    def _make_json_output(
        self,
        uniq_id: str,
        diar_hyp: Dict[str, List[str]],
        word_dict_seq_list: List[Dict[str, float]],
        trans_info_dict: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, str]]:
        """
        Generate json output files and transcripts from the ASR and diarization results.

        Args:
            uniq_id (str):
                A unique ID (key) that identifies each input audio file.
            diar_hyp (list):
                Dictionary containing the word sequence from ASR output.

                Example:
                >>>  diar_hyp['my_audio_01'] = ['0.0 4.375 speaker_1', '4.375 5.125 speaker_0', ...]

            word_dict_seq_list (list):
                List containing words and corresponding word timestamps in dictionary format.

                Example:
                >>> [{'word': 'right', 'start_time': 0.0, 'end_time': 0.04, 'speaker_label': 'speaker_0'},  
                     {'word': 'and', 'start_time': 0.64, 'end_time': 0.68, 'speaker_label': 'speaker_1'},  
                     {'word': 'i', 'start_time': 0.84, 'end_time': 0.88, 'speaker_label': 'speaker_1'},  
                     ...]

        Returns:
            trans_info_dict (dict):
                A dictionary containing overall results of diarization and ASR inference.
                `trans_info_dict` has following keys: `status`, `session_id`, `transcription`, `speaker_count`,
                `words`, `sentences`.

                Example:
                >>> trans_info_dict['my_audio_01'] = 
                    {
                        'status': 'Success',
                        'session_id': 'my_audio_01',
                        'transcription': 'right and i really think ...',
                        'speaker_count': 2,
                        'words': [{'word': 'right', 'start_time': 0.0, 'end_time': 0.04, 'speaker_label': 'speaker_0'},  
                                  {'word': 'and', 'start_time': 0.64, 'end_time': 0.68, 'speaker_label': 'speaker_1'},  
                                  {'word': 'i', 'start_time': 0.84, 'end_time': 0.88, 'speaker_label': 'speaker_1'},  
                                  ...
                                  ]
                        'sentences': [{'sentence': 'right',  'start_time': 0.0, 'end_time': 0.04, 'speaker_label': 'speaker_0'},
                                      {'sentence': 'and i really think ...', 
                                       'start_time': 0.92, 'end_time': 4.12, 'speaker_label': 'speaker_0'},
                                      ...
                                      ]
                    }
        """
        word_seq_list, audacity_label_words = [], []
        labels = diar_hyp[uniq_id]
        n_spk = self.get_num_of_spk_from_labels(labels)
        session_trans_dict = od(
            {
                'status': 'Success',
                'session_id': uniq_id,
                'transcription': '',
                'speaker_count': n_spk,
                'words': [],
                'sentences': [],
            }
        )
        gecko_dict = od({'schemaVersion': 2.0, 'monologues': []})
        start_point, end_point, speaker = labels[0].split()
        prev_speaker = speaker
        terms_list = []

        sentences = []
        sentence = {'speaker': speaker, 'start_point': float(start_point), 'end_point': float(end_point), 'text': ''}

        logging.info(f"Creating results for Session: {uniq_id} n_spk: {n_spk} ")
        for k, line_dict in enumerate(word_dict_seq_list):
            word, speaker = line_dict['word'], line_dict['speaker_label']
            word_seq_list.append(word)
            start_point, end_point = line_dict['start_time'], line_dict['end_time']
            if speaker != prev_speaker:
                if len(terms_list) != 0:
                    gecko_dict['monologues'].append(
                        {'speaker': {'name': None, 'id': prev_speaker}, 'terms': terms_list}
                    )
                    terms_list = []

                # remove trailing space in text
                sentence['text'] = sentence['text'].strip()

                # store last sentence
                sentences.append(sentence)

                # start construction of a new sentence
                sentence = {'speaker': speaker, 'start_point': start_point, 'end_point': end_point, 'text': ''}
            else:
                # correct the ending time
                sentence['end_point'] = end_point

            stt_sec, end_sec = start_point, end_point
            terms_list.append({'start': stt_sec, 'end': end_sec, 'text': word, 'type': 'WORD'})

            # add current word to sentence
            sentence['text'] += word.strip() + ' '

            self.add_json_to_dict(session_trans_dict, word, stt_sec, end_sec, speaker)
            audacity_label_words.append(self.get_audacity_label(word, stt_sec, end_sec, speaker))
            trans_info_dict[uniq_id] = session_trans_dict
            prev_speaker = speaker

        # note that we need to add the very last sentence.
        sentence['text'] = sentence['text'].strip()
        sentences.append(sentence)
        gecko_dict['monologues'].append({'speaker': {'name': None, 'id': speaker}, 'terms': terms_list})

        session_trans_dict['transcription'] = ' '.join(word_seq_list)
        self.write_and_log(uniq_id, session_trans_dict, audacity_label_words, gecko_dict, sentences)
        return trans_info_dict

    def _get_realignment_ranges(self, k: int, word_seq_len: int) -> Tuple[int, int]:
        """
        Calculate word ranges for realignment operation.
        N1, N2 are calculated to not exceed the start and end of the input word sequence.

        Args:
            k (int):
                Index of the current word
            word_seq_len (int):
                Length of the sentence

        Returns:
            N1 (int):
                Start index of the word sequence
            N2 (int):
                End index of the word sequence
        """
        if k < self.N_range[1]:
            N1 = max(k, self.N_range[0])
            N2 = min(word_seq_len - k, self.N_range[1])
        elif k > (word_seq_len - self.N_range[1]):
            N1 = min(k, self.N_range[1])
            N2 = max(word_seq_len - k, self.N_range[0])
        else:
            N1, N2 = self.N_range[1], self.N_range[1]
        return N1, N2

    def _get_word_timestamp_anchor(self, word_ts_stt_end: List[float]) -> float:
        """
        Determine a reference point to match a word with the diarization results.
        word_ts_anchor_pos determines the position of a word in relation to the given diarization labels:
            - 'start' uses the beginning of the word
            - 'end' uses the end of the word
            - 'mid' uses the mean of start and end of the word

        word_ts_anchor_offset determines how much offset we want to add to the anchor position.
        It is recommended to use the default value.

        Args:
            word_ts_stt_end (list):
                List containing start and end of the decoded word.

        Returns:
            word_pos (float):
                Floating point number that indicates temporal location of the word.
        """
        if self.params['word_ts_anchor_pos'] == 'start':
            word_pos = word_ts_stt_end[0]
        elif self.params['word_ts_anchor_pos'] == 'end':
            word_pos = word_ts_stt_end[1]
        elif self.params['word_ts_anchor_pos'] == 'mid':
            word_pos = (word_ts_stt_end[0] + word_ts_stt_end[1]) / 2
        else:
            logging.info(
                f"word_ts_anchor_pos: {self.params['word_ts_anchor']} is not a supported option. Using the default 'start' option."
            )
            word_pos = word_ts_stt_end[0]

        word_pos = word_pos + self.word_ts_anchor_offset
        return word_pos

    def realign_words_with_lm(self, word_dict_seq_list: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """
        Realign the mapping between speaker labels and words using a language model.
        The realigning process calculates the probability of the certain range around the words,
        especially at the boundary between two hypothetical sentences spoken by different speakers.

        Example:
            k-th word: "but"

            hyp_former:
                since i think like tuesday </s> <s>  but he's coming back to albuquerque
            hyp_latter:
                since i think like tuesday but </s> <s>  he's coming back to albuquerque

        The joint probabilities of words in the sentence are computed for these two hypotheses. In addition,
        logprob_diff_threshold parameter is used for reducing the false positive realigning.

        Args:
            word_dict_seq_list (list):
                List containing words and corresponding word timestamps in dictionary format.

        Returns:
            realigned_list (list):
                List of dictionaries containing words, word timestamps and speaker labels.
        """
        word_seq_len = len(word_dict_seq_list)
        hyp_w_dict_list, spk_list = [], []
        for k, line_dict in enumerate(word_dict_seq_list):
            word, spk_label = line_dict['word'], line_dict['speaker_label']
            hyp_w_dict_list.append(word)
            spk_list.append(spk_label)

        realigned_list = []
        org_spk_list = copy.deepcopy(spk_list)
        for k, line_dict in enumerate(word_dict_seq_list):
            if self.N_range[0] < k < (word_seq_len - self.N_range[0]) and (
                spk_list[k] != org_spk_list[k + 1] or spk_list[k] != org_spk_list[k - 1]
            ):
                N1, N2 = self._get_realignment_ranges(k, word_seq_len)
                hyp_former = self.realigning_lm.log_s(
                    ' '.join(hyp_w_dict_list[k - N1 : k] + self.stt_end_tokens + hyp_w_dict_list[k : k + N2])
                )
                hyp_latter = self.realigning_lm.log_s(
                    ' '.join(hyp_w_dict_list[k - N1 : k + 1] + self.stt_end_tokens + hyp_w_dict_list[k + 1 : k + N2])
                )
                log_p = [hyp_former, hyp_latter]
                p_order = np.argsort(log_p)[::-1]
                if log_p[p_order[0]] > log_p[p_order[1]] + self.realigning_lm_params['logprob_diff_threshold']:
                    if p_order[0] == 0:
                        spk_list[k] = org_spk_list[k + 1]
                line_dict['speaker_label'] = spk_list[k]
            realigned_list.append(line_dict)
        return realigned_list

    def calculate_cpWER(self, trans_info_dict: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate cpWER from the multispeaker ASR output. cpWER is calculated as following steps.

        trans_info_dict (dict):
            Dictionary containing overall results of diarization and ASR inference from all sessions.

        Returns:
            wer_results (dict):
                Session-by-session results including DER, miss rate, false alarm rate, WER and cpWER
        """
        word_seq_lists = []
        for audio_file_path in self.audio_file_list:
            uniq_id = get_uniqname_from_filepath(audio_file_path)
            word_seq_lists.append(trans_info_dict[uniq_id]['words'])
        return calculate_total_WER(word_seq_lists, self.ctm_file_list) if self.ctm_exists else {}

    def _get_str_speech_labels(self, speech_labels_float: List[List[float]]) -> List[str]:
        """
        Convert floating point speech labels list to a list containing string values.

        Args:
            speech_labels_float (list):
                List containing start and end timestamps of the speech segments in floating point type
            speech_labels (list):
                List containing start and end timestamps of the speech segments in string format
        """
        speech_labels = []
        for start, end in speech_labels_float:
            speech_labels.append("{:.3f} {:.3f} speech".format(start, end))
        return speech_labels

    def write_session_level_result_in_csv(
        self, der_results: Dict[str, Dict[str, float]], wer_results: Dict[str, Dict[str, float]]
    ):
        """
        This function is for development use when a CTM file is provided.
        Saves the session-level diarization and ASR result into a csv file.

        Args:
            wer_results (dict):
                Dictionary containing session-by-session results of ASR and diarization in terms of
                WER and cpWER.
        """
        target_path = f"{self.root_path}/pred_rttms/ctm_eval.csv"
        logging.info(f"Writing {target_path}")
        total_result_jsons = self.get_total_result_dict(der_results, wer_results)
        try:
            with open(target_path, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.csv_columns)
                writer.writeheader()
                for data in total_result_jsons:
                    writer.writerow(data)
        except IOError:
            logging.info("I/O error has occurred while writing a csv file.")

    def get_total_result_dict(
        self, der_results: Dict[str, Dict[str, float]], wer_results: Dict[str, Dict[str, float]]
    ):
        """
        Merge WER results and DER results into a single dictionary variable.

        Args:
            der_results (dict):
                Dictionary containing FA, MISS, CER and DER values for both aggregated amount and
                each session.
            wer_results (dict):
                Dictionary containing session-by-session WER and cpWER. `wer_results` only
                exists when CTM files are provided.

        Returns:
            total_result_dict (dict):
                Dictionary containing both DER and WER results. This dictionary contains unique-IDs of
                each session and `total` key that includes average (cp)WER and DER/CER/Miss/FA values.
        """
        total_result_dict = {}
        for uniq_id in der_results.keys():
            if uniq_id == 'total':
                continue
            total_result_dict[uniq_id] = {x: "-" for x in self.csv_columns}
            total_result_dict[uniq_id]["uniq_id"] = uniq_id
            if uniq_id in der_results:
                total_result_dict[uniq_id].update(der_results[uniq_id])
            if uniq_id in wer_results:
                total_result_dict[uniq_id].update(wer_results[uniq_id])
        total_result_jsons = list(total_result_dict.values())
        return total_result_jsons

    def _break_lines(self, string_out: str, max_chars_in_line: int = 90) -> str:
        """
        Break the lines in the transcript.

        Args:
            string_out (str):
                Input transcript with speaker labels
            max_chars_in_line (int):
                Maximum characters in each line

        Returns:
            return_string_out (str):
                String variable containing line breaking
        """
        color_str_len = len('\033[1;00m') if self.params['colored_text'] else 0
        split_string_out = string_out.split('\n')
        return_string_out = []
        for org_chunk in split_string_out:
            buffer = []
            if len(org_chunk) - color_str_len > max_chars_in_line:
                color_str = org_chunk[:color_str_len] if color_str_len > 0 else ''
                for i in range(color_str_len, len(org_chunk), max_chars_in_line):
                    trans_str = org_chunk[i : i + max_chars_in_line]
                    if len(trans_str.strip()) > 0:
                        c_trans_str = color_str + trans_str
                        buffer.append(c_trans_str)
                return_string_out.extend(buffer)
            else:
                return_string_out.append(org_chunk)
        return_string_out = '\n'.join(return_string_out)
        return return_string_out

    def write_and_log(
        self,
        uniq_id: str,
        session_trans_dict: Dict[str, Dict[str, float]],
        audacity_label_words: List[str],
        gecko_dict: Dict[str, Dict[str, float]],
        sentences: List[Dict[str, float]],
    ):
        """
        Write output files and display logging messages.

        Args:
            uniq_id (str):
                A unique ID (key) that identifies each input audio file
            session_trans_dict (dict):
                Dictionary containing the transcription output for a session
            audacity_label_words (list):
                List containing word and word timestamp information in Audacity label format
            gecko_dict (dict):
                Dictionary formatted to be opened in  Gecko software
            sentences (list):
                List containing sentence dictionary
        """
        # print the sentences in the .txt output
        string_out = self.print_sentences(sentences)
        if self.params['break_lines']:
            string_out = self._break_lines(string_out)

        # add sentences to the json array
        self.add_sentences_to_dict(session_trans_dict, sentences)

        dump_json_to_file(f'{self.root_path}/pred_rttms/{uniq_id}.json', session_trans_dict)
        dump_json_to_file(f'{self.root_path}/pred_rttms/{uniq_id}_gecko.json', gecko_dict)
        write_txt(f'{self.root_path}/pred_rttms/{uniq_id}.txt', string_out.strip())
        write_txt(f'{self.root_path}/pred_rttms/{uniq_id}.w.label', '\n'.join(audacity_label_words))

    def print_errors(self, der_results: Dict[str, Dict[str, float]], wer_results: Dict[str, Dict[str, float]]):
        """
        Print a slew of error metrics for ASR and Diarization.

        Args:
            der_results (dict):
                Dictionary containing FA, MISS, CER and DER values for both aggregated amount and
                each session.
            wer_results (dict):
                Dictionary containing session-by-session WER and cpWER. `wer_results` only
                exists when CTM files are provided.
        """
        DER_info = f"\nDER                : {der_results['total']['DER']:.4f} \
                     \nFA                 : {der_results['total']['FA']:.4f} \
                     \nMISS               : {der_results['total']['MISS']:.4f} \
                     \nCER                : {der_results['total']['CER']:.4f} \
                     \nSpk. counting acc. : {der_results['total']['spk_counting_acc']:.4f}"
        if self.ctm_exists:
            logging.info(
                DER_info
                + f"\ncpWER              : {wer_results['total']['average_cpWER']:.4f} \
                     \nWER                : {wer_results['total']['average_mixWER']:.4f}"
            )
        else:
            logging.info(DER_info)
        self.write_session_level_result_in_csv(der_results, wer_results)

    def print_sentences(self, sentences: List[Dict[str, float]]):
        """
        Print a transcript with speaker labels and timestamps.

        Args:
            sentences (list):
                List containing sentence-level dictionaries.

        Returns:
            string_out (str):
                String variable containing transcript and the corresponding speaker label.
        """
        # init output
        string_out = ''

        for sentence in sentences:
            # extract info
            speaker = sentence['speaker']
            start_point = sentence['start_point']
            end_point = sentence['end_point']
            text = sentence['text']

            if self.params['colored_text']:
                color = self.color_palette.get(speaker, '\033[0;37m')
            else:
                color = ''

            # cast timestamp to the correct format
            datetime_offset = 16 * 3600
            if float(start_point) > 3600:
                time_str = '%H:%M:%S.%f'
            else:
                time_str = '%M:%S.%f'
            start_point, end_point = max(float(start_point), 0), max(float(end_point), 0)
            start_point_str = datetime.fromtimestamp(start_point - datetime_offset).strftime(time_str)[:-4]
            end_point_str = datetime.fromtimestamp(end_point - datetime_offset).strftime(time_str)[:-4]

            if self.params['print_time']:
                time_str = f'[{start_point_str} - {end_point_str}] '
            else:
                time_str = ''

            # string out concatenation
            string_out += f'{color}{time_str}{speaker}: {text}\n'

        return string_out

    @staticmethod
    def get_audacity_label(word: str, stt_sec: float, end_sec: float, speaker: str) -> str:
        """
        Get a string formatted line for Audacity label.

        Args:
            word (str):
                A decoded word
            stt_sec (float):
                Start timestamp of the word
            end_sec (float):
                End timestamp of the word

        Returns:
            speaker (str):
                Speaker label in string type
        """
        spk = speaker.split('_')[-1]
        return f'{stt_sec}\t{end_sec}\t[{spk}] {word}'

    @staticmethod
    def get_num_of_spk_from_labels(labels: List[str]) -> int:
        """
        Count the number of speakers in a segment label list.
        Args:
            labels (list):
                List containing segment start and end timestamp and speaker labels.

                Example:
                >>> labels = ["15.25 21.82 speaker_0", "21.18 29.51 speaker_1", ... ]

        Returns:
            n_spk (int):
                The number of speakers in the list `labels`

        """
        spk_set = [x.split(' ')[-1].strip() for x in labels]
        return len(set(spk_set))

    @staticmethod
    def add_json_to_dict(session_trans_dict: Dict[str, List[str]], word: str, stt: float, end: float, speaker: str):
        """
        Add a dictionary variable containing timestamps and speaker label for a word to the final transcription
        information  dictionary.

        session_trans_dict (dict):
            Dictionary containing the transcription output for a session.
        word (str):
            A decoded word
        stt (float):
            Start time of the decoded word
        end (float):
            End time of the decoded word
        speaker (str):
            Speaker label of the decoded word
        """
        session_trans_dict['words'].append(
            {'word': word, 'start_time': stt, 'end_time': end, 'speaker_label': speaker}
        )

    @staticmethod
    def add_sentences_to_dict(session_trans_dict, sentences):
        """
        Add information in the `sentence` variable to the result dictionary.

        Args:
            session_trans_dict (dict):
                Dictionary containing the transcription output for a session.
            sentences (list):
                List containing dictionary variables containing word, word timestamps and speaker label.
        """
        # iterate over sentences
        for sentence in sentences:

            # save to session_trans_dict
            session_trans_dict['sentences'].append(
                {
                    'sentence': sentence['text'],
                    'start_time': sentence['start_point'],
                    'end_time': sentence['end_point'],
                    'speaker_label': sentence['speaker'],
                }
            )
