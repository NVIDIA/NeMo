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

import os, json
from dataclasses import dataclass, is_dataclass
from typing import Optional, Union, List, Tuple, Dict, Any

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf, open_dict, DictConfig

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

from copy import deepcopy
from nemo.collections.asr.parts.utils.diarization_utils import read_seglst, OnlineEvaluation
from nemo.utils import logging

from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel
from nemo.core.config import hydra_runner

from nemo.collections.asr.parts.utils.speaker_utils import (
audio_rttm_map as get_audio_rttm_map,
rttm_to_labels,
)
from examples.asr.asr_cache_aware_streaming.start_words import COMMON_SENTENCE_STARTS
from nemo.collections.asr.parts.utils.diarization_utils import (
print_sentences,
get_color_palette,
write_txt,
)

from copy import deepcopy
from nemo.collections.asr.parts.utils.diarization_utils import read_seglst, OnlineEvaluation
from nemo.utils import logging

from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel
from nemo.core.config import hydra_runner

from nemo.collections.asr.parts.utils.speaker_utils import (
audio_rttm_map as get_audio_rttm_map,
rttm_to_labels,
)
from examples.asr.asr_cache_aware_streaming.start_words import COMMON_SENTENCE_STARTS
from nemo.collections.asr.parts.utils.diarization_utils import (
print_sentences,
get_color_palette,
write_txt,
)

from typing import List, Optional
from dataclasses import dataclass
from collections import OrderedDict
import itertools

import time
from functools import wraps
import math

def setup_diarization_model(cfg: DictConfig, map_location: Optional[str] = None) -> SortformerEncLabelModel:
    """Setup model from cfg and return diarization model and model name for next step"""
    if cfg.diar_model_path.endswith(".ckpt"):
        diar_model = SortformerEncLabelModel.load_from_checkpoint(checkpoint_path=cfg.diar_model_path, 
                                                                  map_location=map_location, strict=False)
        model_name = os.path.splitext(os.path.basename(cfg.diar_model_path))[0]
    elif cfg.diar_model_path.endswith(".nemo"):
        diar_model = SortformerEncLabelModel.restore_from(restore_path=cfg.diar_model_path, 
                                                          map_location=map_location)
        model_name = os.path.splitext(os.path.basename(cfg.diar_model_path))[0]
    elif cfg.diar_pretrained_name.startswith("nvidia/"):
        diar_model = SortformerEncLabelModel.from_pretrained(cfg.diar_pretrained_name)
        model_name = os.path.splitext(os.path.basename(cfg.diar_pretrained_name))[0]
    else:
        raise ValueError("cfg.diar_model_path must end with.ckpt or.nemo!")
    return diar_model, model_name

def measure_eta(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Record the end time
        eta = end_time - start_time  # Calculate the elapsed time
        logging.info(f"[ ETA ] for '{func.__name__}': {eta:.4f} seconds")  # Print the ETA
        return result  # Return the original function's result
    return wrapper

def write_seglst(output_filepath, seglst_list):
    with open(output_filepath, "w") as f:
        f.write(json.dumps(seglst_list, indent=2) + "\n")

def extract_transcriptions(hyps):
    """
    The transcribed_texts returned by CTC and RNNT models are different.
    This method would extract and return the text section of the hypothesis.
    """
    if isinstance(hyps[0], Hypothesis):
        transcriptions = []
        for hyp in hyps:
            transcriptions.append(hyp.text)
    else:
        transcriptions = hyps
    return transcriptions


def calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded):
    # for the first step there is no need to drop any tokens after the downsampling as no caching is being used
    if step_num == 0 and not pad_and_drop_preencoded:
        return 0
    else:
        return asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
    
def fix_frame_time_step(
    cfg: Any, 
    new_tokens: List[str], 
    new_words: List[str], 
    frame_inds_seq: List[int]
    ) -> List[int]:
    """
    Adjust the frame indices sequence to match the length of new tokens.

    This function handles mismatches between the number of tokens and the frame indices sequence.
    It adjusts the frame_inds_seq to ensure it has the same length as new_tokens.

    Args:
        cfg (Any): Configuration object containing logging settings.
        new_tokens (List[str]): List of new tokens.
        new_words (List[str]): List of new words.
        frame_inds_seq (List[int]): List of frame indices.

    Returns:
        List[int]: Adjusted frame indices sequence.
    """    
    if len(new_tokens) != len(frame_inds_seq):
        # Sometimes there is a mismatch in the number of tokens between the new tokens and the frame indices sequence.
        if len(frame_inds_seq) > len(new_words):
            # Get unique frame indices sequence
            frame_inds_seq = list(OrderedDict.fromkeys(frame_inds_seq))
            if len(frame_inds_seq) < len(new_tokens):
                deficit = len(new_tokens) - len(frame_inds_seq)
                frame_inds_seq = [frame_inds_seq[0]] * deficit + frame_inds_seq
            elif len(frame_inds_seq) > len(new_tokens):
                deficit = len(frame_inds_seq) - len(new_tokens)
                frame_inds_seq = frame_inds_seq[deficit:]
                
        elif len(frame_inds_seq) < len(new_tokens):
            deficit = len(new_tokens) - len(frame_inds_seq)
            frame_inds_seq = [frame_inds_seq[0]] * deficit + frame_inds_seq
        if cfg.log:
            logging.warning(
                f"Length of new token sequence ({len(new_tokens)}) does not match" 
                f"the length of frame indices sequence ({len(frame_inds_seq)}). Skipping this chunk."
            )
    return frame_inds_seq

def get_simulated_softmax(cfg, speaker_sigmoid: torch.Tensor) -> torch.Tensor:
    """Simulate the softmax operation for speaker diarization."""
    speaker_sigmoid = torch.clamp(speaker_sigmoid, min=cfg.min_sigmoid_val, max=1) 
    speaker_softmax = speaker_sigmoid / speaker_sigmoid.sum()
    speaker_softmax = speaker_softmax.cpu()
    speaker_softmax[cfg.limit_max_spks:] = 0.0 
    return speaker_softmax

def get_speaker_turn_flags(word: str, stt_words: List[str]) -> Tuple[bool, bool]:
    if len(word) == 0:
        is_stt_speaker_turn, is_end_speaker_turn = False, False
    if (word[0].isupper() and word in stt_words) and (("." in word) or ("?" in word)):
        return True, True
    else:
        is_end_speaker_turn = ("." in word) or ("?" in word)
        is_stt_speaker_turn = word[0].isupper() and (word in stt_words) or ("," in word)
    return is_stt_speaker_turn, is_end_speaker_turn

def get_word_dict_content_offline(
    cfg: Any,
    word: str,
    word_index: int,
    diar_pred_out: torch.Tensor,
    time_stt_end_tuple: Tuple[int],
    stt_words: List[str],
    frame_len: float = 0.08
) -> Dict[str, Any]:
    """
    Generate a dictionary containing word information and speaker diarization results.

    This function processes a single word and its associated tokens to determine
    the start and end frames, speaker, and other relevant information.

    Args:
        cfg (Any): Configuration object containing diarization settings.
        word (str): The word being processed.
        word_index (int): Index of the word in the sequence.
        diar_pred_out (torch.Tensor): Diarization prediction output stream.
        time_stt_end_tuple (int): Local time step offset.
        frame_len (float, optional): Length of each frame in seconds. Defaults to 0.08.

    Returns:
        Dict[str, Any]: A dictionary containing word information and diarization results.
    """    
    frame_stt, frame_end = time_stt_end_tuple
        
    # Edge Cases: Sometimes, repeated token indexs can lead to incorrect frame and speaker assignment.
    if frame_stt == frame_end:
        if frame_stt >= diar_pred_out.shape[0] - 1:
            frame_stt, frame_end = (diar_pred_out.shape[1] - 1, diar_pred_out.shape[0])
        else:
            frame_end = frame_stt + 1
    
    # Get the speaker based on the frame-wise softmax probabilities.
    stt_p, end_p = max((frame_stt + cfg.left_frame_shift), 0), (frame_end + cfg.right_frame_shift)
    speaker_sigmoid = diar_pred_out[stt_p:end_p, :].mean(dim=0)
    speaker_softmax = get_simulated_softmax(cfg, speaker_sigmoid)

    speaker_softmax[cfg.limit_max_spks:] = 0.0
    spk_id = speaker_softmax.argmax().item()
    stt_sec, end_sec = frame_stt * frame_len, frame_end * frame_len
    is_stt_speaker_turn, is_end_speaker_turn = get_speaker_turn_flags(word, stt_words=stt_words)
    word_dict = {"word": word,
                 "word_index": word_index,
                'frame_stt': frame_stt,
                'frame_end': frame_end,
                'start_time': round(stt_sec, 3), 
                'end_time': round(end_sec, 3), 
                'speaker': f"speaker_{spk_id}",
                'is_stt_speaker_turn': is_stt_speaker_turn,
                'is_end_speaker_turn': is_end_speaker_turn,
                'speaker_softmax': speaker_softmax} 
    return word_dict

def get_word_dict_content_online(
    cfg: Any,
    word: str,
    word_index: int,
    diar_pred_out_stream: torch.Tensor,
    token_group: List[str],
    frame_inds_seq: List[int],
    time_step_local_offset: int,
    stt_words: List[str],
    frame_len: float = 0.08
) -> Dict[str, Any]:
    """
    Generate a dictionary containing word information and speaker diarization results.

    This function processes a single word and its associated tokens to determine
    the start and end frames, speaker, and other relevant information.

    Args:
        cfg (Any): Configuration object containing diarization settings.
        word (str): The word being processed.
        word_index (int): Index of the word in the sequence.
        diar_pred_out_stream (torch.Tensor): Diarization prediction output stream.
        token_group (List[str]): Group of tokens associated with the word.
        frame_inds_seq (List[int]): Sequence of frame indices.
        time_step_local_offset (int): Local time step offset.
        frame_len (float, optional): Length of each frame in seconds. Defaults to 0.08.

    Returns:
        Dict[str, Any]: A dictionary containing word information and diarization results.
    """    
    _stt, _end = time_step_local_offset, time_step_local_offset + len(token_group) - 1
    if len(token_group) == 1:
        frame_stt, frame_end = frame_inds_seq[_stt], frame_inds_seq[_stt] + 1
    else:
        frame_stt, frame_end = frame_inds_seq[_stt], frame_inds_seq[_end]
        
    # Edge Cases: Sometimes, repeated token indexs can lead to incorrect frame and speaker assignment.
    if frame_stt == frame_end:
        if frame_stt >= diar_pred_out_stream.shape[0] - 1:
            frame_stt, frame_end = (diar_pred_out_stream.shape[1] - 1, diar_pred_out_stream.shape[0])
        else:
            frame_end = frame_stt + 1
    
    # Get the speaker based on the frame-wise softmax probabilities.
    stt_p, end_p = max((frame_stt + cfg.left_frame_shift), 0), (frame_end + cfg.right_frame_shift)
    speaker_sigmoid = diar_pred_out_stream[stt_p:end_p, :].mean(dim=0)
    speaker_softmax = get_simulated_softmax(cfg, speaker_sigmoid)

    speaker_softmax[cfg.limit_max_spks:] = 0.0
    spk_id = speaker_softmax.argmax().item()
    stt_sec, end_sec = frame_stt * frame_len, frame_end * frame_len
    is_stt_speaker_turn, is_end_speaker_turn = get_speaker_turn_flags(word, stt_words=stt_words)
    word_dict = {"word": word,
                 "word_index": word_index,
                'frame_stt': frame_stt,
                'frame_end': frame_end,
                'start_time': round(stt_sec, 3), 
                'end_time': round(end_sec, 3), 
                'speaker': f"speaker_{spk_id}",
                'is_stt_speaker_turn': is_stt_speaker_turn,
                'is_end_speaker_turn': is_end_speaker_turn,
                'speaker_softmax': speaker_softmax} 
    return word_dict
    
def get_multitoken_words(
    cfg, 
    word_and_ts_seq: Dict[str, List],
    word_seq: List[str],
    new_words: List[str],
    fix_prev_words_count: int = 5
) -> Dict[str, List]:
    """
    Fix multi-token words that were not fully captured by the previous chunk window.

    This function compares the words in the current sequence with the previously processed words,
    and updates any multi-token words that may have been truncated in earlier processing.

    Args:
        cfg (DiarizationConfig): Configuration object containing verbose setting.
        word_and_ts_seq (Dict[str, List]): Dictionary containing word sequences and timestamps.
        word_seq (List[str]): List of all words processed so far.
        new_words (List[str]): List of new words in the current chunk.
        fix_prev_words_count (int, optional): Number of previous words to check. Defaults to 5.

    Returns:
        Dict[str, List]: Updated word_and_ts_seq with fixed multi-token words.
    """
    prev_start = max(0, len(word_seq) - fix_prev_words_count - len(new_words))
    prev_end = max(0, len(word_seq) - len(new_words))
    
    for ct, prev_word in enumerate(word_seq[prev_start:prev_end]):
        if len(word_and_ts_seq["words"]) > fix_prev_words_count - ct:
            saved_word = word_and_ts_seq["words"][-fix_prev_words_count + ct]["word"]
            if len(prev_word) > len(saved_word):
                if cfg.verbose:
                    logging.info(f"[Replacing Multi-token Word]: {saved_word} with {prev_word}")
                word_and_ts_seq["words"][-fix_prev_words_count + ct]["word"] = prev_word
    
    return word_and_ts_seq
  
def correct_speaker_assignments(word_and_ts_seq: dict, sentence_render_length: int = None) -> dict:
    """ 
    Correct speaker assignments based on the punctuations and capitalization in the sequence.
    
    Args:
        word_and_ts_seq (dict): Dictionary containing word and time-related information.
        sentence_render_length (int): Number of previous words to consider for speaker assignment.
    
    Returns:
        word_and_ts_seq (dict): Corrected word and time-related information. 
    
    Note:
        This method assumes that PnC equipped ASR is used for speech recognition.
        This method assumes that the speaker assignments are correct for the first and last words in the sequence.
        It modifies the speaker assignments based on the speaker changes in the middle of the sequence.
    """
    WL = len(word_and_ts_seq["words"])

    if sentence_render_length is not None: 
        if WL-sentence_render_length < 1 or WL-1 < 2 or sentence_render_length <= 1:
            return word_and_ts_seq
    elif sentence_render_length is None:
        sentence_render_length = WL 
    
    for idx in range(WL - sentence_render_length, WL-1):
        word_dict = word_and_ts_seq['words'][idx]
        # Correct a starting word attached to the previous speaker turn. 
        if word_dict['is_stt_speaker_turn'] and not word_dict['is_end_speaker_turn']:
            if word_and_ts_seq['words'][idx+1]['speaker'] != word_dict['speaker']:
                word_and_ts_seq['words'][idx]['speaker'] = word_and_ts_seq['words'][idx+1]['speaker']
                
        # Correct a ending word attached to the previous speaker turn. 
        if idx in range(max(1, WL - sentence_render_length), max(2, WL-1)) \
        and word_dict['is_end_speaker_turn'] and not word_dict['is_stt_speaker_turn']:
            if word_and_ts_seq['words'][idx-1]['speaker'] != word_dict['speaker']:
                word_and_ts_seq['words'][idx]['speaker'] = word_and_ts_seq['words'][idx-1]['speaker']
    
        # Correct a middle word wrongly assigned to the other speaker, which is lower case and no punctuation.
        if (not word_dict['is_stt_speaker_turn'] and \
            not word_dict['is_end_speaker_turn'] and \
            not word_dict['word'][0].isupper()):
            # Check whether the current word is isolated in terms of speaker assignment.
            if word_and_ts_seq['words'][idx-1]['speaker'] != word_dict['speaker'] and \
                word_and_ts_seq['words'][idx+1]['speaker'] != word_dict['speaker'] and \
                word_and_ts_seq['words'][idx-1]['speaker'] == \
                word_and_ts_seq['words'][idx+1]['speaker']:
                word_and_ts_seq['words'][idx]['speaker'] = word_and_ts_seq['words'][idx-1]['speaker'] 
                
    return word_and_ts_seq 
   
def append_word_and_ts_seq(
    cfg: Any, 
    word_idx_offset: int, 
    word_and_ts_seq: Dict[str, Any], 
    word_dict: Dict[str, Any]
) -> tuple[int, Dict[str, Any]]:
    """
    Append the word dictionary to the word and time-stamp sequence.

    This function updates the word_and_ts_seq dictionary by appending new word information
    and managing the buffered words and speaker count.

    Args:
        cfg (Any): Configuration object containing parameters like word_window.
        word_idx_offset (int): The current word index offset.
        word_and_ts_seq (Dict[str, Any]): Dictionary containing word sequences and related information.
        word_dict (Dict[str, Any]): Dictionary containing information about the current word.

    Returns:
        tuple[int, Dict[str, Any]]: A tuple containing the updated word_idx_offset and word_and_ts_seq.
    """
    word_and_ts_seq["words"].append(word_dict)
    word_and_ts_seq["buffered_words"].append(word_dict)
    word_and_ts_seq["speaker_count_buffer"].append(word_dict["speaker"])
    word_and_ts_seq["word_window_seq"].append(word_dict['word'])
    
    if len(word_and_ts_seq["words"]) >= cfg.word_window + 1: 
        word_and_ts_seq["buffered_words"].pop(0)
        word_and_ts_seq["word_window_seq"].pop(0)
        word_idx_offset = 0
    
    word_and_ts_seq["speaker_count"] = len(set(word_and_ts_seq["speaker_count_buffer"]))
    return word_idx_offset, word_and_ts_seq

class SpeakerTaggedASR:
    def __init__(
        self,
        cfg,
        asr_model,
        diar_model,
    ):
        # Required configs, models and datasets for inference
        self.cfg = cfg
        self.test_manifest_dict = get_audio_rttm_map(self.cfg.manifest_file)
        self.asr_model = asr_model
        self.diar_model = diar_model
        
        # ASR speaker tagging configs
        self.fix_speaker_assignments = cfg.fix_speaker_assignments
        self._fix_prev_words_count = cfg.fix_prev_words_count
        self._sentence_render_length = int(self._fix_prev_words_count + cfg.update_prev_words_sentence)
        self._frame_len_sec = 0.08
        self._initial_steps = cfg.ignored_initial_frame_steps
        self._all_sentences = []
        self._stt_words = COMMON_SENTENCE_STARTS
        self._init_evaluator() 
        self._frame_hop_length = self.asr_model.encoder.streaming_cfg.valid_out_len
    
    def _init_evaluator(self):  
        self.online_evaluators, self._word_and_ts_seq, self._sentence_and_ts_seq = [], [], []
        for idx, (uniq_id, data_dict) in enumerate(self.test_manifest_dict.items()):
            self._word_and_ts_seq.append({"words": [],
                                    "buffered_words": [],
                                    "token_frame_index": [], 
                                    "offset_count": 0,
                                    "status": "success", 
                                    "sentences": None, 
                                    "last_word_index": 0,
                                    "speaker_count": None,
                                    "transcription": None,
                                    "max_spk_probs": [],
                                    "word_window_seq": [],
                                    "speaker_count_buffer": [],
                                    "sentence_memory": {},
                                    })
            self._sentence_and_ts_seq.append({"words": [],
                        "buffered_words": [],
                        "token_frame_index": [], 
                        "offset_count": 0,
                        "status": "success", 
                        "sentences": None, 
                        "last_word_index": 0,
                        "speaker_count": None,
                        "transcription": None,
                        "max_spk_probs": [],
                        "word_window_seq": [],
                        "speaker_count_buffer": [],
                        "sentence_memory": {},
                        }) 
            if data_dict['seglst_filepath'] is not None:
                ref_seglst = read_seglst(data_dict['seglst_filepath'])
            else:
                ref_seglst = None
            
            if data_dict['rttm_filepath'] is not None:
                ref_rttm_labels = rttm_to_labels(data_dict['rttm_filepath'])
            else:
                ref_rttm_labels = None
            
            eval_instance = OnlineEvaluation(ref_seglst=ref_seglst,
                                             ref_rttm_labels=ref_rttm_labels, 
                                             hyp_seglst=None,
                                             collar=0.25, 
                                             ignore_overlap=False, 
                                             verbose=True)
            self.online_evaluators.append(eval_instance)
       
    def _get_offset_sentence(self, session_trans_dict, offset):
        """
        For the very first word in a session, get the offset sentence.
        
        Args:
            session_trans_dict (dict): Dictionary containing session-related information.
            offset (int): Index of the word for which the offset sentence is needed.
        
        Returns:
            (Dict): Dictionary containing offset sentence information.
        """
        word_dict = session_trans_dict['words'][offset]
        return {'session_id': session_trans_dict['uniq_id'], 
                'speaker': word_dict['speaker'],
                'start_time': word_dict['start_time'], 
                'end_time': word_dict['end_time'], 
                'text': f"{word_dict['word']} ", 
                'words': f"{word_dict['word']} "} 
        
    def _get_sentence(self, word_dict):
        """ 
        Get the sentence for a given word.
        
        Args:
            word_dict: Dictionary containing word-related information.
        """
        return {'speaker': word_dict['speaker'], 
                'start_time':  word_dict['start_time'], 
                'end_time': word_dict['end_time'], 
                'text': ''}
        
    def text_post_processing(self, sentence):
        sentence['text'] =  sentence['text'].replace("twenty twenty twenty twenty twenty twenty twenty twenty twenty twenty twenty twenty twenty twenty twenty twenty twenty twenty twenty twenty twenty three", "twenty twenty three")
        if self.cfg.uppercase_first_letter and len(sentence['text']) > 1:
            sentence['text'] = sentence['text'][:1].upper() + sentence['text'][1:]
        if self.cfg.remove_pnc:
            sentence['text'] = sentence['text'].lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '').upper()
        return sentence
    
    
    def get_sentences_values(self, session_trans_dict: dict, sentence_render_length: int):
        """ 
        Get sentences (speaker-turn-level text) for a given session and sentence render length.
        
        Args:
            session_trans_dict: Dictionary containing session-related information.
            sentence_render_length: Length of the sentences to be generated.
        Returns:
            sentences: List of sentences in the session.
        """
        stt_word_index = max(0, session_trans_dict['last_word_index'] - sentence_render_length)
        if session_trans_dict['sentences'] is None:
            sentence = self._get_offset_sentence(session_trans_dict=session_trans_dict, offset=0)
            sentences = []
            session_trans_dict['last_word_index'] = stt_word_index
            session_trans_dict['sentence_memory'].update({stt_word_index: 
                                                            (deepcopy(sentences), 
                                                             deepcopy(sentence), 
                                                             sentence['speaker']
                                                            )})
            prev_speaker = session_trans_dict['words'][stt_word_index]['speaker']
        else:
            (_sentences, _sentence, prev_speaker) = session_trans_dict['sentence_memory'][stt_word_index]
            sentences, sentence = deepcopy(_sentences), deepcopy(_sentence)
        
        for k in range(stt_word_index + 1, len(session_trans_dict['words'])):
            word_dict = session_trans_dict['words'][k]
            word, end_point = word_dict['word'], word_dict['end_time']
            # if word_dict['speaker'] != prev_speaker:
            if word_dict['speaker'] != prev_speaker or (word_dict['start_time'] - session_trans_dict['words'][k-1]['end_time']) > self.cfg.sentence_break_threshold_in_sec:
                sentence['text'] = sentence['text'].strip()
                sentence = self.text_post_processing(sentence=sentence)
                sentences.append(sentence)
                sentence = self._get_sentence(word_dict=session_trans_dict['words'][k])
            else:
                sentence['end_time'] = end_point
            sentence['text'] += word.strip() + ' '
            sentence = self.text_post_processing(sentence=sentence)
            sentence['words'] = sentence['text']
            sentence['session_id'] = session_trans_dict['uniq_id']
            session_trans_dict['last_word_index'] = k
            prev_speaker = word_dict['speaker']
            session_trans_dict['sentence_memory'][k] = (deepcopy(sentences), deepcopy(sentence), prev_speaker)
        sentence['text'] = sentence['text'].strip()
        sentence = self.text_post_processing(sentence=sentence)
        sentences.append(sentence)
        session_trans_dict['sentences'] = sentences
        return session_trans_dict 
    
    def merge_transcript_and_speakers(self, test_manifest_dict, asr_hypotheses, diar_pred_out):
        transcribed_speaker_texts = [None] * len(test_manifest_dict)
        
        for idx, (uniq_id, data_dict) in enumerate(test_manifest_dict.items()):
            if not len( asr_hypotheses[idx].text) == 0:
                # Get the word-level dictionaries for each word in the chunk
                #  diar_pred_out_stream=diar_pred_out_stream[idx, :, :],
                                                                            
                self._word_and_ts_seq[idx] = self.get_frame_and_words_offline(uniq_id=uniq_id,
                                                                            diar_pred_out=diar_pred_out[idx].squeeze(0),
                                                                            asr_hypothesis=asr_hypotheses[idx],
                                                                            word_and_ts_seq=self._word_and_ts_seq[idx], 
                                                                            )
                if len(self._word_and_ts_seq[idx]["words"]) > 0:
                    self._word_and_ts_seq[idx] = self.get_sentences_values(session_trans_dict=self._word_and_ts_seq[idx], 
                                                                            sentence_render_length=self._sentence_render_length)
                    if self.cfg.generate_scripts:
                        transcribed_speaker_texts[idx] = \
                            print_sentences(sentences=self._word_and_ts_seq[idx]["sentences"], 
                            color_palette=get_color_palette(), 
                            params=self.cfg)
                        write_txt(f'{self.cfg.print_path}'.replace(".sh", f"_{idx}.sh"), 
                                    transcribed_speaker_texts[idx].strip()) 
        return transcribed_speaker_texts, self._word_and_ts_seq
     
    def get_frame_and_words_offline(
        self,
        uniq_id, 
        diar_pred_out, 
        asr_hypothesis,
        word_and_ts_seq,
    ):        
        word_and_ts_seq['uniq_id'] = uniq_id

        for word_index, hyp_word_dict in enumerate(asr_hypothesis.timestep['word']):
            time_stt_end_tuple=(hyp_word_dict['start_offset'], hyp_word_dict['end_offset'])
            word_dict = get_word_dict_content_offline(cfg=self.cfg, 
                                                        word=hyp_word_dict['word'],
                                                        word_index=word_index,
                                                        diar_pred_out=diar_pred_out,
                                                        time_stt_end_tuple=time_stt_end_tuple,
                                                        stt_words=self._stt_words,
                                                        frame_len=self._frame_len_sec
                                                        )
            word_and_ts_seq["words"].append(word_dict)
            word_and_ts_seq["speaker_count_buffer"].append(word_dict["speaker"])
            word_and_ts_seq["word_window_seq"].append(word_dict['word'])
            
        word_and_ts_seq["buffered_words"] = word_and_ts_seq["words"] 
        word_and_ts_seq["speaker_count"] = len(set(word_and_ts_seq["speaker_count_buffer"]))
        if self.cfg.fix_speaker_assignments:     
            word_and_ts_seq = correct_speaker_assignments(word_and_ts_seq=word_and_ts_seq) 
        return word_and_ts_seq
    
    def get_frame_and_words_online(
        self, 
        uniq_id, 
        step_num, 
        diar_pred_out_stream, 
        previous_hypothesis, 
        word_and_ts_seq, 
    ):        
        offset = step_num * self._frame_hop_length
        word_seq = previous_hypothesis.text.split()
        new_words = word_seq[word_and_ts_seq["offset_count"]:]
        new_token_group = self.asr_model.tokenizer.text_to_tokens(new_words)
        new_tokens = list(itertools.chain(*new_token_group))
        frame_inds_seq = (torch.tensor(previous_hypothesis.timestep) + offset).tolist()
        frame_inds_seq = fix_frame_time_step(self.cfg, new_tokens, new_words, frame_inds_seq)
        min_len = min(len(new_words), len(frame_inds_seq))
        word_and_ts_seq['uniq_id'] = uniq_id

        for idx in range(min_len):
            word_and_ts_seq["token_frame_index"].append((new_tokens[idx], frame_inds_seq[idx]))
            word_and_ts_seq["offset_count"] += 1
        
        time_step_local_offset, word_idx_offset = 0, 0
        word_count_offset = len(word_and_ts_seq["words"]) 
        word_and_ts_seq = get_multitoken_words(cfg=self.cfg, 
                                               word_and_ts_seq=word_and_ts_seq, 
                                               word_seq=word_seq, 
                                               new_words=new_words, 
                                               fix_prev_words_count=self._fix_prev_words_count
                                            )
        
        # Get the FIFO queue preds to word_and_ts_seq 
        local_idx = 0
        for local_idx, (token_group, word) in enumerate(zip(new_token_group, new_words)):
            word_dict = get_word_dict_content_online(cfg=self.cfg, 
                                                    word=word,
                                                    word_index= (word_count_offset + local_idx),
                                                    diar_pred_out_stream=diar_pred_out_stream,
                                                    token_group=token_group,
                                                    frame_inds_seq=frame_inds_seq,
                                                    time_step_local_offset=time_step_local_offset,
                                                    stt_words=self._stt_words,
                                                    frame_len=self._frame_len_sec
                                                    )
            # Count the number of speakers in the word window
            time_step_local_offset += len(token_group)
            word_idx_offset, word_and_ts_seq = append_word_and_ts_seq(cfg=self.cfg, 
                                                                      word_idx_offset=word_idx_offset, 
                                                                      word_and_ts_seq=word_and_ts_seq, 
                                                                      word_dict=word_dict)
        
            if self.cfg.fix_speaker_assignments:     
                word_and_ts_seq = correct_speaker_assignments(word_and_ts_seq=word_and_ts_seq, 
                                                                sentence_render_length=self._sentence_render_length)
        return word_and_ts_seq

    @measure_eta 
    def perform_streaming_stt_spk(
        self,
        step_num,
        chunk_audio,
        chunk_lengths,
        cache_last_channel,
        cache_last_time,
        cache_last_channel_len,
        previous_hypotheses,
        asr_pred_out_stream,
        diar_pred_out_stream,
        mem_last_time,
        fifo_last_time,
        left_offset,
        right_offset,
        is_buffer_empty,
        pad_and_drop_preencoded,
    ):

        (
            asr_pred_out_stream,
            transcribed_texts,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
            previous_hypotheses,
        ) = self.asr_model.conformer_stream_step(
            processed_signal=chunk_audio,
            processed_signal_length=chunk_lengths,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=is_buffer_empty,
            previous_hypotheses=previous_hypotheses,
            previous_pred_out=asr_pred_out_stream,
            drop_extra_pre_encoded=calc_drop_extra_pre_encoded(
                self.asr_model, step_num, pad_and_drop_preencoded
            ),
            return_transcription=True,
        )

        if step_num > 0:
            left_offset = 8
            chunk_audio = chunk_audio[..., 1:]
            chunk_lengths -= 1
        (
            mem_last_time,
            fifo_last_time,
            mem_preds,
            fifo_preds,
            diar_pred_out_stream
        ) = self.diar_model.forward_streaming_step(
            processed_signal=chunk_audio.transpose(1, 2),
            processed_signal_length=chunk_lengths,
            mem_last_time=mem_last_time,
            fifo_last_time=fifo_last_time,
            previous_pred_out=diar_pred_out_stream,
            left_offset=left_offset,
            right_offset=right_offset,
        )
        transcribed_speaker_texts = [None] * len(self.test_manifest_dict)
        for idx, (uniq_id, data_dict) in enumerate(self.test_manifest_dict.items()): 
            if not (len( previous_hypotheses[idx].text) == 0 and step_num <= self._initial_steps):
                # Get the word-level dictionaries for each word in the chunk
                self._word_and_ts_seq[idx] = self.get_frame_and_words_online(uniq_id=uniq_id,
                                                                            step_num=step_num, 
                                                                            diar_pred_out_stream=diar_pred_out_stream[idx, :, :],
                                                                            previous_hypothesis=previous_hypotheses[idx], 
                                                                            word_and_ts_seq=self._word_and_ts_seq[idx],
                                                                            )
                if len(self._word_and_ts_seq[idx]["words"]) > 0:
                    self._word_and_ts_seq[idx] = self.get_sentences_values(session_trans_dict=self._word_and_ts_seq[idx], 
                                                                           sentence_render_length=self._sentence_render_length)
                    if self.cfg.eval_mode:
                        der, cpwer, is_update = self.online_evaluators[idx].evaluate_inloop(hyp_seglst=self._word_and_ts_seq[idx]["sentences"], 
                                                                                            end_step_time=self._word_and_ts_seq[idx]["sentences"][-1]["end_time"])
                    if self.cfg.generate_scripts:
                        transcribed_speaker_texts[idx] = \
                            print_sentences(sentences=self._word_and_ts_seq[idx]["sentences"], 
                            color_palette=get_color_palette(), 
                            params=self.cfg)
                        write_txt(f'{self.cfg.print_path}'.replace(".sh", f"_{idx}.sh"), 
                                  transcribed_speaker_texts[idx].strip())
            
        return (transcribed_speaker_texts,
                transcribed_texts,
                asr_pred_out_stream,
                transcribed_texts,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
                previous_hypotheses,
                mem_last_time,
                fifo_last_time,
                diar_pred_out_stream)
    
    @measure_eta 
    def perform_queryless_streaming_stt_spk(
        self,
        step_num,
        chunk_audio,
        chunk_lengths,
        cache_last_channel,
        cache_last_time,
        cache_last_channel_len,
        previous_hypotheses,
        asr_pred_out_stream,
        diar_pred_out_stream,
        mem_last_time,
        fifo_last_time,
        left_offset,
        right_offset,
        is_buffer_empty,
        pad_and_drop_preencoded,
    ):
        
        if step_num > 0:
            left_offset = 8
            chunk_audio = chunk_audio[..., 1:]
            chunk_lengths -= 1
        (
            mem_last_time,
            fifo_last_time,
            mem_preds,
            fifo_preds,
            diar_pred_out_stream
        ) = self.diar_model.forward_streaming_step(
            processed_signal=chunk_audio.transpose(1, 2),
            processed_signal_length=chunk_lengths,
            mem_last_time=mem_last_time,
            fifo_last_time=fifo_last_time,
            previous_pred_out=diar_pred_out_stream,
            left_offset=left_offset,
            right_offset=right_offset,
        )

        spk_targets = diar_pred_out_stream[:, -14:] > 0.5
        (
            asr_pred_out_stream,
            transcribed_texts,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
            previous_hypotheses,
        ) = self.asr_model.conformer_stream_step(
            processed_signal=chunk_audio,
            processed_signal_length=chunk_lengths,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=is_buffer_empty,
            previous_hypotheses=previous_hypotheses,
            previous_pred_out=asr_pred_out_stream,
            drop_extra_pre_encoded=calc_drop_extra_pre_encoded(
                self.asr_model, step_num, pad_and_drop_preencoded
            ),
            return_transcription=True,
            spk_targets=spk_targets
        )

        n_spk = spk_targets.shape[-1]
        transcribed_speaker_texts = [None] * n_spk
        uniq_id = list(self.test_manifest_dict.keys())[0]
        if len(self._word_and_ts_seq) < n_spk:
            self._word_and_ts_seq = [deepcopy(self._word_and_ts_seq[0]) for _ in range(n_spk)]

        # # step 1: save the word and time-stamp sequence for each speaker
        for speaker_index in range(n_spk): 
            if not (len( previous_hypotheses[speaker_index].text) == 0 and step_num <= self._initial_steps):
                # Get the word-level dictionaries for each word in the chunk
                diar_pred_out_stream_idx = torch.zeros_like(diar_pred_out_stream)
                diar_pred_out_stream_idx[:, :, speaker_index] = diar_pred_out_stream[:, :, speaker_index]
                self._word_and_ts_seq[speaker_index] = self.get_frame_and_words_online(uniq_id=uniq_id,
                                                                            step_num=step_num, 
                                                                            diar_pred_out_stream=diar_pred_out_stream_idx[0],
                                                                            previous_hypothesis=previous_hypotheses[speaker_index], 
                                                                            word_and_ts_seq=self._word_and_ts_seq[speaker_index],
                                                                            )
                if len(self._word_and_ts_seq[speaker_index]["words"]) > 0:
                    self._word_and_ts_seq[speaker_index] = self.get_sentences_values(session_trans_dict=self._word_and_ts_seq[speaker_index], 
                                                                           sentence_render_length=self._sentence_render_length)
                    if self.cfg.eval_mode:
                        der, cpwer, is_update = self.online_evaluators[speaker_index].evaluate_inloop(hyp_seglst=self._word_and_ts_seq[speaker_index]["sentences"], 
                                                                                            end_step_time=self._word_and_ts_seq[speaker_index]["sentences"][-1]["end_time"])
                    if self.cfg.generate_scripts:
                        transcribed_speaker_texts[speaker_index] = \
                            print_sentences(sentences=self._word_and_ts_seq[speaker_index]["sentences"], 
                            color_palette=get_color_palette(), 
                            params=self.cfg)
                        write_txt(f'{self.cfg.print_path}'.replace(".sh", f"_spk{speaker_index}.sh"), 
                                  transcribed_speaker_texts[speaker_index].strip())
        
        return (transcribed_speaker_texts,
                transcribed_texts,
                asr_pred_out_stream,
                transcribed_texts,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
                previous_hypotheses,
                mem_last_time,
                fifo_last_time,
                diar_pred_out_stream)

    def _add_speaker_transcriptions(self, transcriptions: list, speaker_transcriptions: List[str], word_and_ts_seq: List[Dict[str, Any]]) -> Tuple[List[Hypothesis], List[Hypothesis]]:
        """ 
        Add speaker tagging into the transcriptions generated from an ASR model.
        
        Args:
            transcriptions:
            speaker_transcriptions (List[str]): List of speaker transcriptions.
            word_and_ts_seq (List[Dict[str, Any]]): List of word-level dictionaries.
            
        Returns:
            Tuple[List[Hypothesis], List[Hypothesis]]: Tuple containing the updated transcriptions with speaker tags.
        """
        trans_hyp, nbest_hyp = transcriptions
        for sess_idx, hypothesis in enumerate(trans_hyp):
            if speaker_transcriptions[sess_idx] is not None:
                trans_hyp[sess_idx].text = speaker_transcriptions[sess_idx]
            speaker_added_word_dicts = [] 
            for word_idx, trans_wdict in enumerate(trans_hyp[0].timestep['word']):
                trans_wdict_copy = deepcopy(trans_wdict)
                trans_wdict_copy['speaker'] = word_and_ts_seq[sess_idx]['words'][word_idx]['speaker']
                speaker_added_word_dicts.append(trans_wdict_copy)
            trans_hyp[sess_idx].timestep['word'] = speaker_added_word_dicts
            w_count = 0
            segment_list = []
            for word_idx, trans_segdict in enumerate(trans_hyp[0].timestep['segment']):
                words = trans_segdict['segment'].split()
                spk_vote_pool = []
                for word in words:
                    assert word.lower() == word_and_ts_seq[sess_idx]['words'][w_count]['word'].lower()
                    spk_int = int(word_and_ts_seq[sess_idx]['words'][w_count]['speaker'].split('_')[-1])
                    spk_vote_pool.append(spk_int)
                    w_count += 1
                trans_segdict['speaker'] = f"speaker_{torch.mode(torch.tensor(spk_vote_pool), dim=0).values.item()}"
                segment_list.append(trans_segdict)
            trans_hyp[sess_idx].timestep['segment'] = segment_list
                
        transcriptions = (trans_hyp, trans_hyp)
        return transcriptions
        
    def perform_offline_stt_spk(self, override_cfg):
        """ 
        Perform offline STT and speaker diarization on the provided manifest file.
        
        Args:
            override_cfg (dict): Override configuration parameters.
            
        Returns:
            transcriptions (Tuple): Tuple containing the speaker-tagged transcripts.
        """
        transcriptions = self.asr_model.transcribe(
            audio=self.cfg.dataset_manifest,
            override_config=override_cfg,
        )
        best_hyp, nbest_hyp = transcriptions
        spk_timestamps, pred_tensors = self.diar_model.diarize(audio=self.cfg.manifest_file, 
                                                               include_tensor_outputs=True)
        speaker_transcriptions, word_and_ts_seq = self.merge_transcript_and_speakers(
                                    test_manifest_dict=self.diar_model._diarize_audio_rttm_map, 
                                    asr_hypotheses=best_hyp, 
                                    diar_pred_out=pred_tensors
                                )
        transcriptions = self._add_speaker_transcriptions(transcriptions, speaker_transcriptions, word_and_ts_seq)
        return transcriptions
