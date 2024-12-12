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

import contextlib
import json
import os
import time
import yaml
from tqdm import tqdm
from dataclasses import dataclass, is_dataclass
from typing import Optional, Union, List, Tuple, Dict

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from omegaconf import open_dict
from pytorch_lightning import seed_everything

# ASR
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

import numpy as np
from copy import deepcopy
from nemo.collections.asr.parts.utils.diarization_utils import read_seglst, OnlineEvaluation
from nemo.utils import logging

# DIARIZATION
# from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel

from nemo.core.config import hydra_runner
from nemo.collections.asr.metrics.der import score_labels
from hydra.core.config_store import ConfigStore

from pyannote.core import Segment, Timeline

from nemo.collections.asr.parts.utils.speaker_utils import (
audio_rttm_map as get_audio_rttm_map,
rttm_to_labels,
)
from nemo.collections.asr.parts.utils.vad_utils import ts_vad_post_processing, timestamps_to_pyannote_object

from nemo.collections.asr.parts.utils.diarization_utils import (
print_sentences,
get_color_palette,
write_txt,
)


import hydra
from typing import List, Optional
from dataclasses import dataclass, field
from beam_search_utils import (
    SpeakerTaggingBeamSearchDecoder,
    load_input_jsons,
    load_reference_jsons,
    run_mp_beam_search_decoding,
    convert_nemo_json_to_seglst,
)
from hydra.core.config_store import ConfigStore
from collections import OrderedDict
import itertools

import time
from functools import wraps
import math

def format_time(seconds):
    minutes = math.floor(seconds / 60)
    sec = seconds % 60
    return f"{minutes}:{sec:05.2f}"

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

class MultiSpeakerASRstreamer:
    def __init__(
        self,
        cfg,
        asr_model,
        diar_model,
        bsd_spk
    ):
        self.cfg = cfg
        self.asr_model = asr_model
        self.diar_model = diar_model
        self.bsd_spk = bsd_spk
        self.fix_prev_words_count = cfg.fix_prev_words_count
        self._sentence_render_length = int(self.fix_prev_words_count + cfg.update_prev_words_sentence)
        self._word_count = 0
        self.test_manifest_dict = get_audio_rttm_map(self.cfg.manifest_file)
        self._initial_steps = cfg.ignored_initial_frame_steps
        self._all_sentences = []
        self._init_evaluator() 
    
    def _init_evaluator(self):  
        self.online_evaluators = []
        self._word_and_ts_seq = []
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
            ref_seglst = read_seglst(data_dict['seglst_filepath'])
            ref_rttm_labels = rttm_to_labels(data_dict.get('rttm_filepath', None))
            
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
        
    def get_sentences_values(self, session_trans_dict, sentence_render_length):
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
            session_trans_dict['sentence_memory'].update({stt_word_index:(deepcopy(sentences), deepcopy(sentence), sentence['speaker'])})
            prev_speaker = session_trans_dict['words'][stt_word_index]['speaker']
        else:
            (_sentences, _sentence, prev_speaker) = session_trans_dict['sentence_memory'][stt_word_index]
            sentences, sentence = deepcopy(_sentences), deepcopy(_sentence)
        
        for k in range(stt_word_index + 1, len(session_trans_dict['words'])):
            word_dict = session_trans_dict['words'][k]
            word, speaker = word_dict['word'], word_dict['speaker']
            start_point, end_point = word_dict['start_time'], word_dict['end_time']
            if word_dict['speaker'] != prev_speaker:
                sentence['text'] = sentence['text'].strip()
                sentences.append(sentence)
                sentence = self._get_sentence(word_dict=session_trans_dict['words'][k])
            else:
                sentence['end_time'] = end_point
            stt_sec, end_sec = start_point, end_point
            sentence['text'] += word.strip() + ' '
            sentence['words'] = sentence['text']
            sentence['session_id'] = session_trans_dict['uniq_id']
            session_trans_dict['last_word_index'] = k
            prev_speaker = word_dict['speaker']
            session_trans_dict['sentence_memory'][k] = (deepcopy(sentences), deepcopy(sentence), prev_speaker)
        sentence['text'] = sentence['text'].strip()
        sentences.append(sentence)
        session_trans_dict['sentences'] = sentences
        return session_trans_dict 


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
                self._word_and_ts_seq[idx] = get_frame_and_words(cfg=self.cfg,
                                                      uniq_id=uniq_id,
                                                      tokenizer=self.asr_model.tokenizer,
                                                      step_num=step_num, 
                                                      diar_pred_out_stream=diar_pred_out_stream[idx, :, :],
                                                      previous_hypothesis=previous_hypotheses[idx], 
                                                      word_and_ts_seq=self._word_and_ts_seq[idx],
                                                      fix_prev_words_count=self.cfg.fix_prev_words_count)
                if len(self._word_and_ts_seq[idx]["words"]) > 0:
                    self._word_and_ts_seq[idx] = self.get_sentences_values(session_trans_dict=self._word_and_ts_seq[idx], 
                                                                           sentence_render_length=self._sentence_render_length)
                    transcribed_speaker_texts[idx] = print_sentences(sentences=self._word_and_ts_seq[idx]["sentences"], color_palette=get_color_palette(), params=self.cfg)
                    if self.cfg.eval_mode:
                        der, cpwer, is_update = self.online_evaluators[idx].evaluate_inloop(hyp_seglst=self._word_and_ts_seq[idx]["sentences"], 
                                                                                            end_step_time=self._word_and_ts_seq[idx]["sentences"][-1]["end_time"])
                    write_txt(f'{self.cfg.print_path}'.replace(".sh", f"_{idx}.sh"), transcribed_speaker_texts[idx].strip())
            if self.cfg.log:         
                logging.info(f"mem: {mem_last_time.shape}, fifo: {fifo_last_time.shape}, pred: {diar_pred_out_stream.shape}")
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
    
@dataclass
class DiarizationConfig:
    # Required configs
    diar_model_path: Optional[str] = None  # Path to a .nemo file
    diar_pretrained_name: Optional[str] = None  # Name of a pretrained model
    audio_dir: Optional[str] = None  # Path to a directory which contains audio files
    # dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest
    
    audio_key: str = 'audio_filepath'  # Used to override the default audio key in dataset_manifest
    postprocessing_yaml: Optional[str] = None  # Path to a yaml file for postprocessing configurations
    eval_mode: bool = True
    no_der: bool = False
    out_rttm_dir: Optional[str] = None
    opt_style: Optional[str] = None
    
    # General configs
    session_len_sec: float = -1 # End-to-end diarization session length in seconds
    batch_size: int = 1
    num_workers: int = 8
    random_seed: Optional[int] = None  # seed number going to be used in seed_everything()
    bypass_postprocessing: bool = True # If True, postprocessing will be bypassed
    log: bool = True # If True, log will be printed
    
    # Eval Settings: (0.25, False) should be default setting for sortformer eval.
    collar: float = 0.25 # Collar in seconds for DER calculation
    ignore_overlap: bool = False # If True, DER will be calculated only for non-overlapping segments
    
    # Streaming diarization configs
    streaming_mode: bool = True # If True, streaming diarization will be used. For long-form audio, set mem_len=step_len
    mem_len: int = 188
    # mem_refresh_rate: int = 0
    fifo_len: int = 188
    step_len: int = 0
    step_left_context: int = 0
    step_right_context: int = 0

    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    allow_mps: bool = False  # allow to select MPS device (Apple Silicon M-series GPU)
    matmul_precision: str = "highest"  # Literal["highest", "high", "medium"]

    # Optuna Config
    optuna_study_name: str = "diar_study"
    storage: str = f"sqlite:///{optuna_study_name}.db"
    output_log_file: str = f"{optuna_study_name}.log"
    optuna_n_trials: int = 100000

    # ASR Configs
    asr_model: Optional[str] = None
    diar_model: Optional[str] = None
    device: str = 'cuda'
    audio_file: Optional[str] = None
    manifest_file: Optional[str] = None
    use_amp: bool = True
    debug_mode: bool = False
    compare_vs_offline: bool = False
    batch_size: int = 32
    chunk_size: int = -1
    shift_size: int = -1
    left_chunks: int = 2
    online_normalization: bool = False
    output_path: Optional[str] = None
    pad_and_drop_preencoded: bool = False
    set_decoder: Optional[str] = None # ["ctc", "rnnt"]
    att_context_size: Optional[str] = None

    
    # Beam search parameters
    arpa_language_model: Optional[str] = None
    beam_prune_logp: float = -40
    word_window: int = 50
    fix_prev_words_count: int = 5
    update_prev_words_sentence: int = 2
    bsd_maj_voting: bool = True
    use_spk_turn_bsd: bool = False
    left_frame_shift: int = -1
    right_frame_shift: int = -1
    min_sigmoid_val: float = 1e-2
    port: List[int] = field(default_factory=list)
    parallel_chunk_word_len: int = 250
    use_ngram: bool = True
    peak_prob: float = 0.95
    finetune_realtime_ratio: float = 1.03
    discarded_frames: int = 8
    feat_len_sec: float = 0.01
    limit_max_spks: int = 2
    alpha: float = 0.2
    beta: float = 0.03
    beam_width: int = 8
    out_dir: Optional[str] = None
    print_time: bool = True
    colored_text: bool = True
    real_time_mode: bool = False
    print_path: str = "./"
    beam_search_enabled: bool = False
    ignored_initial_frame_steps: int = 5
    verbose: bool = False
    break_lines: bool = True

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
    
def fix_frame_time_step(cfg, new_tokens, new_words, frame_inds_seq):
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
                f"Length of new token sequence ({len(new_tokens)}) does not match length of frame indices sequence ({len(frame_inds_seq)}). Skipping this chunk."
            )
    return frame_inds_seq

def get_word_dict_content(cfg, word, word_index, diar_pred_out_stream, token_group, frame_inds_seq, time_step_local_offset, frame_len: float = 0.08):
    _stt, _end = time_step_local_offset, time_step_local_offset + len(token_group)-1
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
    speaker_sigmoid = diar_pred_out_stream[max((frame_stt + cfg.left_frame_shift), 0):(frame_end + cfg.right_frame_shift), :].mean(dim=0)
    speaker_sigmoid = torch.clamp(speaker_sigmoid, min=cfg.min_sigmoid_val, max=1) 
    speaker_softmax = speaker_sigmoid / speaker_sigmoid.sum()
    # _speaker_softmax = torch.softmax(speaker_sigmoid, dim=0)
    speaker_softmax = speaker_softmax.cpu()
    stt_sec, end_sec = frame_stt * frame_len, frame_end * frame_len
    speaker_softmax[cfg.limit_max_spks:] = 0.0
    spk_id = speaker_softmax.argmax().item()
    speaker_votes = torch.zeros(speaker_softmax.shape[0]).to(torch.int)
    speaker_votes[spk_id] += 1
    word_dict = {"word": word,
                 "word_index": word_index,
                'frame_stt': frame_stt,
                'frame_end': frame_end,
                'start_time': round(stt_sec, 3), 
                'end_time': round(end_sec, 3), 
                'speaker': f"speaker_{spk_id}",
                'speaker_votes': speaker_votes,
                'speaker_softmax': speaker_softmax} 
    return word_dict

def get_multitoken_words(cfg, word_and_ts_seq,  word_seq, new_words, fix_prev_words_count=5):
    """Fix the multi-token words which are not fully captured by the previous chunk window.""" 
    prev_start, prev_end = max(0, len(word_seq)-fix_prev_words_count-len(new_words)), max(0, len(word_seq)-len(new_words))
    for ct, prev_word in enumerate(word_seq[prev_start: prev_end]):
        if len(word_and_ts_seq["words"]) > fix_prev_words_count - ct:
            saved_word = word_and_ts_seq["words"][-fix_prev_words_count + ct]["word"]
            if len(prev_word) > len(saved_word):
                if cfg.verbose:
                    logging.info(f"[Replacing Multi-token Word]: {word_and_ts_seq['words'][-fix_prev_words_count + ct]['word']} with {prev_word}")
                word_and_ts_seq["words"][-fix_prev_words_count + ct]["word"] = prev_word
    return word_and_ts_seq
   
def append_word_and_ts_seq(cfg, word_idx_offset, word_and_ts_seq, word_dict): 
    """Append the word dictionary to the word and time-stamp sequence. """ 
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
    
def get_frame_and_words(cfg, uniq_id, tokenizer, step_num, diar_pred_out_stream, previous_hypothesis, word_and_ts_seq, frame_len=0.08, fix_prev_words_count=5):
    offset = step_num * previous_hypothesis.length.item()
    word_seq = previous_hypothesis.text.split()
    new_words = word_seq[word_and_ts_seq["offset_count"]:]
    frame_inds_seq = (torch.tensor(previous_hypothesis.timestep) + offset).tolist()
    new_token_group = tokenizer.text_to_tokens(new_words)
    new_tokens = list(itertools.chain(*new_token_group))
    frame_inds_seq = fix_frame_time_step(cfg, new_tokens, new_words, frame_inds_seq)
    min_len = min(len(new_words), len(frame_inds_seq))
    word_and_ts_seq['uniq_id'] = uniq_id

    for idx in range(min_len):
        word_and_ts_seq["token_frame_index"].append((new_tokens[idx], frame_inds_seq[idx]))
        word_and_ts_seq["offset_count"] += 1
    
    time_step_local_offset, word_idx_offset = 0, 0
    word_count_offset = len(word_and_ts_seq["words"]) 
    word_and_ts_seq = get_multitoken_words(cfg, word_and_ts_seq, word_seq, new_words, fix_prev_words_count=fix_prev_words_count)
    
    # Get the FIFO queue preds to word_and_ts_seq 
    local_idx = 0
    for local_idx, (token_group, word) in enumerate(zip(new_token_group, new_words)):
        word_dict = get_word_dict_content(cfg=cfg, 
                                          word=word,
                                          word_index= (word_count_offset + local_idx),
                                          diar_pred_out_stream=diar_pred_out_stream,
                                          token_group=token_group,
                                          frame_inds_seq=frame_inds_seq,
                                          time_step_local_offset=time_step_local_offset,
                                          frame_len=frame_len
                                          )
        # Count the number of speakers in the word window
        time_step_local_offset += len(token_group)
        word_idx_offset, word_and_ts_seq = append_word_and_ts_seq(cfg, word_idx_offset, word_and_ts_seq, word_dict)
    return word_and_ts_seq

def perform_streaming(cfg, asr_model, diar_model, bsd_spk, streaming_buffer, debug_mode=False):
    batch_size = len(streaming_buffer.streams_length)
    final_offline_tran = None

    cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
        batch_size=batch_size
    )

    previous_hypotheses = None
    streaming_buffer_iter = iter(streaming_buffer)
    asr_pred_out_stream, diar_pred_out_stream  = None, None
    mem_last_time, fifo_last_time = None, None
    left_offset, right_offset = 0, 0

    multispk_asr_streamer = MultiSpeakerASRstreamer(cfg, asr_model, diar_model, bsd_spk)
    session_start_time = time.time()
    feat_frame_count = 0
    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
        loop_start_time = time.time()
        with torch.inference_mode():
            with autocast:
                with torch.no_grad(): 
                    (transcribed_speaker_texts,
                    transcribed_texts,
                    asr_pred_out_stream,
                    transcribed_texts,
                    cache_last_channel,
                    cache_last_time,
                    cache_last_channel_len,
                    previous_hypotheses,
                    mem_last_time,
                    fifo_last_time,
                    diar_pred_out_stream) = multispk_asr_streamer.perform_streaming_stt_spk(
                        step_num=step_num,
                        chunk_audio=chunk_audio,
                        chunk_lengths=chunk_lengths,
                        cache_last_channel=cache_last_channel,
                        cache_last_time=cache_last_time,
                        cache_last_channel_len=cache_last_channel_len,
                        is_buffer_empty=streaming_buffer.is_buffer_empty(),
                        previous_hypotheses=previous_hypotheses,
                        asr_pred_out_stream=asr_pred_out_stream,
                        diar_pred_out_stream=diar_pred_out_stream,
                        mem_last_time=mem_last_time,
                        fifo_last_time=fifo_last_time,
                        left_offset=left_offset,
                        right_offset=right_offset,
                        pad_and_drop_preencoded=False,
                    )

        if debug_mode:
            logging.info(f"Streaming transcriptions: {extract_transcriptions(transcribed_texts)}")
        loop_end_time = time.time()
        feat_frame_count += (chunk_audio.shape[-1] - cfg.discarded_frames)
        if cfg.real_time_mode:
            time_diff = max(0, (time.time() - session_start_time) - feat_frame_count * cfg.feat_len_sec)
            eta_min_sec = format_time(time.time() - session_start_time)
            logging.info(f"[   REAL TIME MODE   ] min:sec - {eta_min_sec} Time difference for real-time mode: {time_diff:.4f} seconds")
            time.sleep(max(0, (chunk_audio.shape[-1] - cfg.discarded_frames)*cfg.feat_len_sec - (loop_end_time - loop_start_time) - time_diff * cfg.finetune_realtime_ratio))
    final_streaming_tran = extract_transcriptions(transcribed_texts)
    return final_streaming_tran, final_offline_tran


@hydra_runner(config_name="DiarizationConfig", schema=DiarizationConfig)
def main(cfg: DiarizationConfig) -> Union[DiarizationConfig]:

    for key in cfg:
        cfg[key] = None if cfg[key] == 'None' else cfg[key]

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if cfg.random_seed:
        pl.seed_everything(cfg.random_seed)
        
    if cfg.diar_model_path is None and cfg.diar_pretrained_name is None:
        raise ValueError("Both cfg.diar_model_path and cfg.pretrained_name cannot be None!")
    if cfg.audio_dir is None and cfg.manifest_file is None:
        raise ValueError("Both cfg.audio_dir and cfg.manifest_file cannot be None!")

    # setup GPU
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    if cfg.cuda is None:
        if torch.cuda.is_available():
            device = [0]  # use 0th CUDA device
            accelerator = 'gpu'
            map_location = torch.device('cuda:0')
        elif cfg.allow_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = [0]
            accelerator = 'mps'
            map_location = torch.device('mps')
        else:
            device = 1
            accelerator = 'cpu'
            map_location = torch.device('cpu')
    else:
        device = [cfg.cuda]
        accelerator = 'gpu'
        map_location = torch.device(f'cuda:{cfg.cuda}')

    if cfg.diar_model_path.endswith(".ckpt"):
        diar_model = SortformerEncLabelModel.load_from_checkpoint(checkpoint_path=cfg.diar_model_path, map_location=map_location, strict=False)
    elif cfg.diar_model_path.endswith(".nemo"):
        diar_model = SortformerEncLabelModel.restore_from(restore_path=cfg.diar_model_path, map_location=map_location)
    else:
        raise ValueError("cfg.diar_model_path must end with.ckpt or.nemo!")
    
    diar_model._cfg.test_ds.session_len_sec = cfg.session_len_sec
    trainer = pl.Trainer(devices=device, accelerator=accelerator)
    diar_model.set_trainer(trainer)
    
    diar_model = diar_model.eval()
    diar_model._cfg.test_ds.manifest_filepath = cfg.manifest_file
    # infer_audio_rttm_dict = get_audio_rttm_map(cfg.manifest_file)
    diar_model._cfg.test_ds.batch_size = cfg.batch_size
    
    # Model setup for inference 
    diar_model._cfg.test_ds.num_workers = cfg.num_workers
    diar_model.setup_test_data(test_data_config=diar_model._cfg.test_ds)    
    
    # Steaming mode setup 
    diar_model.streaming_mode = cfg.streaming_mode
    diar_model.sortformer_modules.step_len = cfg.step_len
    diar_model.sortformer_modules.mem_len = cfg.mem_len
    diar_model.sortformer_modules.step_left_context = cfg.step_left_context
    diar_model.sortformer_modules.step_right_context = cfg.step_right_context
    diar_model.sortformer_modules.fifo_len = cfg.fifo_len
    diar_model.sortformer_modules.log = cfg.log

    args = cfg
    if (args.audio_file is None and args.manifest_file is None) or (
        args.audio_file is not None and args.manifest_file is not None
    ):
        raise ValueError("One of the audio_file and manifest_file should be non-empty!")

    if args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=args.asr_model)
    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.asr_model)

    logging.info(asr_model.encoder.streaming_cfg)
    if args.set_decoder is not None:
        if hasattr(asr_model, "cur_decoder"):
            asr_model.change_decoding_strategy(decoder_type=args.set_decoder)
        else:
            raise ValueError("Decoder cannot get changed for non-Hybrid ASR models.")

    if args.att_context_size is not None:
        if hasattr(asr_model.encoder, "set_default_att_context_size"):
            asr_model.encoder.set_default_att_context_size(att_context_size=json.loads(args.att_context_size))
        else:
            raise ValueError("Model does not support multiple lookaheads.")

    global autocast
    autocast = torch.amp.autocast(asr_model.device.type, enabled=args.use_amp)

    # configure the decoding config
    decoding_cfg = asr_model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.strategy = "greedy"
        decoding_cfg.preserve_alignments = False
        if hasattr(asr_model, 'joint'):  # if an RNNT model
            decoding_cfg.greedy.max_symbols = 10
            decoding_cfg.fused_batch_size = -1
        asr_model.change_decoding_strategy(decoding_cfg)

    asr_model = asr_model.to(args.device)
    asr_model.eval()

    # chunk_size is set automatically for models trained for streaming. For models trained for offline mode with full context, we need to pass the chunk_size explicitly.
    if args.chunk_size > 0:
        if args.shift_size < 0:
            shift_size = args.chunk_size
        else:
            shift_size = args.shift_size
        asr_model.encoder.setup_streaming_params(
            chunk_size=args.chunk_size, left_chunks=args.left_chunks, shift_size=shift_size
        )

    # In streaming, offline normalization is not feasible as we don't have access to the whole audio at the beginning
    # When online_normalization is enabled, the normalization of the input features (mel-spectrograms) are done per step
    # It is suggested to train the streaming models without any normalization in the input features.
    if args.online_normalization:
        if asr_model.cfg.preprocessor.normalize not in ["per_feature", "all_feature"]:
            logging.warning(
                "online_normalization is enabled but the model has no normalization in the feature extration part, so it is ignored."
            )
            online_normalization = False
        else:
            online_normalization = True

    else:
        online_normalization = False

    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=online_normalization,
        pad_and_drop_preencoded=args.pad_and_drop_preencoded,
    )
    
    bsd_spk = None
    
    if args.audio_file is not None:
        # stream a single audio file
        processed_signal, processed_signal_length, stream_id = streaming_buffer.append_audio_file(
            args.audio_file, stream_id=-1
        )
        perform_streaming(
            cfg=cfg,
            asr_model=asr_model,
            diar_model=diar_model,
            bsd_spk=bsd_spk,
            streaming_buffer=streaming_buffer,
        )
    else:
        # stream audio files in a manifest file in batched mode
        samples = []
        all_streaming_tran = []
        all_offline_tran = []
        all_refs_text = []

        with open(args.manifest_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                samples.append(item)

        # Override batch size: The batch size should be equal to the number of samples in the manifest file
        args.batch_size = len(samples)
        logging.info(f"Loaded {len(samples)} from the manifest at {args.manifest_file}.")

        start_time = time.time()
        for sample_idx, sample in enumerate(samples):
            processed_signal, processed_signal_length, stream_id = streaming_buffer.append_audio_file(
                sample['audio_filepath'], offset=sample['offset'], duration=sample['duration'], stream_id=-1
            )
            if "text" in sample:
                all_refs_text.append(sample["text"])
            logging.info(f'Added this sample to the buffer: {sample["audio_filepath"]}')

            if (sample_idx + 1) % args.batch_size == 0 or sample_idx == len(samples) - 1:
                logging.info(f"Starting to stream samples {sample_idx - len(streaming_buffer) + 1} to {sample_idx}...")
                streaming_tran, offline_tran = perform_streaming(
                    cfg=cfg,
                    asr_model=asr_model,
                    diar_model=diar_model,
                    bsd_spk=bsd_spk,
                    streaming_buffer=streaming_buffer,
                    debug_mode=args.debug_mode,
                )
                
if __name__ == '__main__':
    main()
