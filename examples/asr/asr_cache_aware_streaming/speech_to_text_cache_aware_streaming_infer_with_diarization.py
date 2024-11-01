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

"""
This script can be used to simulate cache-aware streaming for ASR models. The ASR model to be used with this script need to get trained in streaming mode. Currently only Conformer models supports this streaming mode.
You may find examples of streaming models under 'NeMo/example/asr/conf/conformer/streaming/'.

It works both on a manifest of audio files or a single audio file. It can perform streaming for a single stream (audio) or perform the evalution in multi-stream model (batch_size>1).
The manifest file must conform to standard ASR definition - containing `audio_filepath` and `text` as the ground truth.

# Usage

## To evaluate a model in cache-aware streaming mode on a single audio file:

python speech_to_text_streaming_infer.py \
    --asr_model=asr_model.nemo \
    --audio_file=audio_file.wav \
    --compare_vs_offline \
    --use_amp \
    --debug_mode

## To evaluate a model in cache-aware streaming mode on a manifest file:

python speech_to_text_streaming_infer.py \
    --asr_model=asr_model.nemo \
    --manifest_file=manifest_file.json \
    --batch_size=16 \
    --compare_vs_offline \
    --use_amp \
    --debug_mode

You may drop the '--debug_mode' and '--compare_vs_offline' to speedup the streaming evaluation.
If compare_vs_offline is not used, then significantly larger batch_size can be used.
Setting `--pad_and_drop_preencoded` would perform the caching for all steps including the first step.
It may result in slightly different outputs from the sub-sampling module compared to offline mode for some techniques like striding and sw_striding.
Enabling it would make it easier to export the model to ONNX.

## Hybrid ASR models
For Hybrid ASR models which have two decoders, you may select the decoder by --set_decoder DECODER_TYPE, where DECODER_TYPE can be "ctc" or "rnnt".
If decoder is not set, then the default decoder would be used which is the RNNT decoder for Hybrid ASR models.

## Multi-lookahead models
For models which support multiple lookaheads, the default is the first one in the list of model.encoder.att_context_size. To change it, you may use --att_context_size, for example --att_context_size [70,1].


## Evaluate a model trained with full context for offline mode

You may try the cache-aware streaming with a model trained with full context in offline mode.
But the accuracy would not be very good with small chunks as there is inconsistency between how the model is trained and how the streaming inference is done.
The accuracy of the model on the borders of chunks would not be very good.

To use a model trained with full context, you need to pass the chunk_size and shift_size arguments.
If shift_size is not passed, chunk_size would be used as the shift_size too.
Also argument online_normalization should be enabled to simulate a realistic streaming.
The following command would simulate cache-aware streaming on a pretrained model from NGC with chunk_size of 100, shift_size of 50 and 2 left chunks as left context.
The chunk_size of 100 would be 100*4*10=4000ms for a model with 4x downsampling and 10ms shift in feature extraction.

python speech_to_text_streaming_infer.py \
    --asr_model=stt_en_conformer_ctc_large \
    --chunk_size=100 \
    --shift_size=50 \
    --left_chunks=2 \
    --online_normalization \
    --manifest_file=manifest_file.json \
    --batch_size=16 \
    --compare_vs_offline \
    --use_amp \
    --debug_mode

"""


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
from nemo.utils import logging

# DIARIZATION
from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.core.config import hydra_runner
from nemo.collections.asr.metrics.der import score_labels
from hydra.core.config_store import ConfigStore

from pyannote.core import Segment, Timeline
from nemo.collections.asr.parts.utils.speaker_utils import audio_rttm_map as get_audio_rttm_map
from nemo.collections.asr.parts.utils.vad_utils import ts_vad_post_processing, timestamps_to_pyannote_object

from nemo.collections.asr.parts.utils.diarization_utils import (
get_session_trans_dict,
init_session_trans_dict,
init_session_gecko_dict,
print_sentences,
get_color_palette,
write_txt,
)
from nemo.collections.asr.parts.utils.speaker_utils import (
labels_to_pyannote_object,
generate_diarization_output_lines,
rttm_to_labels,
get_uem_object,
)


import hydra
from typing import List, Optional
from dataclasses import dataclass, field
import kenlm
from beam_search_utils import (
    SpeakerTaggingBeamSearchDecoder,
    load_input_jsons,
    load_reference_jsons,
    run_mp_beam_search_decoding,
    convert_nemo_json_to_seglst,
)
from hydra.core.config_store import ConfigStore
# from hyper_optim import (
#     optuna_hyper_optim, 
#     evaluate,
#     evaluate_diff,
# )

from collections import OrderedDict
import itertools




# cs = ConfigStore.instance()
# cs.store(name="config", node=RealigningLanguageModelParameters)

# ############### DIARIZATION CONFIGS ################
# @dataclass
# class PostProcessingParams:
#     window_length_in_sec: float = 0.15
#     shift_length_in_sec: float = 0.01
#     smoothing: bool = False
#     overlap: float = 0.5
#     onset: float = 0.5
#     offset: float = 0.5
#     pad_onset: float = 0.0
#     pad_offset: float = 0.0
#     min_duration_on: float = 0.0
#     min_duration_off: float = 0.0
#     filter_speech_first: bool = True

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
    batch_size: int = 4
    num_workers: int = 0
    random_seed: Optional[int] = None  # seed number going to be used in seed_everything()
    bypass_postprocessing: bool = True # If True, postprocessing will be bypassed
    log: bool = False # If True, log will be printed
    
    # Eval Settings: (0.25, False) should be default setting for sortformer eval.
    collar: float = 0.25 # Collar in seconds for DER calculation
    ignore_overlap: bool = False # If True, DER will be calculated only for non-overlapping segments
    
    # Streaming diarization configs
    streaming_mode: bool = True # If True, streaming diarization will be used. For long-form audio, set mem_len=step_len
    mem_len: int = 100
    # mem_refresh_rate: int = 0
    fifo_len: int = 100
    step_len: int = 100
    step_left_context: int = 100
    step_right_context: int = 100

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
    use_amp: bool = False
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
    # batch_size: int = 32
    # use_mp: bool = True
    arpa_language_model: Optional[str] = None
    beam_prune_logp: float = -100
    word_window: int = 32
    port: List[int] = field(default_factory=list)
    parallel_chunk_word_len: int = 250
    use_ngram: bool = True
    peak_prob: float = 0.95
    limit_max_spks: int = 2
    alpha: float = 0.5
    beta: float = 0.05
    beam_width: int = 16
    out_dir: Optional[str] = None
    print_time: bool = True
    colored_text: bool = True
    print_path: str = "./"
    beam_search_enabled: bool = True

# def load_postprocessing_from_yaml(postprocessing_yaml):
#     """ 
#     Load postprocessing parameters from a YAML file.

#     Args:
#         postprocessing_yaml (str): 
#             Path to a YAML file for postprocessing configurations.

#     Returns:
#         postprocessing_params (dataclass): 
#             Postprocessing parameters loaded from the YAML file.
#     """
#     # Add PostProcessingParams as a field
#     postprocessing_params = OmegaConf.structured(PostProcessingParams())
#     if postprocessing_yaml is None:
#         logging.info(f"No postprocessing YAML file has been provided. Default postprocessing configurations will be applied.")
#     else:
#         # Load postprocessing params from the provided YAML file
#         with open(postprocessing_yaml, 'r') as file:
#             yaml_params = yaml.safe_load(file)['parameters']
#             # Update the postprocessing_params with the loaded values
#             logging.info(f"Postprocessing YAML file '{postprocessing_yaml}' has been loaded.")
#             for key, value in yaml_params.items():
#                 if hasattr(postprocessing_params, key):
#                     setattr(postprocessing_params, key, value)
#     return postprocessing_params

def convert_pred_mat_to_segments(
    audio_rttm_map_dict: Dict[str, Dict[str, str]], 
    postprocessing_cfg, 
    batch_preds_list: List[torch.Tensor], 
    unit_10ms_frame_count:int = 8,
    bypass_postprocessing: bool = False,
    out_rttm_dir: str | None = None,
    ):
    """
    Convert prediction matrix to time-stamp segments.

    Args:
        audio_rttm_map_dict (dict): dictionary of audio file path, offset, duration and RTTM filepath.
        batch_preds_list (List[torch.Tensor]): list of prediction matrices containing sigmoid values for each speaker.
            Dimension: [(1, frames, num_speakers), ..., (1, frames, num_speakers)]
        unit_10ms_frame_count (int, optional): number of 10ms segments in a frame. Defaults to 8.
        bypass_postprocessing (bool, optional): if True, postprocessing will be bypassed. Defaults to False.

    Returns:
       all_hypothesis (list): list of pyannote objects for each audio file.
       all_reference (list): list of pyannote objects for each audio file.
       all_uems (list): list of pyannote objects for each audio file.
    """
    batch_pred_ts_segs, all_hypothesis, all_reference, all_uems = [], [], [], []
    cfg_vad_params = OmegaConf.structured(postprocessing_cfg)
    for sample_idx, (uniq_id, audio_rttm_values) in tqdm(enumerate(audio_rttm_map_dict.items()), total=len(audio_rttm_map_dict), desc="Running post-processing"):
        spk_ts = []
        offset, duration = audio_rttm_values['offset'], audio_rttm_values['duration']
        speaker_assign_mat = batch_preds_list[sample_idx].squeeze(dim=0)
        speaker_timestamps = [[] for _ in range(speaker_assign_mat.shape[-1])]
        for spk_id in range(speaker_assign_mat.shape[-1]):
            ts_mat = ts_vad_post_processing(speaker_assign_mat[:, spk_id], 
                                            cfg_vad_params=cfg_vad_params, 
                                            unit_10ms_frame_count=unit_10ms_frame_count, 
                                            bypass_postprocessing=bypass_postprocessing)
            ts_mat = ts_mat + offset
            ts_mat = torch.clamp(ts_mat, min=offset, max=(offset + duration))
            ts_seg_list = ts_mat.tolist()
            speaker_timestamps[spk_id].extend(ts_seg_list)
            spk_ts.append(ts_seg_list)
        all_hypothesis, all_reference, all_uems = timestamps_to_pyannote_object(speaker_timestamps, 
                                                                                uniq_id, 
                                                                                audio_rttm_values, 
                                                                                all_hypothesis, 
                                                                                all_reference, 
                                                                                all_uems,
                                                                               )
        batch_pred_ts_segs.append(spk_ts) 
    return all_hypothesis, all_reference, all_uems

############### Cache-aware streaming ASR ################
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
    
def get_sentences_values(session_trans_dict):
    word_dict_seq_list = session_trans_dict['words']
    prev_speaker = word_dict_seq_list[0]['speaker']
    sentences = []
    sentence = {'speaker': prev_speaker,
                'start_time': session_trans_dict['words'][0]['start_time'], 
                'end_time': session_trans_dict['words'][0]['end_time'], 
                'text': ''}
    for k, word_dict in enumerate(word_dict_seq_list):
        word, speaker = word_dict['word'], word_dict['speaker']
        start_point, end_point = word_dict['start_time'], word_dict['end_time']
        if speaker != prev_speaker:
            sentence['text'] = sentence['text'].strip()
            sentences.append(sentence)
            sentence = {'speaker': speaker, 'start_time': start_point, 'end_time': end_point, 'text': ''}
        else:
            sentence['end_time'] = end_point
        stt_sec, end_sec = start_point, end_point
        sentence['text'] += word.strip() + ' '
        prev_speaker = speaker

    session_trans_dict['words'] = word_dict_seq_list
    sentence['text'] = sentence['text'].strip()
    sentences.append(sentence)
    session_trans_dict['sentences'] = sentences
    return session_trans_dict 


def fix_frame_time_step(new_tokens, new_words, frame_inds_seq):
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
        logging.warning(
            f"Length of new token sequence ({len(new_tokens)}) does not match length of frame indices sequence ({len(frame_inds_seq)}). Skipping this chunk."
        )
    return frame_inds_seq

def get_word_dict_content(word, diar_pred_out_stream, token_group, frame_inds_seq, time_step_local_offset, frame_len: float = 0.08):
    _stt, _end = time_step_local_offset, time_step_local_offset + len(token_group)-1
    if len(token_group) == 1:
        frame_stt, frame_end = frame_inds_seq[_stt], frame_inds_seq[_stt] + 1
    else:
        frame_stt, frame_end = frame_inds_seq[_stt], frame_inds_seq[_end]
        
    # Edge Cases: Sometimes, repeated token indexs can lead to incorrect frame and speaker assignment.
    if frame_stt == frame_end:
        if frame_stt >= diar_pred_out_stream.shape[1] - 1:
            frame_stt, frame_end = (diar_pred_out_stream.shape[1] - 1, diar_pred_out_stream.shape[1])
        else:
            frame_end = frame_stt + 1
        
    speaker_sigmoid = diar_pred_out_stream[0, frame_stt:frame_end, :].mean(dim=0)
    # speaker_softmax = speaker_sigmoid / speaker_sigmoid.sum()
    speaker_softmax = torch.softmax(speaker_sigmoid, dim=0)
    speaker_softmax = speaker_softmax.cpu()
    stt_sec, end_sec = frame_stt * frame_len, frame_end * frame_len
    spk_id = speaker_softmax.argmax().item()
    word_dict = {"word": word,
                'frame_stt': frame_stt,
                'frame_end': frame_end,
                'start_time': round(stt_sec, 3), 
                'end_time': round(end_sec, 3), 
                'speaker': f"speaker_{spk_id}",
                'speaker_softmax': speaker_softmax} 
    return word_dict                 
        
def get_frame_and_words(cfg, tokenizer, step_num, diar_pred_out_stream, previous_hypotheses, word_and_ts_seq, frame_len=0.08):
    current_frame_range = [step_num * previous_hypotheses[0].length.item(), (step_num + 1) * previous_hypotheses[0].length.item()]
    offset = current_frame_range[0]
    word_seq = previous_hypotheses[0].text.split()
    new_words = word_seq[word_and_ts_seq["offset_count"]:]
    frame_inds_seq = (torch.tensor(previous_hypotheses[0].timestep) + offset).tolist()
    new_token_group = tokenizer.text_to_tokens(new_words)
    new_tokens = list(itertools.chain(*new_token_group))
    frame_inds_seq = fix_frame_time_step(new_tokens, new_words, frame_inds_seq)
    min_len = min(len(new_words), len(frame_inds_seq))
    for idx in range(min_len):
        word_and_ts_seq["token_frame_index"].append((new_tokens[idx], frame_inds_seq[idx]))
        word_and_ts_seq["offset_count"] += 1
    
    time_step_local_offset = 0 
    for token_group, word in zip(new_token_group, new_words):
        word_dict = get_word_dict_content(word=word,
                                          diar_pred_out_stream=diar_pred_out_stream,
                                          token_group=token_group,
                                          frame_inds_seq=frame_inds_seq,
                                          time_step_local_offset=time_step_local_offset,
                                          frame_len=frame_len
                                          )
        # Count the number of speakers in the word window
        word_and_ts_seq["speaker_count_buffer"].append(word_dict["speaker"])
        if len(word_and_ts_seq["speaker_count_buffer"]) > cfg.word_window:
            word_and_ts_seq["speaker_count_buffer"].pop(0)
        word_and_ts_seq["speaker_count"] = len(set(word_and_ts_seq["speaker_count_buffer"]))
            
        word_and_ts_seq["words"].append(word_dict)
        time_step_local_offset += len(token_group)                                        
    return word_and_ts_seq 

def perform_streaming(
    cfg, asr_model, diar_model, bsd_spk, streaming_buffer, compare_vs_offline=False, debug_mode=False, pad_and_drop_preencoded=False
):
    batch_size = len(streaming_buffer.streams_length)
    if compare_vs_offline:
        # would pass the whole audio at once through the model like offline mode in order to compare the results with the stremaing mode
        # the output of the model in the offline and streaming mode should be exactly the same
        with torch.inference_mode():
            with autocast:
                processed_signal, processed_signal_length = streaming_buffer.get_all_audios()
                with torch.no_grad():
                    (
                        pred_out_offline,
                        transcribed_texts,
                        cache_last_channel_next,
                        cache_last_time_next,
                        cache_last_channel_len,
                        best_hyp,
                    ) = asr_model.conformer_stream_step(
                        processed_signal=processed_signal,
                        processed_signal_length=processed_signal_length,
                        return_transcription=True,
                    )
        final_offline_tran = extract_transcriptions(transcribed_texts)
        logging.info(f" Final offline transcriptions:   {final_offline_tran}")
    else:
        final_offline_tran = None

    cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
        batch_size=batch_size
    )

    previous_hypotheses = None
    streaming_buffer_iter = iter(streaming_buffer)
    asr_pred_out_stream, diar_pred_out_stream  = None, None
    mem_last_time, fifo_last_time = None, None
    left_offset, right_offset = 0, 0
    word_and_ts_seq = {"words": [], "token_frame_index": [], "offset_count": 0,
                        "status": "success", 
                        "sentences": None, 
                        "speaker_count": None,
                        "transcription": None,
                        "speaker_count_buffer": [],
                        }
    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
        with torch.inference_mode():
            with autocast:
                # keep_all_outputs needs to be True for the last step of streaming when model is trained with att_context_style=regular
                # otherwise the last outputs would get dropped

                with torch.no_grad():
                    (
                        asr_pred_out_stream,
                        transcribed_texts,
                        cache_last_channel,
                        cache_last_time,
                        cache_last_channel_len,
                        previous_hypotheses,
                    ) = asr_model.conformer_stream_step(
                        processed_signal=chunk_audio,
                        processed_signal_length=chunk_lengths,
                        cache_last_channel=cache_last_channel,
                        cache_last_time=cache_last_time,
                        cache_last_channel_len=cache_last_channel_len,
                        keep_all_outputs=streaming_buffer.is_buffer_empty(),
                        previous_hypotheses=previous_hypotheses,
                        previous_pred_out=asr_pred_out_stream,
                        drop_extra_pre_encoded=calc_drop_extra_pre_encoded(
                            asr_model, step_num, pad_and_drop_preencoded
                        ),
                        return_transcription=True,
                    )

                    if step_num > 0:
                        left_offset = 8
                        chunk_audio = chunk_audio[..., 1:]
                        chunk_lengths -= 1
                    

                    (
                        diar_pred_out_stream,
                        mem_last_time,
                        fifo_last_time,
                    ) = diar_model.forward_streaming_step(
                        processed_signal=chunk_audio.transpose(1, 2),
                        processed_signal_length=chunk_lengths,
                        mem_last_time=mem_last_time,
                        fifo_last_time=fifo_last_time,
                        previous_pred_out=diar_pred_out_stream,
                        left_offset=left_offset,
                        right_offset=right_offset,
                    )
                    # Get the word-level dictionaries for each word in the chunk
                    word_and_ts_seq = get_frame_and_words(cfg=cfg,
                                                          tokenizer=asr_model.tokenizer,
                                                          step_num=step_num, 
                                                          diar_pred_out_stream=diar_pred_out_stream,
                                                          previous_hypotheses=previous_hypotheses, 
                                                          word_and_ts_seq=word_and_ts_seq) 
                    if cfg.beam_search_enabled: 
                        if len(word_and_ts_seq["words"]) > cfg.word_window:
                            extra_len = len(word_and_ts_seq["words"]) - cfg.word_window
                            words = word_and_ts_seq["words"][extra_len:]
                            bsd_words = bsd_spk.beam_search_diarization_single(word_dict_seq_list=words, speaker_count=word_and_ts_seq["speaker_count"])
                            word_and_ts_seq["words"] = word_and_ts_seq["words"][:extra_len] + bsd_words
                        else:
                            word_and_ts_seq["words"] = bsd_spk.beam_search_diarization_single(word_dict_seq_list=word_and_ts_seq["words"], speaker_count=word_and_ts_seq["speaker_count"])
                    word_and_ts_seq = get_sentences_values(session_trans_dict=word_and_ts_seq)
                    string_out = print_sentences(sentences=word_and_ts_seq["sentences"], color_palette=get_color_palette(), params=cfg) 
                    write_txt(f'{cfg.print_path}', string_out.strip())
                    logging.info(f"mem: {mem_last_time.shape}, fifo: {fifo_last_time.shape}, pred: {diar_pred_out_stream.shape}")
        if debug_mode:
            logging.info(f"Streaming transcriptions: {extract_transcriptions(transcribed_texts)}")

    final_streaming_tran = extract_transcriptions(transcribed_texts)
    logging.info(f"Final streaming transcriptions: {final_streaming_tran}")

    if compare_vs_offline:
        # calculates and report the differences between the predictions of the model in offline mode vs streaming mode
        # Normally they should be exactly the same predictions for streaming models
        pred_out_stream_cat = torch.cat(asr_pred_out_stream)
        pred_out_offline_cat = torch.cat(pred_out_offline)
        if pred_out_stream_cat.size() == pred_out_offline_cat.size():
            diff_num = torch.sum(pred_out_stream_cat != pred_out_offline_cat).cpu().numpy()
            logging.info(
                f"Found {diff_num} differences in the outputs of the model in streaming mode vs offline mode."
            )
        else:
            logging.info(
                f"The shape of the outputs of the model in streaming mode ({pred_out_stream_cat.size()}) is different from offline mode ({pred_out_offline_cat.size()})."
            )

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
    infer_audio_rttm_dict = get_audio_rttm_map(cfg.manifest_file)
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
    arpa_model = kenlm.Model(cfg.arpa_language_model)
    bsd_spk = SpeakerTaggingBeamSearchDecoder(loaded_kenlm_model=arpa_model, cfg=cfg)
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
            compare_vs_offline=args.compare_vs_offline,
            pad_and_drop_preencoded=args.pad_and_drop_preencoded,
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

        logging.info(f"Loaded {len(samples)} from the manifest at {args.manifest_file}.")

        start_time = time.time()
        for sample_idx, sample in enumerate(samples):
            processed_signal, processed_signal_length, stream_id = streaming_buffer.append_audio_file(
                sample['audio_filepath'], stream_id=-1
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
                    compare_vs_offline=args.compare_vs_offline,
                    debug_mode=args.debug_mode,
                    pad_and_drop_preencoded=args.pad_and_drop_preencoded,
                )
                all_streaming_tran.extend(streaming_tran)
                if args.compare_vs_offline:
                    all_offline_tran.extend(offline_tran)
                streaming_buffer.reset_buffer()

        if args.compare_vs_offline and len(all_refs_text) == len(all_offline_tran):
            offline_wer = word_error_rate(hypotheses=all_offline_tran, references=all_refs_text)
            logging.info(f"WER% of offline mode: {round(offline_wer * 100, 2)}")
        if len(all_refs_text) == len(all_streaming_tran):
            streaming_wer = word_error_rate(hypotheses=all_streaming_tran, references=all_refs_text)
            logging.info(f"WER% of streaming mode: {round(streaming_wer*100, 2)}")

        end_time = time.time()
        logging.info(f"The whole streaming process took: {round(end_time - start_time, 2)}s")

        # stores the results including the transcriptions of the streaming inference in a json file
        if args.output_path is not None and len(all_refs_text) == len(all_streaming_tran):
            fname = (
                "streaming_out_"
                + os.path.splitext(os.path.basename(args.asr_model))[0]
                + "_"
                + os.path.splitext(os.path.basename(args.manifest_file))[0]
                + ".json"
            )

            hyp_json = os.path.join(args.output_path, fname)
            os.makedirs(args.output_path, exist_ok=True)
            with open(hyp_json, "w") as out_f:
                for i, hyp in enumerate(all_streaming_tran):
                    record = {
                        "pred_text": hyp,
                        "text": all_refs_text[i],
                        "wer": round(word_error_rate(hypotheses=[hyp], references=[all_refs_text[i]]) * 100, 2),
                    }
                    out_f.write(json.dumps(record) + '\n')

if __name__ == '__main__':
    main()
