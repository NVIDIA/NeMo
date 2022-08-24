#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
# import pyaudio as pa
import argparse
import os
import nemo
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
from nemo.collections.asr.parts.utils.diarization_utils import ASR_DIAR_OFFLINE, ASR_DIAR_ONLINE, write_txt
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
    getMultiScaleCosAffinityMatrix,
    getTempInterpolMultiScaleCosAffinityMatrix
)
from nemo.collections.asr.parts.utils.nmesc_clustering import COSclustering
from nemo.collections.asr.parts.utils.streaming_utils import get_samples, AudioFeatureIterator
# , FrameBatchVAD

from nemo.core.config import hydra_runner
from nemo.utils import logging
import hydra
from typing import List, Optional, Dict
from omegaconf import DictConfig, OmegaConf
from omegaconf.listconfig import ListConfig
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

from nemo.utils import logging

# from pyctcdecode import build_ctcdecoder

# GRadio input
import gradio as gr
import time
import librosa
import soundfile
import tempfile
import os
import uuid
import numpy as np
seed_everything(42)

try:
    from pyctcdecode import build_ctcdecoder

    PYCTCDECODE = True
except ImportError:
    PYCTCDECODE = False

class ASR_DIAR_ONLINE_DEMO(ASR_DIAR_ONLINE):
    def __init__(self, cfg):
        super().__init__(cfg)

    def realign_speaker_labels(self, words, memory_cluster_labels):
        """
        Realign
        """
        cluster_labels = torch.tensor(memory_cluster_labels)
        words_mat = np.array(words)
        scale_idx = 0
        sent_turn = [ self.check_t(x) for x in words]
        stt_idx = torch.where( torch.cat((torch.tensor([False]), (torch.tensor(sent_turn) == True)))[:-1] == True )[0]
        end_idx = torch.where( (torch.tensor(sent_turn) == True) == True )[0]
        for idx in range(stt_idx.shape[0]-1):
            range_spk_labels = cluster_labels[stt_idx[idx]:end_idx[idx+1]+1].tolist()
            range_words = words[stt_idx[idx]:end_idx[idx+1]+1]
            # Ignore one-word speaker turns.
            if stt_idx[idx] == end_idx[idx+1] or len(set(range_spk_labels)) <= 1:
                continue
            range_spk_labels = torch.tensor(range_spk_labels)
            fixed_range_labels = torch.mode(range_spk_labels)[0] * torch.ones_like(range_spk_labels)
            spk_label_count = torch.bincount(range_spk_labels)
            # Limit the change of speakers to prevent excessive change from realigning.
            if torch.min(spk_label_count) < self.max_lm_realigning_words:
                memory_cluster_labels[stt_idx[idx]: end_idx[idx+1]+1] = fixed_range_labels.tolist()
        self.fixed_speaker_labels  = copy.deepcopy([ f"speaker_{x}" for x in memory_cluster_labels])
    
    def _fix_speaker_label_per_word(self, words, word_ts_list, pred_diar_labels):
        word_speaker_labels = []
        # assert len(word_ts_list) == len(words) == len(pred_diar_labels)
        start_point, end_point, speaker = pred_diar_labels[0].split()
        idx = 0
        for j, word_ts_stt_end in enumerate(word_ts_list):
            word_pos = self.get_word_timestamp_anchor(word_ts_stt_end)
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
    
    def punctuate_words(self, words):
        """
        Punctuate the transcribe word based on the loaded punctuation model.
        """
        if len(words) == 0:
            return []
        elif self.punctuation_model is not None:
            words = self.punctuation_model.add_punctuation_capitalization([' '.join(words)])[0].split()
            for idx in range(1, len(words)):
                if any([ x in words[idx-1] for x in [".", "?"] ]):
                    words[idx] = words[idx].capitalize()
            # words = [w.replace(",", "") for w in words]
            return words
    
    def check_t(self, x):
        return any([ _str in x for _str in ['.', ',', '?'] ] )
    
    def update_word_and_word_ts(self, words, word_timetamps):
        """
        Stitch the existing word sequence in the buffer with the new word sequence.
        """
        update_margin =  -1* float(self.frame_len * self.word_update_margin)
        if len(self.word_seq) == 0:
            if self.punctuation_model:
                words = self.punctuate_words(words)
            self.word_seq.extend(words)
            self.word_ts_seq.extend(word_timetamps)

        elif len(words) > 0:
            # Find the first word that starts after frame_start point in state buffer self.word_seq
            before_frame_start_old = torch.tensor(self.word_ts_seq)[:,0] < self.frame_start + update_margin
            if all(before_frame_start_old):
                old_end = len(self.word_seq)
            else:
                old_end = torch.where(before_frame_start_old == False)[0][0].item()
            
            # Find the first word that starts after frame_start point in incoming words
            before_frame_start_new = torch.tensor(word_timetamps)[:,0] < self.frame_start + update_margin
            if all(before_frame_start_new):
                new_stt = len(word_timetamps)
            else:
                new_stt = torch.where(before_frame_start_new == False)[0][0].item()
            
            del self.word_seq[old_end:]
            del self.word_ts_seq[old_end:]
            if self.punctuation_model_path:
                punc_margin = len(words[:new_stt])
                punc_stt = max(new_stt-punc_margin, 0)
                punctuated_words = self.punctuate_words(words[punc_stt:])[punc_margin:]
                self.word_seq.extend(punctuated_words)
            else:
                self.word_seq.extend(words[new_stt:])
            self.word_ts_seq.extend(word_timetamps[new_stt:])
    
    def streaming_step(self, sample_audio):
        loop_start_time = time.time()
        assert len(sample_audio) == int(self.sample_rate * self.frame_len)
        words, timestamps, diar_hyp = self.run_step(sample_audio)
        if diar_hyp != []:
            assert len(words) == len(timestamps)
            self.update_word_and_word_ts(words, timestamps)
            if self.punctuation_model_path:
                self._fix_speaker_label_per_word(self.word_seq, self.word_ts_seq, diar_hyp)
                self.string_out = self._get_speaker_label_per_word(self.word_seq, self.word_ts_seq, diar_hyp)
                write_txt(f"{self.diar._out_dir}/print_script.sh", self.string_out)
            else:
                total_riva_dict = {}
                word_dict_seq_list = self.get_word_dict_seq_list(diar_hyp, self.word_seq, self.word_ts_seq, self.word_ts_seq)
                sentences = self.make_json_output(self.diar.uniq_id, diar_hyp, word_dict_seq_list, total_riva_dict, write_files=False)
                self.string_out = self.print_sentences(sentences, self.params)
                write_txt(f"{self.diar._out_dir}/print_script.sh", self.string_out.strip())
        self.simulate_delay(loop_start_time) 
    

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

    def print_word_colored(self, string_out, word, params, first_word=False):
        word = word.strip()
        if first_word:
            space = ""
            word = word.capitalize() if self.capitalize_first_word else word
        else:
            space = " "
        word = word.replace(",", "").replace(".","").replace("?","").lower()
        return string_out + space +  word

    def punctuate_words(self, words):
        """
        Punctuate the transcribe word based on the loaded punctuation model.
        """
        if len(words) == 0:
            return []
        elif self.punctuation_model is not None:
            words = self.punctuation_model.add_punctuation_capitalization([' '.join(words)])[0].split()
            for idx in range(1, len(words)):
                if any([ x in words[idx-1] for x in [".", "?"] ]):
                    words[idx] = words[idx].capitalize()
            # words = [w.replace(",", "") for w in words]
            return words

    def _get_speaker_label_per_word(self, words, word_ts_list, pred_diar_labels):
        if len(words) == 0:
            return ''
        params = self.params
        start_point, end_point = word_ts_list[0]
        speaker = self.fixed_speaker_labels[0]
        old_speaker = speaker
        idx, string_out = 0, ''
        string_out = self.print_time_colored(string_out, speaker, start_point, end_point, params, replace_time=False, space='')
        if len(words) == 0:
            return string_out
        for j, word_ts_stt_end in enumerate(word_ts_list):
            old_speaker = speaker
            if old_speaker == self.fixed_speaker_labels[j]:
                speaker = self.fixed_speaker_labels[j]
                start_point, end_point = word_ts_stt_end
                string_out = self.print_time_colored(string_out, speaker, start_point, end_point, params, replace_time=True, space='')
                string_out = self.print_word_colored(string_out, words[j], params, first_word=False)
                old_speaker = self.fixed_speaker_labels[j]
            else:
                idx += 1
                idx = min(idx, len(pred_diar_labels)-1)
                start_point, end_point = word_ts_stt_end
                speaker = self.fixed_speaker_labels[j]
                if speaker != old_speaker:
                    last_word = string_out.split(" ")[-1]
                    if "," in string_out[-1]:
                        string_out = string_out[:-1] + string_out[-1].replace(",",".")
                    string_out = self.print_time_colored(string_out, speaker, start_point, end_point, params, space='')
                else:
                    string_out = self.print_time_colored(string_out, speaker, start_point, end_point, params, replace_time=True, space='')
                string_out = self.print_word_colored(string_out, words[j], params, first_word=True)
        
        if self.rttm_file_path and len(words) > 0:
            string_out = self.print_online_DER_info(self.diar.uniq_id, string_out, pred_diar_labels, params)
        logging.info(
            "Streaming Diar [{}][frame-  {}th  ]:".format(
                self.diar.uniq_id, self.frame_index
            )
        )
        return string_out 

@hydra_runner(config_path="conf", config_name="online_diarization_with_asr.yaml")
def main(cfg):
    asr_diar = ASR_DIAR_ONLINE(cfg=cfg)
    # asr_diar = ASR_DIAR_ONLINE_DEMO(cfg=cfg)
    diar = asr_diar.diar

    if cfg.diarizer.asr.parameters.streaming_simulation:
        diar.uniq_id = cfg.diarizer.simulation_uniq_id
        asr_diar.get_audio_rttm_map(diar.uniq_id)
        diar.single_audio_file_path = diar.AUDIO_RTTM_MAP[diar.uniq_id]['audio_filepath']
        diar.rttm_file_path = diar.AUDIO_RTTM_MAP[diar.uniq_id]['rttm_filepath']
        asr_diar.rttm_file_path = diar.rttm_file_path
    else:
        diar.rttm_file_path = None

    diar._init_segment_variables()
    diar.device = asr_diar.device
    write_txt(f"{diar._out_dir}/print_script.sh", "")
    
    if cfg.diarizer.asr.parameters.streaming_simulation:
        samplerate, sdata = wavfile.read(diar.single_audio_file_path)
        if  diar.AUDIO_RTTM_MAP[diar.uniq_id]['offset'] and  diar.AUDIO_RTTM_MAP[diar.uniq_id]['duration']:
            sdata = sdata[int(samplerate*diar.AUDIO_RTTM_MAP[diar.uniq_id]['offset']):int(samplerate*diar.AUDIO_RTTM_MAP[diar.uniq_id]['offset']+samplerate*diar.AUDIO_RTTM_MAP[diar.uniq_id]['duration'])]

        for index in range(int(np.floor(sdata.shape[0]/asr_diar.n_frame_len))):
            asr_diar.buffer_counter = index
            sample_audio = sdata[asr_diar.CHUNK_SIZE*(asr_diar.buffer_counter):asr_diar.CHUNK_SIZE*(asr_diar.buffer_counter+1)]
            asr_diar.streaming_step(sample_audio)
    else:
        isTorch = torch.cuda.is_available()
        iface = gr.Interface(
            fn=asr_diar.audio_queue_launcher,
            inputs=[
                gr.Audio(source="microphone", type="numpy", streaming=True), 
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

if __name__ == "__main__":
    main()
