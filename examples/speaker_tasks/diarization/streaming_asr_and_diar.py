#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import pyaudio as pa
import argparse
import os
import nemo
import nemo.collections.asr as nemo_asr
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
from nemo.collections.asr.models import OnlineDiarizer
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
from nemo.collections.asr.parts.utils.streaming_utils import get_samples, AudioFeatureIterator, FrameBatchVAD

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
import nemo.collections.asr as nemo_asr

from nemo.utils import logging

# from pyctcdecode import build_ctcdecoder

# GRadio input
import gradio as gr
import time
import librosa
import soundfile
import nemo.collections.asr as nemo_asr
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
    def __init__(self, diar, cfg):
        super().__init__(diar, cfg)

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
    
    def check_t(self, x):
        return any([ _str in x for _str in ['.', ',', '?'] ] )
    
    def callback_sim(self, sample_audio):
        start_time = time.time()
        assert len(sample_audio) == int(self.sample_rate * self.frame_len)
        words, timestamps, diar_hyp = self.transcribe(sample_audio)
        if diar_hyp != []:
            assert len(words) == len(timestamps)
            self._update_word_and_word_ts(words, timestamps)
            if self.punctuation_model_path:
                self._fix_speaker_label_per_word(self.word_seq, self.word_ts_seq, diar_hyp)
                self.string_out = self._get_speaker_label_per_word(self.word_seq, self.word_ts_seq, diar_hyp)
            else:
                word_dict_seq_list = self.get_word_dict_seq_list(diar_hyp, self.word_seq, self.word_ts_seq, self.word_ts_seq)
                sentences = self.make_json_output(self.diar.uniq_id, diar_hyp, word_dict_seq_list, total_riva_dict, write_files=False)
                self.string_out = self.print_sentences(sentences, self.params)
            write_txt(f"{self.diar._out_dir}/print_script.sh", self.string_out.strip())

    def _get_speaker_label_per_word(self, words, word_ts_list, pred_diar_labels):
        if len(words) == 0:
            return ''
        params = self.params
        start_point, end_point = word_ts_list[0]
        speaker = self.fixed_speaker_labels[0]
        old_speaker = speaker
        idx, string_out = 0, ''
        string_out = self.print_time_colored(string_out, speaker, start_point, end_point, params)
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
    diar = OnlineDiarizer(cfg)
    asr_diar = ASR_DIAR_ONLINE(diar, cfg=cfg.diarizer)

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
        for index in range(int(np.floor(sdata.shape[0]/asr_diar.n_frame_len))):
            asr_diar.buffer_counter = index
            sample_audio = sdata[asr_diar.CHUNK_SIZE*(asr_diar.buffer_counter):asr_diar.CHUNK_SIZE*(asr_diar.buffer_counter+1)]
            asr_diar.callback_sim(sample_audio)
    else:
        isTorch = torch.cuda.is_available()
        iface = gr.Interface(
            fn=asr_diar.audio_queue_launcher,
            inputs=[
                gr.inputs.Audio(source="microphone", type='filepath'), 
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
