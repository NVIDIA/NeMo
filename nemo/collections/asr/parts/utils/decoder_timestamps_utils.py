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
import json
import math
import os
import sys
from collections import OrderedDict as od
from omegaconf import OmegaConf, open_dict
from datetime import datetime
from typing import Union, Optional, List, Type, Tuple, Dict
import librosa
import tempfile
import diff_match_patch
import wget
from tqdm.auto import tqdm
import numpy as np
import soundfile as sf
import torch
import nemo.collections.asr as nemo_asr

from nemo.collections.asr.metrics.wer import WER, word_error_rate
from nemo.collections.asr.metrics.wer_bpe import WERBPE
from nemo.collections.asr.metrics.rnnt_wer_bpe import RNNTBPEWER
from nemo.collections.asr.models import ClusteringDiarizer, EncDecCTCModel, EncDecCTCModelBPE, EncDecRNNTBPEModel
from nemo.collections.asr.parts.utils.speaker_utils import (
    get_uniqname_from_filepath,
    labels_to_rttmfile,
    rttm_to_labels,
    write_rttm2manifest,
)
from nemo.collections.asr.parts.utils.streaming_utils import AudioFeatureIterator, FrameBatchASR
from nemo.collections.asr.parts.utils.speaker_utils import audio_rttm_map
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo.utils import logging

from nemo.core.config import hydra_runner
from nemo.utils import logging

__all__ = ['ASR_TIMESTAMPS']

try:
    from pyctcdecode import build_ctcdecoder
except:
    logging.info("You must install kenlm and pyctcdecode to use language models")


def get_ts_from_decoded_prediction(time_stride, decoded_char_list, hypothesis, char_ts):
    # decoded_char_list = self.tokenizer.ids_to_tokens(decoded_prediction)
    stt_idx, end_idx = 0, len(decoded_char_list) - 1
    stt_ch_idx, end_ch_idx = 0, 0
    space = '▁'
    word_ts, word_seq = [], []
    word_open_flag = False
    hyp_seq_list = hypothesis.split()
    for idx, ch in enumerate(decoded_char_list):
        if idx != end_idx and ( ch == space and space in decoded_char_list[idx + 1]):
            continue

        if space in ch:
            _stt = char_ts[idx]
            stt_ch_idx = idx
            word_open_flag = True
            temp_end_idx = idx
        elif ch != '':
            temp_end_idx = idx
        
        if word_open_flag and (idx == end_idx or space in decoded_char_list[idx + 1]):
            _end = round(char_ts[temp_end_idx] + time_stride, 2)
            stitched_word = ''.join(decoded_char_list[stt_ch_idx : idx + 1]).replace(space, '')
            word_seq.append(stitched_word)
            word_ts.append([_stt, _end])
            ct = len(word_ts)-1
            word_open_flag = False
    assert len(word_seq) == len(word_ts) == len(hyp_seq_list), "Hypothesis does not match word time stamp."
    return word_seq, word_ts

class WERBPE_TS(WERBPE):
    """
    This is WERBPE_TS class that is modified for generating word_timestamps with logits.
    The functions in WER class is modified to save the word_timestamps whenever BPE token
    is being saved into a list.
    This class is designed to support ASR models based on CTC and BPE.
    Please refer to the definition of WERBPE class for more information.
    """

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        batch_dim_index=0,
        use_cer=False,
        ctc_decode=True,
        log_prediction=True,
        dist_sync_on_step=False,
    ):

        super().__init__(tokenizer, batch_dim_index, use_cer, ctc_decode, log_prediction, dist_sync_on_step)

    def ctc_decoder_predictions_tensor_with_ts(
        self, time_stride, predictions: torch.Tensor, predictions_len: torch.Tensor = None
    ) -> List[str]:
        hypotheses, timestamps, word_timestamps = [], [], []
        # '⁇' string should removed since it causes error on string split.
        unk = '⁇'
        prediction_cpu_tensor = predictions.long().cpu()
        # iterate over batch
        self.time_stride = time_stride
        for ind in range(prediction_cpu_tensor.shape[self.batch_dim_index]):
            prediction = prediction_cpu_tensor[ind].detach().numpy().tolist()
            if predictions_len is not None:
                prediction = prediction[: predictions_len[ind]]
            # CTC decoding procedure
            decoded_prediction, char_ts, timestamp_list = [], [], []
            previous = self.blank_id
            for pdx, p in enumerate(prediction):
                if (p != previous or previous == self.blank_id) and p != self.blank_id:
                    decoded_prediction.append(p)
                    char_ts.append(round(pdx * self.time_stride, 2))
                    timestamp_list.append(round(pdx * self.time_stride, 2))

                previous = p

            hypothesis = self.decode_tokens_to_str_with_ts(decoded_prediction)
            hypothesis = hypothesis.replace(unk, '')
            word_ts = self.get_ts_from_decoded_prediction(decoded_prediction, hypothesis, char_ts)

            hypotheses.append(hypothesis)
            timestamps.append(timestamp_list)
            word_timestamps.append(word_ts)
        return hypotheses, timestamps, word_timestamps

    def decode_tokens_to_str_with_ts(self, tokens: List[int]) -> str:
        hypothesis = self.tokenizer.ids_to_text(tokens)
        return hypothesis

    def decode_ids_to_tokens_with_ts(self, tokens: List[int]) -> List[str]:
        token_list = self.tokenizer.ids_to_tokens(tokens)
        return token_list

    def get_ts_from_decoded_prediction(self, decoded_prediction, hypothesis, char_ts):
        decoded_char_list = self.tokenizer.ids_to_tokens(decoded_prediction)
        stt_idx, end_idx = 0, len(decoded_char_list) - 1
        stt_ch_idx, end_ch_idx = 0, 0
        space = '▁'
        word_ts, word_seq = [], []
        word_open_flag = False
        for idx, ch in enumerate(decoded_char_list):
            if idx != end_idx and (space == ch and space in decoded_char_list[idx + 1]):
                continue

            if (idx == stt_idx or space == decoded_char_list[idx - 1] or (space in ch and len(ch) > 1)) and (
                ch != space
            ):
                _stt = char_ts[idx]
                stt_ch_idx = idx
                word_open_flag = True

            if word_open_flag and ch != space and (idx == end_idx or space in decoded_char_list[idx + 1]):
                _end = round(char_ts[idx] + self.time_stride, 2)
                end_ch_idx = idx
                word_open_flag = False
                word_ts.append([_stt, _end])
                stitched_word = ''.join(decoded_char_list[stt_ch_idx : end_ch_idx + 1]).replace(space, '')
                word_seq.append(stitched_word)
        assert len(word_ts) == len(hypothesis.split()), "Hypothesis does not match word time stamp."
        return word_ts


class WER_TS(WER):
    """
    This is WER class that is modified for generating timestamps with logits.
    The functions in WER class is modified to save the timestamps whenever character
    is being saved into a list.
    This class is designed to support ASR models based on CTC and Character-level tokens.
    Please refer to the definition of WER class for more information.
    """

    def __init__(
        self,
        vocabulary,
        batch_dim_index=0,
        use_cer=False,
        ctc_decode=True,
        log_prediction=True,
        dist_sync_on_step=False,
    ):
        super().__init__(vocabulary, batch_dim_index, use_cer, ctc_decode, log_prediction, dist_sync_on_step)

    def decode_tokens_to_str_with_ts(self, tokens: List[int], timestamps: List[int]) -> str:
        """
        Accepts frame-level tokens and timestamp list and collects the timestamps for
        start and end of each word.
        """
        token_list, timestamp_list = self.decode_ids_to_tokens_with_ts(tokens, timestamps)
        hypothesis = ''.join(self.decode_ids_to_tokens(tokens))
        return hypothesis, timestamp_list

    def decode_ids_to_tokens_with_ts(self, tokens: List[int], timestamps: List[int]) -> List[str]:
        token_list, timestamp_list = [], []
        for i, c in enumerate(tokens):
            if c != self.blank_id:
                token_list.append(self.labels_map[c])
                timestamp_list.append(timestamps[i])
        return token_list, timestamp_list

    def ctc_decoder_predictions_tensor_with_ts(
        self, predictions: torch.Tensor, predictions_len: torch.Tensor = None,
    ) -> List[str]:
        """
        A shortened version of the original function ctc_decoder_predictions_tensor().
        Replaced decode_tokens_to_str() function with decode_tokens_to_str_with_ts().
        """
        hypotheses, timestamps = [], []
        prediction_cpu_tensor = predictions.long().cpu()
        for ind in range(prediction_cpu_tensor.shape[self.batch_dim_index]):
            prediction = prediction_cpu_tensor[ind].detach().numpy().tolist()
            if predictions_len is not None:
                prediction = prediction[: predictions_len[ind]]

            # CTC decoding procedure with timestamps
            decoded_prediction, decoded_timing_list = [], []
            previous = self.blank_id
            for pdx, p in enumerate(prediction):
                if (p != previous or previous == self.blank_id) and p != self.blank_id:
                    decoded_prediction.append(p)
                    decoded_timing_list.append(pdx)
                previous = p

            text, timestamp_list = self.decode_tokens_to_str_with_ts(decoded_prediction, decoded_timing_list)
            hypotheses.append(text)
            timestamps.append(timestamp_list)

        return hypotheses, timestamps

def get_wer_feat_logit(audio_file_path, asr, frame_len, tokens_per_chunk, delay, model_stride_in_secs):
    """
    Create a preprocessor to convert audio samples into raw features,
    Normalization will be done per buffer in frame_bufferer.
    """

    hyps = []
    tokens_list = []
    logprobs_list = []
    # for idx, audio_file_path in enumerate(audio_file_list):
    asr.reset()
    # samples = 
    asr.read_audio_file_and_return(audio_file_path, delay, model_stride_in_secs)
    hyp, tokens, log_prob = asr.transcribe_with_ts(tokens_per_chunk, delay)
    return hyp, tokens, log_prob


def get_samples(audio_file, target_sr=16000):
    """Read samples from the given audio_file path.
    """
    with sf.SoundFile(audio_file, 'r') as f:
        dtype = 'int16'
        sample_rate = f.samplerate
        samples = f.read(dtype=dtype)
        if sample_rate != target_sr:
            samples = librosa.core.resample(samples, sample_rate, target_sr)
        samples = samples.astype('float32') / 32768
        samples = samples.transpose()
        del f
    return samples

class FrameBatchASR_Logits(FrameBatchASR):
    """
    A class for streaming frame-based ASR.
    Inherits from FrameBatchASR and adds new capability of returning the logit output.
    Please refer to FrameBatchASR for more detailed information.
    """

    def __init__(self, asr_model, frame_len=1.6, total_buffer=4.0, batch_size=4):
        super().__init__(asr_model, frame_len, total_buffer, batch_size)
        self.all_logprobs = []
    
    def clear_buffer(self):
        self.all_logprobs = []
        self.all_preds = []

    def read_audio_file_and_return(self, audio_filepath: str, delay, model_stride_in_secs):
        samples = get_samples(audio_filepath)
        samples = np.pad(samples, (0, int(delay * model_stride_in_secs * self.asr_model._cfg.sample_rate)))
        frame_reader = AudioFeatureIterator(samples, self.frame_len, self.raw_preprocessor, self.asr_model.device)
        self.set_frame_reader(frame_reader)
        del samples
    
    @torch.no_grad()
    def _get_batch_preds(self):
        device = self.asr_model.device
        for batch in iter(self.data_loader):

            feat_signal, feat_signal_len = batch
            feat_signal, feat_signal_len = feat_signal.to(device), feat_signal_len.to(device)
            log_probs, encoded_len, predictions = self.asr_model(
                processed_signal=feat_signal, processed_signal_length=feat_signal_len
            )
            preds = torch.unbind(predictions)
            for pred in preds:
                self.all_preds.append(pred.cpu().numpy())
            log_probs_tup = torch.unbind(log_probs)
            for log_prob in log_probs_tup:
                self.all_logprobs.append(log_prob)
            del encoded_len
            del predictions

    def transcribe_with_ts(
        self, tokens_per_chunk: int, delay: int,
    ):
        self.infer_logits()
        self.unmerged = []
        self.part_logprobs = []
        for idx, pred in enumerate(self.all_preds):
            decoded = pred.tolist()
            _stt, _end = len(decoded) - 1 - delay, len(decoded) - 1 - delay + tokens_per_chunk
            self.unmerged += decoded[len(decoded) - 1 - delay:len(decoded) - 1 - delay + tokens_per_chunk]
            self.part_logprobs.append(self.all_logprobs[idx][_stt:_end, :])
        self.unmerged_logprobs = torch.cat(self.part_logprobs, 0)
        assert len(self.unmerged) == self.unmerged_logprobs.shape[0], "Unmerged decoded result and log prob lengths are different."
        return self.greedy_merge(self.unmerged), self.unmerged, self.unmerged_logprobs



class ASR_TIMESTAMPS:
    """
    A Class designed for performing ASR and diarization together.
    """

    def __init__(self, **cfg_diarizer):
        self.manifest_filepath = cfg_diarizer['manifest_filepath']
        self.params = cfg_diarizer['asr']['asr_parameters']
        self.ctc_decoder_params = cfg_diarizer['asr']['ctc_decoder_parameters']
        self.ASR_model_name = cfg_diarizer['asr']['model_path']
        self.nonspeech_threshold = self.params['asr_based_vad_threshold']
        self.root_path = None
        self.run_ASR = None
        self.encdec_class = None
        self.normalizer = normalizer = Normalizer(input_case='lower_cased', lang='en')
        self.AUDIO_RTTM_MAP = audio_rttm_map(self.manifest_filepath)
        self.audio_file_list = [value['audio_filepath'] for _, value in self.AUDIO_RTTM_MAP.items()]

    def set_asr_model(self, ASR_model_name):
        """
        Setup the parameters for the given ASR model
        Currently, the following models are supported:
            stt_en_conformer_ctc_large
            stt_en_conformer_ctc_medium
            stt_en_conformer_ctc_small
            QuartzNet15x5Base-En
        """
        if 'quartznet' in ASR_model_name.lower():
            self.run_ASR = self.run_ASR_QuartzNet_CTC
            self.encdec_class = EncDecCTCModel
            self.params['offset'] = -0.18
            self.model_stride_in_secs = 0.02
            self.asr_delay_sec = -1 * self.params['offset']

        elif 'conformer' in ASR_model_name.lower():
            self.run_ASR = self.run_ASR_BPE_CTC
            self.encdec_class = EncDecCTCModelBPE
            self.model_stride_in_secs = 0.04
            self.asr_delay_sec = 0.0
            self.params['offset'] = -0.072
            self.chunk_len_in_sec = 5
            self.total_buffer_in_secs = 25

        elif 'citrinet' in ASR_model_name.lower():
            self.run_ASR = self.run_ASR_CitriNet_CTC
            self.encdec_class = EncDecCTCModelBPE
            self.model_stride_in_secs = 0.08
            self.asr_delay_sec = 0.0
            self.params['offset'] = -0.18

        elif 'conformer_transducer' in ASR_model_name.lower() or 'contextnet' in ASR_model_name.lower():
            self.run_ASR = self.run_ASR_BPE_RNNT
            self.get_speech_labels_list = self.save_VAD_labels_list
            self.encdec_class = EncDecRNNTBPEModel
            self.model_stride_in_secs = 0.08
            self.asr_delay_sec = 0.0
            self.params['offset'] = 0.065
        else:
            raise ValueError(f"Cannot find the ASR model class for: {self.params['ASR_model_name']}")
        
        if ASR_model_name.endswith('.nemo'):
            asr_model = self.encdec_class.restore_from(restore_path=ASR_model_name)
        else:
            asr_model = self.encdec_class.from_pretrained(model_name=ASR_model_name, strict=False)

        
        if self.ctc_decoder_params['pretrained_language_model'] and 'pyctcdecode' in sys.modules:
            self.decoder = self._load_ASR_LM_model(asr_model)
        else:
            self.decoder = None
        
        self.params['time_stride'] = self.model_stride_in_secs
        self.asr_batch_size = 16
        return asr_model

    def _load_ASR_LM_model(self, asr_model: Type[Union[EncDecCTCModel, EncDecCTCModelBPE]]) -> Type[build_ctcdecoder]:
        self.hot_words = {}
        kenlm_model = self.ctc_decoder_params['pretrained_language_model']
        logging.info(f"Loading language model : {self.ctc_decoder_params['pretrained_language_model']}")
        vocab = asr_model.tokenizer.tokenizer.get_vocab()
        labels = list(vocab.keys())
        labels[0] = "<unk>"
        decoder = build_ctcdecoder(
          labels,
          kenlm_model, 
          alpha=self.ctc_decoder_params['alpha'], 
          beta=self.ctc_decoder_params['beta'])
        return decoder

    def run_ASR_QuartzNet_CTC(self, _asr_model: Type[EncDecCTCModel]):
        """
        Run an QuartzNet ASR model and collect logit, timestamps and text output

        Args:
            _asr_model (class):
                The loaded NeMo ASR model.

        Returns:
            words_list (list):
                Dictionary of the sequence of words from hypothesis.
            words_ts_list (list):
                Dictionary of the time-stamps of words.
        """
        words_dict, word_ts_dict= {}, {}

        wer_ts = WER_TS(
            vocabulary=_asr_model.decoder.vocabulary,
            batch_dim_index=0,
            use_cer=_asr_model._cfg.get('use_cer', False),
            ctc_decode=True,
            dist_sync_on_step=True,
            log_prediction=_asr_model._cfg.get("log_prediction", False),
        )

        with torch.cuda.amp.autocast():
            transcript_logits_list = _asr_model.transcribe(self.audio_file_list, batch_size=1, logprobs=True)
            for idx, logit_np in enumerate(transcript_logits_list):
                uniq_id = get_uniqname_from_filepath(self.audio_file_list[idx])
                log_prob = torch.from_numpy(logit_np)
                logits_len = torch.from_numpy(np.array([log_prob.shape[0]]))
                greedy_predictions = log_prob.argmax(dim=-1, keepdim=False).unsqueeze(0)
                text, char_ts = wer_ts.ctc_decoder_predictions_tensor_with_ts(
                    greedy_predictions, predictions_len=logits_len
                )
                _trans, char_ts_in_feature_frame_idx = self.clean_trans_and_TS(text[0], char_ts[0])
                _spaces_in_sec, hyp_words = self._get_spaces(
                    _trans, char_ts_in_feature_frame_idx, self.params['time_stride']
                )
                word_ts = self.get_word_ts_from_spaces(
                    char_ts_in_feature_frame_idx, _spaces_in_sec, end_stamp=logit_np.shape[0]
                )
                assert len(hyp_words) == len(word_ts), "Words and word-timestamp list length does not match."
                words_dict[uniq_id] = hyp_words
                word_ts_dict[uniq_id] = word_ts
            
        return words_dict, word_ts_dict 

    def run_ASR_CitriNet_CTC(self, _asr_model: Type[EncDecCTCModelBPE]):
        """
        Run the CitriNet ASR model and collect logit, timestamps and text output

        Args:
            audio_file_list (list):
                Dictionary of audio file paths.
            _asr_model (class):
                The loaded NeMo ASR model.

        Returns:
            words_list (list):
                Dictionary of the sequence of words from hypothesis.
            words_ts_list (list):
                Dictionary of the time-stamps of words.
        """
        words_dict, word_ts_dict= {}, {}

        werbpe_ts = WERBPE_TS(
            tokenizer=_asr_model.tokenizer,
            batch_dim_index=0,
            use_cer=_asr_model._cfg.get('use_cer', False),
            ctc_decode=True,
            dist_sync_on_step=True,
            log_prediction=_asr_model._cfg.get("log_prediction", False),
        )

        with torch.cuda.amp.autocast():
            transcript_logits_list = _asr_model.transcribe(self.audio_file_list, batch_size=1, logprobs=True)
            for idx, logit_np in enumerate(transcript_logits_list):
                uniq_id = get_uniqname_from_filepath(self.audio_file_list[idx])
                
                log_prob = torch.from_numpy(logit_np)
                greedy_predictions = log_prob.argmax(dim=-1, keepdim=False).unsqueeze(0)
                logits_len = torch.from_numpy(np.array([log_prob.shape[0]]))
                text, char_ts, word_ts = werbpe_ts.ctc_decoder_predictions_tensor_with_ts(
                    self.model_stride_in_secs, greedy_predictions, predictions_len=logits_len
                )
                hyp_words, word_ts = text[0].split(), word_ts[0]
                word_ts = self.add_offset_to_word_ts(word_ts, self.params['offset'])
                assert len(hyp_words) == len(word_ts), "Words and word-timestamp list length does not match."
                words_dict[uniq_id] = hyp_words
                word_ts_dict[uniq_id] = word_ts
            
        return words_dict, word_ts_dict 

    def set_rnnt_model_parameters(self, _asr_model: Type[EncDecRNNTBPEModel]):
        data_dir = 'configs'
        MODEL_CONFIG_PATH = os.path.join(data_dir,'contextnet_rnnt.yaml')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.exists(MODEL_CONFIG_PATH):
            config_url = self.params['cfg_context_rnnt']
            filename = wget.download(config_url, data_dir)
        contextnet_config = OmegaConf.load(MODEL_CONFIG_PATH)
        
        decoding_config = copy.deepcopy(contextnet_config.model.decoding)
        decoding_config.preserve_alignments = True
        decoding_config.strategy = "greedy_batch"
        with open_dict(decoding_config):
            decoding_config.preserve_alignments = True
        
        _asr_model.change_decoding_strategy(decoding_config)
        _asr_model.preprocessor.featurizer.dither = 0.0
        _asr_model.preprocessor.featurizer.pad_to = 0

        # Freeze the encoder and decoder modules
        _asr_model.encoder.freeze()
        _asr_model.decoder.freeze()
        _asr_model.joint.freeze()

        _asr_model.eval()
    
    def get_rnnt_alignment(self, _asr_model: Type[EncDecRNNTBPEModel], hypothesis):
        decoded_char_list, char_ts = [], []
        space = '▁'
        decoded_text, alignments = hypothesis.text, hypothesis.alignments
        for ti in range(len(alignments)):
            t_u = []
            for uj in range(len(alignments[ti])):
                token = alignments[ti][uj]
                token = token.to('cpu').numpy().tolist()
                decoded_token = _asr_model.decoding.decode_ids_to_tokens([token])[0] if token != _asr_model.decoding.blank_id else ''  # token at index len(vocab) == RNNT blank token
                t_u.append(decoded_token)

            concat_w = ''.join(t_u)
            if len(concat_w) >= 2 and space in concat_w[1:]:
                sep_list = concat_w.split(space)
            else:
                sep_list = [concat_w]
            
            for k, token in enumerate(sep_list):
                char = space + token if k != 0 else token
                decoded_char_list.append(char)
                char_ts.append(round(ti * self.model_stride_in_secs, 2))
        return decoded_char_list, decoded_text, char_ts

    def run_ASR_BPE_RNNT(self, _asr_model: Type[EncDecRNNTBPEModel]):
        """
        Run the CitriNet ASR model and collect logit, timestamps and text output

        Args:
            audio_file_list (list):
                Dictionary of audio file paths.
            _asr_model (class):
                The loaded NeMo ASR model.

        Returns:
            words_list (list):
                Dictionary of the sequence of words from hypothesis.
            words_ts_list (list):
                Dictionary of the time-stamps of words.
        """
        words_dict, word_ts_dict= {}, {}
        
        self.set_rnnt_model_parameters(_asr_model) 
        device = next(_asr_model.parameters()).device
        
        hypotheses_list = []
        batch_size = 1
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, 'manifest.json'), 'w') as fp:
                for audio_file in self.audio_file_list:
                    uniq_id = get_uniqname_from_filepath(audio_file)
                    entry = {'audio_filepath': audio_file, 'duration': self.AUDIO_RTTM_MAP[uniq_id]['duration'], 'text': '-'}
                    fp.write(json.dumps(entry) + '\n')
                config = {'paths2audio_files': self.audio_file_list, 'batch_size': batch_size, 'temp_dir': tmpdir}
            temporary_datalayer = _asr_model._setup_transcribe_dataloader(config)
            for batch in tqdm(temporary_datalayer, desc="Transcribing"):
            
                encoded, encoded_len = _asr_model.forward(input_signal=batch[0].to(device), input_signal_length=batch[1].to(device))
                hypotheses, _ = _asr_model.decoding.rnnt_decoder_predictions_tensor(encoded, encoded_len, return_hypotheses=True)
                hypotheses_list += hypotheses
                del encoded, encoded_len

            for hypothesis in hypotheses_list:
                decoded_hypothesis = _asr_model.decoding.decode_ids_to_tokens(hypothesis.y_sequence.cpu().numpy().tolist())
                decoded_char_list, decoded_text, char_ts = self.get_rnnt_alignment(_asr_model, hypothesis)
                hyp_words, word_ts = get_ts_from_decoded_prediction(self.model_stride_in_secs, decoded_char_list, decoded_text, char_ts)
                word_ts = self.add_offset_to_word_ts(word_ts, self.params['offset'])
                assert len(hyp_words) == len(word_ts), "Words and word-timestamp list length does not match."
                words_dict[uniq_id] = hyp_words
                word_ts_dict[uniq_id] = word_ts
            
            return words_dict, word_ts_dict 
    
    
    def set_buffered_infer_params(self, _asr_model: Type[EncDecCTCModelBPE]):
        cfg = copy.deepcopy(_asr_model._cfg)
        OmegaConf.set_struct(cfg.preprocessor, False)

        # some changes for streaming scenario
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        cfg.preprocessor.normalize = "None"
        preprocessor = nemo_asr.models.EncDecCTCModelBPE.from_config_dict(cfg.preprocessor)
        preprocessor.to(_asr_model.device)

        if cfg.preprocessor.normalize != "per_feature":
            logging.error("Only EncDecCTCModelBPE models trained with per_feature normalization are supported currently")

        # Disable config overwriting
        OmegaConf.set_struct(cfg.preprocessor, True)
        _asr_model.eval()
        
        onset_delay = (
            math.ceil(((self.total_buffer_in_secs - self.chunk_len_in_sec) / 2) / self.model_stride_in_secs) + 1
        )
        mid_delay = math.ceil(
            (self.chunk_len_in_sec + (self.total_buffer_in_secs - self.chunk_len_in_sec) / 2)
            / self.model_stride_in_secs
        )
        tokens_per_chunk = math.ceil(self.chunk_len_in_sec / self.model_stride_in_secs)

        return onset_delay, mid_delay, tokens_per_chunk
        

    def run_ASR_BPE_CTC(self, _asr_model: Type[EncDecCTCModelBPE]) -> Tuple[List[str], Dict[str, float]]:
        """
        Run a CTC-BPE based ASR model and collect logit, timestamps and text output

        Args:
            audio_file_list (list):
                List of audio file paths.
            _asr_model (class):
                The loaded NeMo ASR model.

        Returns:
            words_list (list):
                Dictionary of the sequence of words from hypothesis.
            words_ts_list (list):
                Dictionary of the time-stamps of words.
        """
        torch.manual_seed(0)
        torch.set_grad_enabled(False)
        words_dict, word_ts_dict = {}, {}

        werbpe_ts = WERBPE_TS(
            tokenizer=_asr_model.tokenizer,
            batch_dim_index=0,
            use_cer=_asr_model._cfg.get('use_cer', False),
            ctc_decode=True,
            dist_sync_on_step=True,
            log_prediction=_asr_model._cfg.get("log_prediction", False),
        )

        frame_asr = FrameBatchASR_Logits(
            asr_model=_asr_model,
            frame_len=self.chunk_len_in_sec,
            total_buffer=self.total_buffer_in_secs,
            batch_size=self.asr_batch_size,
        )
        onset_delay, mid_delay, tokens_per_chunk = self.set_buffered_infer_params(_asr_model)
        onset_delay_in_sec = round(onset_delay * self.model_stride_in_secs, 2)
        
        with torch.cuda.amp.autocast():
            logging.info(f"Running ASR model {self.ASR_model_name}")

            for idx, audio_file_path in enumerate(self.audio_file_list):
                uniq_id = get_uniqname_from_filepath(audio_file_path)
                logging.info(f"[{idx+1}/{len(self.audio_file_list)}] FrameBatchASR: {audio_file_path}")
                frame_asr.clear_buffer()

                hyp, greedy_predictions_list, log_prob = get_wer_feat_logit(
                    audio_file_path,
                    frame_asr,
                    self.chunk_len_in_sec,
                    tokens_per_chunk,
                    mid_delay,
                    self.model_stride_in_secs,
                )
                if self.decoder:
                    logging.info(f"Running beam-search decoder with LM {self.ctc_decoder_params['pretrained_language_model']}")
                    log_prob = log_prob.unsqueeze(0).cpu().numpy()[0]
                    hyp_words, word_ts = self.run_pyctcdecode(log_prob, onset_delay_in_sec)
                else:
                    logits_len = torch.from_numpy(np.array([len(greedy_predictions_list)]))
                    greedy_predictions_list = greedy_predictions_list[onset_delay:-mid_delay]
                    greedy_predictions = torch.from_numpy(np.array(greedy_predictions_list)).unsqueeze(0)
                    text, char_ts, word_ts = werbpe_ts.ctc_decoder_predictions_tensor_with_ts(
                        self.model_stride_in_secs, greedy_predictions, predictions_len=logits_len
                    )
                    hyp_words, word_ts = text[0].split(), word_ts[0]
                    word_ts = self.add_offset_to_word_ts(word_ts, self.params['offset'])
                assert len(hyp_words) == len(word_ts), "Words and word-timestamp list length does not match."
                words_dict[uniq_id] = hyp_words
                word_ts_dict[uniq_id] = word_ts

        return words_dict, word_ts_dict
    
    def get_word_ts_from_spaces(self, char_ts: List[float], _spaces_in_sec: List[float], end_stamp: float) -> List[float]:
        """
        Get word-timestamps from the spaces in the decoded prediction.

        Args:
            char_ts (list):
                The time-stamps for each character.
            _spaces_in_sec (list):
                List contains the start and the end time of each space.
            end_stamp (float):
                The end time of the session in sec.

        Return:
            word_timestamps (list):
                List of the timestamps for the resulting words.
        """
        start_stamp_in_sec = round(char_ts[0] * self.params['time_stride'] - self.asr_delay_sec, 2)
        end_stamp_in_sec = round(end_stamp * self.params['time_stride'] - self.asr_delay_sec, 2)
        word_timetamps_middle = [
            [
                round(_spaces_in_sec[k][1] - self.asr_delay_sec, 2),
                round(_spaces_in_sec[k + 1][0] - self.asr_delay_sec, 2),
            ]
            for k in range(len(_spaces_in_sec) - 1)
        ]
        word_timestamps = (
            [[start_stamp_in_sec, round(_spaces_in_sec[0][0] - self.asr_delay_sec, 2)]]
            + word_timetamps_middle
            + [[round(_spaces_in_sec[-1][1] - self.asr_delay_sec, 2), end_stamp_in_sec]]
        )
        return word_timestamps

    def run_pyctcdecode(self, logprob: np.ndarray, onset_delay_in_sec: float, beam_width: int=32) -> Tuple[List[str], List[float]]:
        """
        Run pyctcdecode with the loaded pretrained language model.

        Args:
            logprob (torch.tensor)

        Return:
            hyp_words (list):
                List of words in the hypothesis.
            word_ts (list):
                List of word timestamps from the decoder.
        """
        beams = self.decoder.decode_beams(logprob, beam_width=self.ctc_decoder_params['beam_width'])
        word_ts_beam, words_beam = [], []
        for idx, (word, _) in enumerate(beams[0][2]):
            ts = self.get_word_ts_from_wordframes(idx, beams[0][2], self.model_stride_in_secs, onset_delay_in_sec)
            word_ts_beam.append(ts)
            words_beam.append(word)
        hyp_words, word_ts = words_beam, word_ts_beam
        return hyp_words, word_ts


    @staticmethod
    def get_word_ts_from_wordframes(idx, word_frames, frame_duration, onset_delay):
        offset = -1 * 2.25 * frame_duration - onset_delay
        frame_begin = word_frames[idx][1][0]
        if frame_begin == -1:
            frame_begin = word_frames[idx-1][1][1] if idx != 0 else 0
        frame_end = word_frames[idx][1][1]
        return [round(max(frame_begin*frame_duration + offset, 0), 2),
                round(max(frame_end*frame_duration + offset, 0), 2)]

    @staticmethod 
    def add_offset_to_word_ts(word_ts, offset):
        for k in range(len(word_ts)):
            try:
                word_ts[k] = [round(word_ts[k][0]+offset, 2), round(word_ts[k][1]+offset,2)]
            except:
                ipdb.set_trace()
        return word_ts
