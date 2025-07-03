import copy
import os
from typing import Optional
import soundfile as sf
import librosa

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from nemo.collections.asr.data.audio_to_text_lhotse_prompted import PromptedAudioToTextMiniBatch
from nemo.collections.asr.parts.utils.streaming_utils import BatchedFeatureFrameBufferer
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.preprocessing.features import normalize_batch
from nemo.collections.asr.parts.preprocessing.segment import get_samples, AudioSegment
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import LengthsType, MelSpectrogramType, NeuralType
from nemo.collections.asr.parts.utils.streaming_utils import *
from nemo.collections.asr.parts.utils.asr_multispeaker_utils import get_hidden_length_from_sample_length
from nemo.collections.asr.parts.utils.streaming_tgt_spk_audio_buffer_ctc_batchview_sample_utils import AudioBuffersDatalayer_tgt_spk, AudioIterator_tgt_spk
from lhotse.dataset.collation import collate_vectors

from nemo.collections.asr.parts.utils.asr_tgtspeaker_utils import (
    get_separator_audio,
)

import torch.nn.functional as F

# class for streaming batched audio-based ASR with ctc

class BatchedFrameASRCTC_tgt_spk():
    def __init__(
            self,
            asr_model,
            frame_len=1.6,
            total_buffer=4.0,
            batch_size=4,
            add_reference_audio = False,
            ref_audio_offset = 0,
            ref_audio_duration = 3,
            ref_separater_duration = 1,
    ):
        self.frame_bufferer = BatchedAudioBufferer_tgt_spk(
            asr_model = asr_model,
            frame_len = frame_len, 
            batch_size = batch_size,
            total_buffer = total_buffer
        )
        self.asr_model = asr_model
        self.frame_len = frame_len
        self.total_buffer = total_buffer
        self.batch_size = batch_size
        self.asr_model = asr_model
        self.decoder = getattr(asr_model, "decoder", None)

        self.batch_size = batch_size
        self.all_logits = []
        if self.decoder is None:
            self.blank_id = len(asr_model.tokenizer.vocabulary)
        elif hasattr(asr_model.decoder, "vocabulary"):
            self.blank_id = len(asr_model.decoder.vocabulary)
        else:
            self.blank_id = len(asr_model.joint.vocabulary)
        self.tokenizer = asr_model.tokenizer
        self.toks_unmerged = []
        self.frame_buffers = []
        cfg = copy.deepcopy(asr_model._cfg)
        self.cfg = cfg
        self.frame_len = frame_len
        OmegaConf.set_struct(cfg.preprocessor, False)

        # some changes for streaming scenario
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        cfg.preprocessor.normalize = "None"
        # import ipdb; ipdb.set_trace()
        self.raw_preprocessor = ASRModel.from_config_dict(cfg.preprocessor)
        self.raw_preprocessor.to(asr_model.device)
        self.preprocessor = self.raw_preprocessor
        self.add_reference_audio = add_reference_audio
        self.ref_audio_offset = ref_audio_offset
        self.ref_audio_duration = ref_audio_duration
        self.ref_separater_duration = ref_separater_duration
        self.reset()

    def reset(self):
        self.batch_index_map = {idx: idx for idx in range(self.batch_size)}
        self.all_preds = [[] for _ in range(self.batch_size)]
        self.data_layer = [AudioBuffersDatalayer_tgt_spk() for _ in range(self.batch_size)]
        self.data_loader = [
            DataLoader(self.data_layer[idx], batch_size=1, shuffle=False) for idx in range(self.batch_size)
        ]
        self.frame_bufferer.reset()
        self.query_pred_len = [0 for _ in range(self.batch_size)]

    def get_partial_samples(self, audio_file: str, offset: float, duration: float, target_sr: int = 16000, dtype: str = 'float32'):
        try:
            with sf.SoundFile(audio_file, 'r') as f:
                start = int(offset * target_sr)
                f.seek(start)
                end = int((offset + duration) * f.samplerate)
                samples = f.read(dtype=dtype, frames = end - start)
                if f.samplerate != target_sr:
                    samples = librosa.core.resample(samples, orig_sr=f.samplerate, target_sr=target_sr)
                samples = samples.transpose()
        except:
            raise ValueError('Frame exceed audio')
        return samples

    def read_audio_file(self, audio_filepaths: list, offsets, durations, query_audio_files, query_offsets, query_durations, separater_freq, separater_duration, separater_unvoice_ratio,delay, model_stride_in_secs):
        max_query_len = -1
        for idx in range(self.batch_size):
            samples = self.get_partial_samples(audio_filepaths[idx], offsets[idx], durations[idx])
            samples = np.pad(samples, (0, int(delay * model_stride_in_secs * self.asr_model._cfg.sample_rate)))
            #query related_variables
            query_samples = self.get_partial_samples(query_audio_files[idx], query_offsets[idx], query_durations[idx])
            separater_audio = get_separator_audio(separater_freq, self.asr_model._cfg.sample_rate, separater_duration, separater_unvoice_ratio)
            separater_audio = separater_audio.astype(np.float32)
            if self.add_reference_audio:
                ref_audio_path = audio_filepaths[idx]
                ref_audio = self.get_partial_samples(ref_audio_path, self.ref_audio_offset, self.ref_audio_duration)
                self.ref_separater_audio = get_separator_audio(separater_freq, self.asr_model._cfg.sample_rate, self.ref_separater_duration, 0.5)
                query_samples = np.concatenate([query_samples, separater_audio, ref_audio, self.ref_separater_audio])
            else:
                query_samples = np.concatenate([query_samples, separater_audio])
            if len(query_samples) > max_query_len:
                max_query_len = len(query_samples)
            frame_reader = AudioIterator_tgt_spk(samples, query_samples, self.
            frame_len, self.asr_model.device)
            self.query_pred_len[idx] = get_hidden_length_from_sample_length(len(query_samples), 160, 8)
            self.set_frame_reader(frame_reader, idx)   
        self.frame_bufferer.query_buffer_placeholder = np.zeros((self.batch_size, max_query_len + self.frame_bufferer.feature_buffer_len))

            

    def set_frame_reader(self, frame_reader, idx):
        self.frame_bufferer.set_frame_reader(frame_reader, idx)

    @torch.no_grad()
    def infer_logits(self):
        frame_buffers = self.frame_bufferer.get_buffers_batch()
        while len(frame_buffers) > 0:
            self.frame_buffers += frame_buffers[:]
            for idx, buffer in enumerate(frame_buffers):
                self.data_layer[idx].set_signal([buffer])
                
            self._get_batch_preds()
            frame_buffers = self.frame_bufferer.get_buffers_batch()

    @torch.no_grad()
    def _get_batch_preds(self):
        device = self.asr_model.device
        data_iters = [iter(data_loader) for data_loader in self.data_loader]
        feat_signals = []
        feat_signal_lens = []
        new_batch_keys = []

        for idx in range(self.batch_size):
            if self.frame_bufferer.signal_end[idx]:
                continue
            batch = next(data_iters[idx])
            feat_signal, feat_signal_len = batch
            feat_signal, feat_signal_len = feat_signal.to(device), feat_signal_len.to(device)
            
            feat_signals.append(feat_signal[0])
            feat_signal_lens.append(feat_signal_len)

            #preserve batch indices
            new_batch_keys.append(idx)
        
        if len(feat_signals) == 0:
            return

        feat_signal = collate_vectors(feat_signals, padding_value = 0)
        feat_signal_len = torch.cat(feat_signal_lens, 0)

        del feat_signals, feat_signal_lens        
        encoded, encoded_len, _, _ = self.asr_model.train_val_forward([feat_signal, feat_signal_len, None, None, None], 0)

        log_probs = self.asr_model.ctc_decoder(encoder_output = encoded)
        predictions = log_probs.argmax(dim = -1, keepdim = False)

        for idx, pred in enumerate(predictions):
            global_index_key = new_batch_keys[idx]  # get index of this sample in the global batch

            has_signal_ended = self.frame_bufferer.signal_end[global_index_key]
            if not has_signal_ended:
                self.all_preds[global_index_key].append(pred[self.query_pred_len[global_index_key]:encoded_len[idx]].cpu().numpy())


        # Position map update
        if len(new_batch_keys) != len(self.batch_index_map):
            for new_batch_idx, global_index_key in enumerate(new_batch_keys):
                self.batch_index_map[global_index_key] = new_batch_idx  # let index point from global pos -> local pos

        del encoded, encoded_len
    def transcribe(
            self, 
            tokens_per_chunk: int, 
            delay: int,
    ):
        """
        Performs "middle token" alignment prediction using the buffered audio chunk.
        """
        self.tokens_per_chunk = tokens_per_chunk
        self.delay = delay
        self.infer_logits()

        self.unmerged = [[] for _ in range(self.batch_size)]
        for idx, preds in enumerate(self.all_preds):
            signal_end_idx = self.frame_bufferer.signal_end_index[idx]
            if signal_end_idx is None:
                raise ValueError("Signal did not end")

            for a_idx, pred in enumerate(preds):
                decoded = pred.tolist()
                self.unmerged[idx] += decoded[max(0,len(decoded) - 1 - delay) : len(decoded) - 1 - delay + tokens_per_chunk]
        output = []
        for idx in range(self.batch_size):
            output.append(self.greedy_merge(self.unmerged[idx]))
        return output 
        
    def greedy_merge(self, preds):
        decoded_prediction = []
        previous = self.blank_id
        for p in preds:
            if (p != previous or previous == self.blank_id) and p != self.blank_id:
                decoded_prediction.append(p)
            previous = p
        hypothesis = self.tokenizer.ids_to_text(decoded_prediction)
        return hypothesis

class BatchedAudioBufferer_tgt_spk():
    def __init__(
            self,
            asr_model,
            frame_len=1.6,
            total_buffer=4.0,
            batch_size=4,
            pad_to_buffer_len=True
    ):
        if hasattr(asr_model.preprocessor, 'log') and asr_model.preprocessor.log:
            self.ZERO_LEVEL_SPEC_DB_VAL = -16.635  # Log-Melspectrogram value for zero signal
        else:
            self.ZERO_LEVEL_SPEC_DB_VAL = 0.0
        self.asr_model = asr_model
        self.sr = asr_model._cfg.sample_rate
        self.frame_len = frame_len
        self.feature_frame_len = int(frame_len * self.sr)
        total_buffer_len = int(total_buffer * self.sr)
        self.n_feat = 1 #asr_model._cfg.preprocessor.features
        self.buffer = (
            np.ones([batch_size, self.n_feat, total_buffer_len], dtype=np.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        )
        self.pad_to_buffer_len = pad_to_buffer_len
        self.batch_size = batch_size

        self.feature_buffer_len = total_buffer_len

        self.feature_buffer = (
            np.ones([self.batch_size, self.n_feat, self.feature_buffer_len], dtype=np.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        )
        self.frame_buffers = []
        self.buffered_features_size = 0
        self.buffered_len = 0
        self.all_frame_reader = [None for _ in range(self.batch_size)]
        self.signal_end = [False for _ in range(self.batch_size)]
        self.signal_end_index = [None for _ in range(self.batch_size)]
        self.reset()
        
    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        self.feature_buffer = (
            np.ones([self.batch_size, self.n_feat, self.feature_buffer_len], dtype=np.float32)
            * self.ZERO_LEVEL_SPEC_DB_VAL
        )
        self.all_frame_reader = [None for _ in range(self.batch_size)]
        self.signal_end = [False for _ in range(self.batch_size)]
        self.signal_end_index = [None for _ in range(self.batch_size)]
        self.buffer_number = 0

    def get_batch_frames(self):
        if all(self.signal_end):
            return []
        
        batch_frames = []
        for idx, frame_reader in enumerate(self.all_frame_reader):
            query_features = frame_reader._query_samples
            try:
                frame = next(frame_reader)
                batch_frames.append([query_features, frame])
            except StopIteration:
                batch_frames.append([query_features, None])
                self.signal_end[idx] = True
                if self.signal_end_index[idx] is None:
                    self.signal_end_index[idx] = self.buffer_number
        self.buffer_number += 1
        return batch_frames
    
    def get_frame_buffers(self, frames):
        self.frame_buffers = []
        for idx in range(self.batch_size):
            frame = frames[idx]
            if frame is not None:
                self.buffer[idx, :, :-self.feature_frame_len] = self.buffer[idx, :, self.feature_frame_len:]
                self.buffer[idx, :, -self.feature_frame_len:] = frame
                self.frame_buffers.append(np.copy(self.buffer[idx]))
            else:
                self.buffer[idx, :, :] *= 0.0
                self.frame_buffers.append(np.copy(self.buffer[idx]))
        return self.frame_buffers
    
    def set_frame_reader(self, frame_reader, idx):
        self.all_frame_reader[idx] = frame_reader
        self.signal_end[idx] = False
        self.signal_end_index[idx] = None

    def get_buffers_batch(self):
        batch_frames = self.get_batch_frames()
        while len(batch_frames) > 0 :
            #batch_frames is a list of [query_features, frame]
            frame_buffers = self.get_frame_buffers([x[1] for x in batch_frames])
            for i, frame_buffer in enumerate(frame_buffers):
                self.query_buffer_placeholder[i, :batch_frames[i][0].shape[0]] = batch_frames[i][0]
                self.query_buffer_placeholder[i, batch_frames[i][0].shape[0]:batch_frames[i][0].shape[0] + frame_buffer.shape[1]] = frame_buffer[0,:]
                query_buffer_len = batch_frames[i][0].shape[0] + frame_buffer.shape[1]
                # frame_buffers[i] = np.concatenate([batch_frames[i][0], frame_buffer[0,:]], axis = 0)
                frame_buffers[i] = self.query_buffer_placeholder[i,:query_buffer_len]
            del batch_frames
            return frame_buffers
        return []
        
    
    #no normalization required, skip feature_buffer update and normalize functionality



    
    
    

        