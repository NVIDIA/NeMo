# Copyright 2021 The HuggingFace Team. All rights reserved.
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


import numpy as np
import torch
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR, AudioFeatureIterator
from torch.nn.utils.rnn import pad_sequence

class FrameBatchASRWrapper(FrameBatchASR):
    def read_audio_samples(self, samples, delay, model_stride_in_secs):
        samples = np.pad(samples, (0, int(delay * model_stride_in_secs * self.asr_model._cfg.sample_rate)))
        frame_reader = AudioFeatureIterator(samples, self.frame_len, self.raw_preprocessor, self.asr_model.device)
        self.set_frame_reader(frame_reader)

    def predict(self, samples, delay, tokens_per_chunk, model_stride_in_secs):
        self.reset()
        self.read_audio_samples(samples, delay, model_stride_in_secs)
        hyp = self.transcribe(tokens_per_chunk, delay)
        return hyp

    def get_batch_preds(self, batch_samples, delay, tokens_per_chunk, model_stride_in_secs):
        hyps = []
        for samples in batch_samples:
            hyp = self.predict(samples, delay, tokens_per_chunk, model_stride_in_secs)
            hyps.append(hyp)
        return hyps
            

class HFChunkedASR:
    def __init__(self, asr_model, chunk_len_in_secs=30, overlapping=5):
        self.asr_model = asr_model
        self.chunk_len_in_secs = chunk_len_in_secs
        self.overlapping = overlapping
        self.chunk_len = int(self.asr_model._cfg.sample_rate * self.chunk_len_in_secs)
        self.overlap_len = int(self.asr_model._cfg.sample_rate * self.overlapping)
        self.subsampling_rate = asr_model.encoder.subsampling_rate
        
        self.samples_to_logits_ratio = self.get_sample_logits_ratio() // self.subsampling_rate
    
    def get_sample_logits_ratio(self):
        preprocessor = self.asr_model.preprocessor
        samples = torch.random.rand(1, preprocessor.sample_rate)
        samples_len = torch.tensor([preprocessor.sample_rate], dtype=torch.int32)
        processed, processed_len = preprocessor(input_signal=samples, length=samples_len)
        return int(samples_len[0] / processed_len[0])
        
    def get_batch_preds(self, samples, batch_size):
        hyps = []
        all_item_info = []
        all_log_probs = []
        all_log_probs_len = []
        for batch_wavs, batch_len, batch_idx in self.batch_chunk_iter(samples, batch_size):
            # pad samples
            batch_wavs = torch.nn.utils.rnn.pad_sequence(batch_wavs, batch_first=True).cuda()
            lengths = torch.Tensor(batch_len, dtype=torch.int32).cuda()
            all_item_info.extend(batch_idx) # idx,  (chunk_len, _stride_left, _stride_right), is_last
            
            log_probs, encoded_len, greedy_predictions = self.asr_model.forward(input_signal=batch_wavs,
                                                                              input_signal_length=lengths)
            all_log_probs.extend(list(log_probs))
            all_log_probs_len.extend(list(encoded_len))
        
        all_probs_to_predict = []
        cur_seq_log_probs = []
        all_encoded_len = []
        for item, log_prob, log_prob_len in zip(all_item_info, all_log_probs, all_log_probs_len):
            idx, (chunk_len, _stride_left, _stride_right), is_last = item
            left_start = _stride_left // self.samples_to_logits_ratio
            right_end = _stride_right // self.samples_to_logits_ratio
            
            log_prob = log_prob[left_start: log_prob_len - right_end, :]
            cur_seq_log_probs.append(log_prob)
            if is_last:
                cur_seq_log_probs = torch.concat(cur_seq_log_probs, 0)
                all_probs_to_predict.append(cur_seq_log_probs)
                all_encoded_len.append(cur_seq_log_probs.shape[0])
                cur_seq_log_probs = []
       
        if len(cur_seq_log_probs) > 0:
            cur_seq_log_probs = torch.concat(cur_seq_log_probs, 0)
            all_probs_to_predict.append(cur_seq_log_probs)
            all_encoded_len.append(cur_seq_log_probs.shape[0])
        
        # it's better to have ctc_decoder_predictions tensor accept list of tensors   
        all_log_probs = pad_sequence(all_probs_to_predict, 0)
        all_encoded_len = torch.tensor(all_encoded_len, dtype=torch.int32).cuda()
            
        # ctc_decoder_predictions_tensor could be replaced with riva.asrlib.decoder
        transcribed_texts, _ = self.model.decoding.ctc_decoder_predictions_tensor(
                  decoder_outputs=all_log_probs, decoder_lengths=all_encoded_len, return_hypotheses=False,
              )
            
        return transcribed_texts
    
    def batch_chunk_iter(self, inputs, batch_size):
        cur_batch_samples = []
        cur_batch_samples_len = []
        cur_batch_idx = []
        for idx, samples in enumerate(inputs):
            for item in self.chunk_iter(samples, self.chunk_len, self.overlap_len, self.overlap_len):
                cur_batch_samples.append(item["chunk"])
                cur_batch_samples_len.append(item["chunk_len"])
                cur_batch_idx.append((idx, item["stride"], item["is_last"]))
                if len(cur_batch_samples) == batch_size:
                    yield cur_batch_samples, cur_batch_samples_len, cur_batch_idx
                    cur_batch_samples = []
                    cur_batch_samples_len = []
                    cur_batch_idx = []
        
        if len(cur_batch_samples) > 0:
            yield cur_batch_samples, cur_batch_samples_len, cur_batch_idx
    
    def chunk_iter(self, inputs, chunk_len, stride_left, stride_right):
        # referrence: https://huggingface.co/blog/asr-chunking
        # https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/automatic_speech_recognition.py#L60
        inputs_len = inputs.shape[0]
        step = chunk_len - stride_left - stride_right
        for chunk_start_idx in range(0, inputs_len, step):
            chunk_end_idx = chunk_start_idx + chunk_len
            chunk = inputs[chunk_start_idx:chunk_end_idx]
            #processed = feature_extractor(chunk, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
            _stride_left = 0 if chunk_start_idx == 0 else stride_left
            # all right strides must be full, otherwise it is the last item
            is_last = chunk_end_idx > inputs_len if stride_right > 0 else chunk_end_idx >= inputs_len
            _stride_right = 0 if is_last else stride_right
    
            chunk_len = chunk.shape[0]
            stride = (chunk_len, _stride_left, _stride_right)
            if chunk.shape[0] > _stride_left:
                yield {"is_last": is_last, "stride": stride, "chunk": chunk, "chunk_len": chunk_len}
            if is_last:
                break