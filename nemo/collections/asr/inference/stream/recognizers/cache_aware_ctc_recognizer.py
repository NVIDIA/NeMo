# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


import math
from typing import List, Optional, Set

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

from nemo.collections.asr.inference.asr.cache_aware_ctc_inference import CacheAwareCTCInference
from nemo.collections.asr.inference.stream.buffering.cache_feature_bufferer import BatchedCacheFeatureBufferer
from nemo.collections.asr.inference.stream.decoders.greedy.greedy_ctc_decoder import CTCGreedyDecoder
from nemo.collections.asr.inference.stream.endpointing.greedy.greedy_ctc_endpointing import CTCGreedyEndpointing
from nemo.collections.asr.inference.stream.framing.multi_stream import ContinuousBatchedRequestStreamer
from nemo.collections.asr.inference.stream.framing.request import FeatureBuffer, Frame, RequestType
from nemo.collections.asr.inference.stream.framing.request_options import ASRRequestOptions
from nemo.collections.asr.inference.stream.recognizers.base_recognizer import BaseRecognizer
from nemo.collections.asr.inference.stream.state.cache_aware_ctc_state import CacheAwareCTCStreamingState
from nemo.collections.asr.inference.stream.text.text_processing import StreamingTextPostprocessor
from nemo.collections.asr.inference.utils.bpe_decoder import BPEDecoder
from nemo.collections.asr.inference.utils.context_manager import CacheAwareContextManager
from nemo.collections.asr.inference.utils.endpointing_utils import millisecond_to_frames
from nemo.collections.asr.inference.utils.recognizer_utils import (
    get_confidence_utils,
    get_leading_punctuation_regex_pattern,
    make_preprocessor_deterministic,
    normalize_log_probs,
    remove_leading_punctuation_spaces,
)


class CacheAwareCTCSpeechRecognizer(BaseRecognizer):

    def __init__(
        self,
        cfg: DictConfig,
        asr_model: CacheAwareCTCInference,
        pnc_model: Optional["PunctuationCapitalizer"] = None,
        itn_model: Optional["AlignmentPreservingInverseNormalizer"] = None,
    ):

        # ASR Related fields
        self.asr_model = asr_model
        self.device = self.asr_model.device
        self.supports_punctuation = self.asr_model.supports_punctuation()
        self.asr_supported_puncts = self.asr_model.supported_punctuation()
        self.leading_regex_pattern = get_leading_punctuation_regex_pattern(self.asr_supported_puncts)
        self.blank_id = self.asr_model.get_blank_id()
        self.vocabulary = self.asr_model.get_vocabulary()
        self.sep = self.asr_model.word_separator

        # Set the attention context size if it is provided
        if cfg.streaming.att_context_size is not None:
            self.asr_model.set_default_att_context_size(att_context_size=cfg.streaming.att_context_size)

        # Streaming related fields
        self.streaming_cfg = cfg.streaming
        self.sample_rate = self.streaming_cfg.sample_rate

        self.asr_model_cfg = self.asr_model.copy_asr_config()
        self.model_normalize_type = self.asr_model_cfg.preprocessor.normalize
        self.asr_model_cfg = make_preprocessor_deterministic(self.asr_model_cfg)

        model_stride = self.asr_model.get_subsampling_factor()  # 8, 4, etc
        window_stride_in_secs = self.asr_model_cfg.preprocessor.window_stride  # 0.01
        self.model_stride_in_secs = window_stride_in_secs * model_stride  # 0.08, 0.04, etc
        self.model_stride_in_milisecs = math.ceil(self.model_stride_in_secs * 1000)  # 80, 40, etc

        self.pre_encode_cache_size = self.asr_model.get_pre_encode_cache_size()
        self.model_chunk_size = self.asr_model.get_chunk_size()
        if isinstance(self.model_chunk_size, list):
            self.model_chunk_size = self.model_chunk_size[1]

        self.use_cache = getattr(self.streaming_cfg, "use_cache", True)
        self.use_feat_cache = getattr(self.streaming_cfg, "use_feat_cache", True)

        if self.streaming_cfg.get("chunk_size_in_secs", None) is not None:
            self.chunk_size_in_secs = self.streaming_cfg.chunk_size_in_secs
            self.tokens_per_frame = math.ceil(np.trunc(self.chunk_size_in_secs / window_stride_in_secs) / model_stride)
            # overwrite the encoder streaming params with proper shift size for cache aware streaming
            self.asr_model.setup_streaming_params(
                chunk_size=self.model_chunk_size // model_stride, shift_size=self.tokens_per_frame
            )
        else:
            self.chunk_size_in_secs = self.model_chunk_size * window_stride_in_secs
            self.tokens_per_frame = math.ceil(self.model_chunk_size / model_stride)

        if isinstance(self.pre_encode_cache_size, list):
            self.pre_encode_cache_size = self.pre_encode_cache_size[1]
        self.pre_encode_cache_size_in_secs = self.pre_encode_cache_size * window_stride_in_secs

        # Context Manager
        self.batch_size = self.streaming_cfg.batch_size

        self.context_manager = CacheAwareContextManager(
            cache_aware_model=self.asr_model, num_slots=self.batch_size, use_cache=self.use_cache
        )

        # Feature Bufferer
        model_chunk_size_in_secs = self.model_chunk_size * window_stride_in_secs

        if self.use_cache:
            # if using cache, we need to pad some samples for pre_encode
            self.buffer_size_in_secs = self.pre_encode_cache_size_in_secs + model_chunk_size_in_secs
            self.drop_left_context = None
            self.valid_out_len = None
        else:
            # if not using cache, we need to keep left context in buffer, but no extra padding in pre_encode
            left_context_size = self.asr_model.get_att_context_size()[0]
            if left_context_size < 0:
                raise ValueError(f"Left context size should not be a negative value: {left_context_size}")
            self.buffer_size_in_secs = (
                model_chunk_size_in_secs + left_context_size * model_stride * window_stride_in_secs
            )
            self.drop_left_context = left_context_size
            self.valid_out_len = self.tokens_per_frame

        if self.use_feat_cache:
            # Only calculate mel-spec features for last chunk
            chunk_size_for_feature_buffer = self.chunk_size_in_secs
        else:
            # Calculate mel-spec features for the whole buffer
            chunk_size_for_feature_buffer = self.buffer_size_in_secs

        # Cached Feature Bufferer
        self.request_type = RequestType.from_str(self.streaming_cfg.request_type)
        if self.request_type is not RequestType.FRAME:
            raise ValueError(f"Request type {self.request_type} is not supported for cache-aware streaming.")

        self.bufferer = BatchedCacheFeatureBufferer(
            sample_rate=self.sample_rate,
            buffer_size_in_secs=self.buffer_size_in_secs,
            chunk_size_in_secs=chunk_size_for_feature_buffer,
            preprocessor_cfg=self.asr_model_cfg.preprocessor,
            device=self.device,
        )

        # Confidence related fields
        self.conf_func, self.confidence_aggregator = get_confidence_utils(cfg.confidence)

        # BPE Decoder
        self.bpe_decoder = BPEDecoder(
            vocabulary=self.vocabulary,
            tokenizer=self.asr_model.tokenizer,
            confidence_aggregator=self.confidence_aggregator,
            asr_supported_puncts=self.asr_supported_puncts,
            word_boundary_tolerance=self.streaming_cfg.word_boundary_tolerance,
            token_duration_in_secs=self.model_stride_in_secs,
        )

        # CTC greedy decoder
        self.ctc_decoder = CTCGreedyDecoder(vocabulary=self.vocabulary, conf_func=self.conf_func)
        self.return_tail_result = cfg.return_tail_result

        # Endpointing
        self.stop_history_eou_in_millisecs = cfg.endpointing.stop_history_eou
        self.residue_tokens_at_end = cfg.endpointing.residue_tokens_at_end
        self.endpointer = CTCGreedyEndpointing(
            vocabulary=self.vocabulary,
            ms_per_timestep=self.model_stride_in_milisecs,
            stop_history_eou=self.stop_history_eou_in_millisecs,
            residue_tokens_at_end=self.residue_tokens_at_end,
        )

        # PnC and ITN related fields
        self.text_postprocessor = StreamingTextPostprocessor(
            text_postprocessor_cfg=cfg.text_postprocessor,
            pnc_model=pnc_model,
            itn_model=itn_model,
            asr_supported_puncts=self.asr_supported_puncts,
            asr_supports_punctuation=self.supports_punctuation,
            confidence_aggregator=self.confidence_aggregator,
            sep=self.sep,
            segment_separators=self.asr_model.segment_separators,
            automatic_punctuation=cfg.automatic_punctuation,
            verbatim_transcripts=cfg.verbatim_transcripts,
        )

        super().__init__()

    def reset_session(self) -> None:
        """Reset the frame buffer and internal state pool"""
        self.bufferer.reset()
        self.context_manager.reset()
        super().reset_session()

    def augment_options_with_defaults(self, options: ASRRequestOptions) -> ASRRequestOptions:
        """Augment the options with the default values."""
        enable_itn = self.text_postprocessor.is_itn_enabled() if options.enable_itn is None else options.enable_itn
        enable_pnc = self.text_postprocessor.is_pnc_enabled() if options.enable_pnc is None else options.enable_pnc
        stop_history_eou = (
            self.stop_history_eou_in_millisecs if options.stop_history_eou is None else options.stop_history_eou
        )

        return ASRRequestOptions(
            enable_itn=enable_itn, enable_pnc=enable_pnc, stop_history_eou=stop_history_eou  # In milliseconds
        )

    def create_state(self, options: ASRRequestOptions) -> CacheAwareCTCStreamingState:
        """Create new empty state."""
        state = CacheAwareCTCStreamingState()
        state.set_global_offset(0)
        new_options = self.augment_options_with_defaults(options)

        eou_label_buffer_size = 0
        if new_options.stop_history_eou > 0:
            eou_label_buffer_size = millisecond_to_frames(new_options.stop_history_eou, self.model_stride_in_milisecs)
            eou_label_buffer_size += self.residue_tokens_at_end
        state.setup_label_buffer(eou_label_buffer_size, self.blank_id)
        state.set_options(new_options)
        return state

    def get_sep(self) -> str:
        """Return the separator for the text postprocessor."""
        return self.sep

    def preprocess(self, buffers: List[Tensor], right_paddings: Optional[List[int]] = None):
        """Preprocess the feature buffers by stacking them and computing the lengths"""
        feature_buffers = [f_buffer.unsqueeze_(0) for f_buffer in buffers]
        feature_buffer_lens = torch.tensor([f_buffer.shape[2] for f_buffer in feature_buffers], device=self.device)
        if right_paddings is not None:
            right_paddings = torch.tensor(right_paddings, device=feature_buffer_lens.device)
            feature_buffer_lens = feature_buffer_lens - right_paddings
        feature_buffers = torch.cat(feature_buffers).to(self.device)
        return feature_buffers, feature_buffer_lens

    def run_greedy_decoder(self, state: CacheAwareCTCStreamingState, frame: Frame, log_probs: torch.Tensor):
        """
        Run the greedy CTC decoder on the log_probs and update the state
        Args:
            state: The state of the stream
            frame: The current frame
            log_probs: The log probabilities of the current frame
        Returns:
            updates the state and returns a boolean indicating if EOU is detected
        """
        eou_detected = frame.is_last
        last_token = state.label_buffer[-1] if len(state.label_buffer) > 0 else self.blank_id
        cur_output = self.ctc_decoder(log_probs, compute_confidence=True, previous=last_token)
        cur_labels = self.ctc_decoder.get_labels(log_probs)
        state.update_label_buffer(cur_labels)

        if not eou_detected:
            emissions = state.get_label_buffer()
            pivot_point = len(emissions) - 1
            eou_detected, _ = self.endpointer.detect_eou_near_pivot(
                emissions, pivot_point, stop_history_eou=state.options.stop_history_eou
            )

        state.update_state(cur_output, eou_detected=eou_detected)
        state.increment_global_offset(self.tokens_per_frame)
        return eou_detected

    def decode_log_probs(
        self, frames: List[Frame], log_probs: Tensor, tail_log_probs: Tensor | None, ready_state_ids: Set
    ):

        for idx, frame in enumerate(frames):
            state = self.get_state(frame.stream_id)
            eou_detected = self.run_greedy_decoder(state, frame, log_probs[idx])

            if eou_detected:
                # Form words and push them to the state
                decoded_words, merge_first_word = self.bpe_decoder.bpe_decode(
                    state.tokens, state.timesteps, state.confidences
                )
                state.push_back(decoded_words, merge_first_word, self.confidence_aggregator)
                state.cleanup_after_eou()

                # Add the state to the ready state ids
                ready_state_ids.add(frame.stream_id)

            if tail_log_probs is not None:
                last_token = state.label_buffer[-1] if len(state.label_buffer) > 0 else self.blank_id
                tail_output = self.ctc_decoder(tail_log_probs[idx], compute_confidence=False, previous=last_token)
                state.set_incomplete_segment_tokens(tail_output["tokens"])

    def cache_aware_transcribe_step(
        self,
        frames: List[Frame],
        buffered_features: List[Tensor],
        right_paddings: List[int] | None,
        ready_state_ids: Set,
        keep_all_outputs: bool = False,
    ) -> None:
        """
        Cache Aware Transcribe Step
        It receives a list of frames and features and do the following:

        1. Preprocess the features by stacking them and computing the lengths
        2. Get the context and mapping from the context manager for cache aware streaming
        3. Perform a streaming step with the ASR model
        4. Update the cache and reset the cache slots for the streams that has ended
        5. Decode the log probabilities and update the state
        """
        feature_buffers, feature_buffer_lens = self.preprocess(buffered_features, right_paddings)

        stream_ids = [frame.stream_id for frame in frames]
        eos_flags = [frame.is_last for frame in frames]
        context, mapping = self.context_manager.get_context(stream_ids)

        drop_extra_pre_encoded = 0 if not self.use_cache else self.asr_model.drop_extra_pre_encoded
        log_probs, tail_log_probs, new_context = self.asr_model.stream_step(
            processed_signal=feature_buffers,
            processed_signal_length=feature_buffer_lens,
            context=context,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
            keep_all_outputs=keep_all_outputs,
            drop_left_context=self.drop_left_context,
            valid_out_len=self.valid_out_len,
            return_tail_result=self.return_tail_result,
        )
        if log_probs is not None:
            log_probs = normalize_log_probs(log_probs)
        self.context_manager.update_cache(stream_ids, new_context, mapping)
        self.context_manager.reset_slots(stream_ids, eos_flags)
        self.decode_log_probs(frames, log_probs, tail_log_probs, ready_state_ids)

    def transcribe_step_for_frames(self, frames: List[Frame]) -> None:
        """
        Transcribes the frames in a streaming manner.
        After detecting EOU, it updates the state and run text postprocessor.
        If there are multiple streams, it waits until all states are ready to run text postprocessor.
        """
        all_fbuffers, right_paddings = self.bufferer.update(frames)

        ready_state_ids = set()
        if len(all_fbuffers) > 0:
            # streams that contains multiple frames
            nonfinal_frames, nonfinal_fbuffers = [], []
            final_frames, final_fbuffers = [], []
            final_right_paddings = []
            for jdx, bfeature in enumerate(all_fbuffers):
                frame = frames[jdx]
                if frame.is_last:
                    final_frames.append(frame)
                    final_fbuffers.append(bfeature)
                    final_right_paddings.append(right_paddings[jdx])
                else:
                    nonfinal_frames.append(frame)
                    nonfinal_fbuffers.append(bfeature)

            if len(nonfinal_frames) > 0:
                self.cache_aware_transcribe_step(
                    nonfinal_frames, nonfinal_fbuffers, None, ready_state_ids, keep_all_outputs=False
                )
            if len(final_frames) > 0:
                self.cache_aware_transcribe_step(
                    final_frames, final_fbuffers, final_right_paddings, ready_state_ids, keep_all_outputs=True
                )

        # Postprocess the ready states
        if len(ready_state_ids) > 0:
            self.text_postprocessor.process([self.get_state(stream_id) for stream_id in ready_state_ids])
            ready_state_ids.clear()

        self.create_partial_transcript([self.get_state(frame.stream_id) for frame in frames])

    def transcribe_step_for_feature_buffers(self, fbuffers: List[FeatureBuffer]) -> None:
        """Transcribe a step for feature buffers"""
        raise NotImplementedError("Feature buffer type is not supported for cache aware streaming.")

    def create_partial_transcript(self, states: List[CacheAwareCTCStreamingState]) -> None:
        """Create partial transcript from the state."""
        for state in states:
            # state tokens represent all tokens accumulated since the EOU
            # incomplete segment tokens are the remaining tokens on the right side of the buffer after EOU
            all_tokens = state.tokens + state.incomplete_segment_tokens
            if len(all_tokens) > 0:
                pt_string = self.bpe_decoder.tokenizer.ids_to_text(all_tokens)
                state.partial_transcript = remove_leading_punctuation_spaces(pt_string, self.leading_regex_pattern)
            else:
                state.partial_transcript = ""

    def get_request_generator(self) -> ContinuousBatchedRequestStreamer:
        """Initialize the request generator."""
        # for cache aware streaming we need to process one frame at a time -> n_frames_per_stream=1
        request_generator = ContinuousBatchedRequestStreamer(
            n_frames_per_stream=1,
            frame_size_in_secs=self.chunk_size_in_secs,
            sample_rate=self.sample_rate,
            batch_size=self.batch_size,
            request_type=self.request_type,
            preprocessor=None,
            buffer_size_in_secs=None,
            device=None,
            pad_last_frame=True,
        )
        return request_generator
