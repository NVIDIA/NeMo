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
from typing import List, Optional, Tuple

import torch
from omegaconf import DictConfig
from torch import Tensor

from nemo.collections.asr.inference.asr.ctc_inference import CTCInference
from nemo.collections.asr.inference.stream.buffering.audio_bufferer import BatchedAudioBufferer
from nemo.collections.asr.inference.stream.buffering.feature_bufferer import BatchedFeatureBufferer
from nemo.collections.asr.inference.stream.decoders.greedy.greedy_ctc_decoder import ClippedCTCGreedyDecoder
from nemo.collections.asr.inference.stream.endpointing.greedy.greedy_ctc_endpointing import CTCGreedyEndpointing
from nemo.collections.asr.inference.stream.framing.multi_stream import ContinuousBatchedRequestStreamer
from nemo.collections.asr.inference.stream.framing.request import FeatureBuffer, Frame, Request, RequestType
from nemo.collections.asr.inference.stream.framing.request_options import ASRRequestOptions
from nemo.collections.asr.inference.stream.recognizers.base_recognizer import BaseRecognizer
from nemo.collections.asr.inference.stream.state.ctc_state import CTCStreamingState
from nemo.collections.asr.inference.stream.text.text_processing import StreamingTextPostprocessor
from nemo.collections.asr.inference.utils.bpe_decoder import BPEDecoder
from nemo.collections.asr.inference.utils.recognizer_utils import (
    drop_trailing_features,
    get_confidence_utils,
    get_leading_punctuation_regex_pattern,
    make_preprocessor_deterministic,
    normalize_features,
    normalize_log_probs,
    remove_leading_punctuation_spaces,
)
from nemo.collections.asr.models import ASRModel


class CTCBufferedSpeechRecognizer(BaseRecognizer):

    def __init__(
        self,
        cfg: DictConfig,
        asr_model: CTCInference,
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

        self.asr_model_cfg = self.asr_model.copy_asr_config()
        self.asr_model_cfg = make_preprocessor_deterministic(self.asr_model_cfg)
        self.preprocessor = ASRModel.from_config_dict(self.asr_model_cfg.preprocessor)
        self.preprocessor.to(self.device)

        # Streaming related fields
        self.streaming_cfg = cfg.streaming
        self.sample_rate = self.streaming_cfg.sample_rate

        self.subsampling_factor = self.asr_model.get_subsampling_factor()
        self.window_stride = self.asr_model_cfg.preprocessor.window_stride
        self.model_stride_in_secs = self.window_stride * self.subsampling_factor
        self.model_stride_in_milisecs = self.model_stride_in_secs * 1000
        self.chunk_size = self.streaming_cfg.chunk_size
        self.left_padding_size = self.streaming_cfg.left_padding_size
        self.right_padding_size = self.streaming_cfg.right_padding_size
        self.buffer_size_in_secs = self.chunk_size + self.left_padding_size + self.right_padding_size
        self.expected_feature_buffer_len = int(self.buffer_size_in_secs / self.window_stride)

        self.tokens_per_frame = math.ceil(self.chunk_size / self.model_stride_in_secs)
        self.tokens_per_frame_float = self.chunk_size / self.model_stride_in_secs
        self.tokens_per_buffer_float = self.buffer_size_in_secs / self.model_stride_in_secs
        self.tokens_per_right_padding_float = self.right_padding_size / self.model_stride_in_secs
        self.mid_delay = math.ceil((self.chunk_size + self.right_padding_size) / self.model_stride_in_secs)

        # Request type
        self.request_type = RequestType.from_str(self.streaming_cfg.request_type)
        if self.request_type is RequestType.FEATURE_BUFFER:
            # Feature buffering: It will be used when the input is feature buffers
            self.bufferer = BatchedFeatureBufferer(
                sample_rate=self.sample_rate,
                buffer_size_in_secs=self.buffer_size_in_secs,
                preprocessor_cfg=self.asr_model_cfg.preprocessor,
                device=self.device,
            )
        elif self.request_type is RequestType.FRAME:
            # Audio buffering: It will be used when the input is audio frames
            self.bufferer = BatchedAudioBufferer(
                sample_rate=self.sample_rate, buffer_size_in_secs=self.buffer_size_in_secs
            )
        else:
            raise ValueError(f"Unknown request type: {self.request_type}")

        # Confidence related fields
        self.conf_func, self.confidence_aggregator = get_confidence_utils(cfg.confidence)

        # Endpointing
        self.stop_history_eou_in_millisecs = cfg.endpointing.stop_history_eou
        self.endpointer = CTCGreedyEndpointing(
            vocabulary=self.vocabulary,
            ms_per_timestep=self.model_stride_in_milisecs,
            stop_history_eou=self.stop_history_eou_in_millisecs,
            residue_tokens_at_end=cfg.endpointing.residue_tokens_at_end,
        )

        # BPE Decoder
        self.bpe_decoder = BPEDecoder(
            vocabulary=self.vocabulary,
            tokenizer=self.asr_model.tokenizer,
            confidence_aggregator=self.confidence_aggregator,
            asr_supported_puncts=self.asr_supported_puncts,
            word_boundary_tolerance=self.streaming_cfg.word_boundary_tolerance,
            token_duration_in_secs=self.model_stride_in_secs,
        )

        # CTC Decoder
        self.ctc_decoder = ClippedCTCGreedyDecoder(
            vocabulary=self.vocabulary,
            conf_func=self.conf_func,
            endpointer=self.endpointer,
            tokens_per_frame=self.tokens_per_frame,
        )
        self.return_tail_result = cfg.return_tail_result

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

        # Keep small amount of extra padding
        self.padding_mode = self.streaming_cfg.padding_mode
        if self.padding_mode not in ["left", "right"]:
            raise ValueError(f"Unknown padding mode: {self.padding_mode}")
        self.right_padding = self.padding_mode == "right"
        self.tail_padding_in_samples = int(self.chunk_size * self.sample_rate * 0.45)
        self.tail_padding_in_samples = max(self.tail_padding_in_samples, 6400)
        self.zero_log_probs = None
        if self.right_padding:
            self.zero_log_probs = self.init_zero_log_probs()

        super().__init__()

    def init_zero_log_probs(self) -> Tensor:
        """Initialize the log probabilities for the zero buffer."""
        buffer_size_in_samples = int(self.buffer_size_in_secs * self.sample_rate)
        zero_buffer = torch.zeros(1, buffer_size_in_samples, device=self.device)
        zero_features, zero_features_len = self.preprocess(
            buffers=zero_buffer,
            buffer_lens=torch.tensor([zero_buffer.shape[1]], device=self.device),
            expected_feature_buffer_len=self.expected_feature_buffer_len,
        )
        return self.asr_model.get_logprobs(processed_signal=zero_features, processed_signal_length=zero_features_len)[
            0
        ]

    def reset_session(self) -> None:
        """Reset the frame buffer and internal state pool"""
        self.bufferer.reset()
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

    def create_state(self, options: ASRRequestOptions) -> CTCStreamingState:
        """Create new empty state."""
        state = CTCStreamingState()
        state.set_global_offset(-self.tokens_per_right_padding_float)
        new_options = self.augment_options_with_defaults(options)
        state.set_options(new_options)
        return state

    def get_sep(self) -> str:
        """Return the separator for the text postprocessor."""
        return self.sep

    def get_cut_off_range(self, T: int, is_last: bool) -> Tuple[int, int]:
        """Compute the start and end indices to clip the log probs."""
        start = max(T - 1 - self.mid_delay, 0)
        end = T if is_last else min(start + self.tokens_per_frame, T)
        return start, end

    def preprocess(self, buffers: Tensor, buffer_lens: Tensor, expected_feature_buffer_len: int) -> Tuple:
        """Preprocess the buffered frames and extract features."""
        feature_buffers, feature_buffer_lens = self.preprocessor(input_signal=buffers, length=buffer_lens)
        feature_buffers = drop_trailing_features(feature_buffers, expected_feature_buffer_len)
        feature_buffers = normalize_features(feature_buffers, feature_buffer_lens)
        feature_buffer_lens = feature_buffer_lens.clamp(max=feature_buffers.shape[2])
        return feature_buffers, feature_buffer_lens

    def get_logprobs_given_raw_signals(
        self, frames: List[Frame], raw_signals: List[Tensor], left_paddings: List[int]
    ) -> Tensor:
        """Get log probs from the ASR model."""

        if self.right_padding:
            left_paddings = torch.tensor(left_paddings, dtype=torch.int64, device=self.device)

        buffers = []
        for i in range(len(raw_signals)):
            buffer = raw_signals[i]
            # Roll the buffered frames to the left by the left padding
            # This is done to avoid the padding at the beginning of the buffered frames
            # which can cause the performance degradation
            if self.right_padding:
                lpad = left_paddings[i].item()
                if lpad > 0:
                    buffer = buffer.roll(shifts=-lpad)
            buffers.append(buffer.unsqueeze_(0))

        # Only final frames have right padding
        # Keep some amount of extra padding to avoid the performance degradation
        right_paddings = torch.tensor(
            [frame.size - frame.valid_size - self.tail_padding_in_samples for frame in frames], device=self.device
        ).clamp(min=0)

        # Create and adjust the buffer lens
        buffer_lens = torch.tensor([buffers[0].size(1)] * len(buffers), device=self.device)
        buffer_lens = buffer_lens - right_paddings
        if self.right_padding:
            buffer_lens = buffer_lens - left_paddings

        # Preprocess the buffers with corresponding buffer lens
        feature_buffers, feature_buffer_lens = self.preprocess(
            buffers=torch.cat(buffers).to(self.device),
            buffer_lens=buffer_lens,
            expected_feature_buffer_len=self.expected_feature_buffer_len,
        )

        # Get the log probabilities from the ASR model
        log_probs = self.asr_model.get_logprobs(
            processed_signal=feature_buffers, processed_signal_length=feature_buffer_lens
        ).clone()

        # Roll back the log probabilities to the right
        if self.right_padding:
            for i in range(len(log_probs)):
                lpad = left_paddings[i]
                if lpad > 0:
                    lpad = int(lpad / self.sample_rate / self.model_stride_in_secs)
                    log_probs[i] = log_probs[i].roll(lpad, dims=0)
                    log_probs[i][:lpad, :] = self.zero_log_probs[:lpad, :]
        return log_probs

    def get_logprobs_given_processed_signals(
        self, fbuffers: List[FeatureBuffer], processed_signals: List[Tensor]
    ) -> Tensor:
        """Get log probs from the ASR model."""
        processed_signals = torch.cat([sig.unsqueeze_(0) for sig in processed_signals]).to(self.device)
        processed_signals = drop_trailing_features(processed_signals, self.expected_feature_buffer_len)
        processed_signal_lengths = torch.tensor([f.valid_size for f in fbuffers], device=self.device)
        processed_signals = normalize_features(processed_signals, processed_signal_lengths)
        processed_signal_lengths = processed_signal_lengths.clamp(max=processed_signals.shape[2])

        log_probs = self.asr_model.get_logprobs(
            processed_signal=processed_signals, processed_signal_length=processed_signal_lengths
        ).clone()

        if self.right_padding:
            for i in range(len(log_probs)):
                lpad = int(fbuffers[i].roll_size / self.subsampling_factor)
                if lpad > 0:
                    log_probs[i] = log_probs[i].roll(lpad, dims=0)
                    log_probs[i][:lpad, :] = self.zero_log_probs[:lpad, :]
        return log_probs

    def buffer_frames(self, frames: List[Frame]) -> Tensor:
        """Buffer the frames and get the log probabilities."""
        raw_signals, left_paddings = self.bufferer.update(frames)
        log_probs = None
        if len(raw_signals) > 0:
            log_probs = self.get_logprobs_given_raw_signals(frames, raw_signals, left_paddings)
        return log_probs

    def buffer_feature_buffers(self, fbuffers: List[FeatureBuffer]) -> Tensor:
        """Buffer the feature buffers and get the log probabilities."""
        processed_signals = self.bufferer.update(fbuffers)
        log_probs = None
        if len(processed_signals) > 0:
            log_probs = self.get_logprobs_given_processed_signals(fbuffers, processed_signals)
        return log_probs

    def run_greedy_decoder(
        self, state: CTCStreamingState, request: Request, buffer_log_probs: Tensor, start: int, end: int
    ) -> bool:
        """Run Greedy decoder, update state and trigger EOU detection."""
        clipped_output, tail_output, eou_detected, start_idx, end_idx = self.ctc_decoder(
            buffer_log_probs,
            start,
            end,
            request.is_last,
            is_start=request.is_first,
            return_partial_result=self.return_tail_result,
            state_start_idx=state.decoder_start_idx,
            state_end_idx=state.decoder_end_idx,
            stop_history_eou=state.options.stop_history_eou,
        )

        state.update_state(clipped_output, eou_detected)
        state.set_last_token(clipped_output["last_token"], clipped_output["last_token_idx"])
        state.update_from_decoder_results(start_idx, end_idx)
        state.increment_global_offset(self.tokens_per_frame_float)
        state.set_incomplete_segment_tokens(tail_output["tokens"])
        return eou_detected

    def shared_transcribe_step(self, requests: List[Request], log_probs: Tensor) -> None:
        """
        Shared transcribe step for frames and feature buffers.
        Args:
            requests: List of frames or feature buffers to transcribe.
            log_probs: Log probabilities from the ASR model.
        """
        postponed_requests = [(frame_idx, frame.stream_id) for frame_idx, frame in enumerate(requests)]
        next_postponed_requests = []

        while len(postponed_requests) > 0:

            ready_state_ids = set()
            for frame_idx, stream_id in postponed_requests:

                if stream_id in ready_state_ids:
                    # Skip if the state is already ready
                    next_postponed_requests.append((frame_idx, stream_id))
                    continue

                request = requests[frame_idx]
                state = self.get_state(stream_id)
                lp = log_probs[frame_idx].cpu()
                start, end = self.get_cut_off_range(lp.shape[0], request.is_last)
                eou_detected = self.run_greedy_decoder(state, request, lp, start, end)

                if eou_detected:
                    # Form words and push them to the state
                    decoded_words, merge_first_word = self.bpe_decoder.bpe_decode(
                        state.tokens, state.timesteps, state.confidences
                    )
                    state.push_back(decoded_words, merge_first_word, self.confidence_aggregator)
                    state.cleanup_after_eou()

                    # If there are multiple streams, we need to wait until all streams are ready
                    ready_state_ids.add(stream_id)
            states = [self.get_state(request.stream_id) for request in requests]
            self.create_partial_transcript(states)
            if len(ready_state_ids) > 0:
                self.text_postprocessor.process([self.get_state(stream_id) for stream_id in ready_state_ids])
                ready_state_ids.clear()

            postponed_requests = next_postponed_requests.copy()
            next_postponed_requests.clear()

    def transcribe_step_for_feature_buffers(self, fbuffers: List[FeatureBuffer]) -> None:
        """
        Transcribe a step for feature buffers.
        Args:
            fbuffers: List of feature buffers to transcribe.
        """
        log_probs = self.buffer_feature_buffers(fbuffers)
        if log_probs is not None:
            log_probs = normalize_log_probs(log_probs)
            self.shared_transcribe_step(requests=fbuffers, log_probs=log_probs)

    def transcribe_step_for_frames(self, frames: List[Frame]) -> None:
        """
        Transcribe a step for frames.
        Args:
            frames: List of frames to transcribe.
        """
        log_probs = self.buffer_frames(frames)
        if log_probs is not None:
            log_probs = normalize_log_probs(log_probs)
            self.shared_transcribe_step(requests=frames, log_probs=log_probs)

    def get_request_generator(self) -> ContinuousBatchedRequestStreamer:
        """Initialize the request generator."""
        request_generator = ContinuousBatchedRequestStreamer(
            n_frames_per_stream=1,
            frame_size_in_secs=self.chunk_size,
            sample_rate=self.sample_rate,
            batch_size=self.streaming_cfg.batch_size,
            request_type=self.request_type,
            preprocessor=self.preprocessor,
            buffer_size_in_secs=self.buffer_size_in_secs,
            device=self.device,
            pad_last_frame=True,
            right_pad_features=self.right_padding,
            tail_padding_in_samples=self.tail_padding_in_samples,
        )
        return request_generator

    def create_partial_transcript(self, states: List[CTCStreamingState]) -> None:
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
