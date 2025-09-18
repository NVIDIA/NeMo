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


from typing import Callable, Dict, List, Tuple

import torch

from nemo.collections.asr.inference.stream.decoders.greedy.greedy_decoder import GreedyDecoder
from nemo.collections.asr.inference.utils.normalization import normalize_log_probs


class RNNTGreedyDecoder(GreedyDecoder):

    def __init__(self, vocabulary: List[str], conf_func: Callable = None):
        """
        Initialize the RNNTGreedyDecoder
        Args:
            vocabulary (List[str]): list of vocabulary tokens
            conf_func (Callable): function to compute confidence
        """
        super().__init__(vocabulary, conf_func)

    @staticmethod
    def get_labels(alignment: List[List[Tuple[torch.Tensor, torch.Tensor]]]) -> List[int]:
        """
        Get the first emitted labels from each timestep
        Args:
            alignment (List[List[Tuple[torch.Tensor, torch.Tensor]]]): alignment
        Returns:
            List[int]: list of first emitted labels
        """
        tokens = []
        for t in range(len(alignment)):
            _, token_id = alignment[t][0]
            tokens.append(int(token_id))
        return tokens

    def __call_with_timestamps__(
        self,
        global_timestamps: torch.Tensor | List[int],
        tokens: torch.Tensor | List[int],
        length: int,
        offset: int = 0,
    ):
        """
        Decode the RNNT alignment with timestamps
        """
        if isinstance(global_timestamps, list):
            global_timestamps = torch.tensor(global_timestamps)
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens)

        output = {"tokens": [], "timesteps": [], "confidences": [], "last_token": None, "last_token_idx": None}
        cur_labels = [self.blank_id] * length
        new_offset = len(tokens)
        if offset > 0:
            trimmed_tokens = tokens[offset:].tolist()
            trimmed_timestamps = global_timestamps[offset:].tolist()
        else:
            trimmed_tokens = tokens.tolist()
            trimmed_timestamps = global_timestamps.tolist()

        if len(trimmed_tokens) == 0:
            return output, cur_labels, new_offset

        output["tokens"].extend(trimmed_tokens)
        output["timesteps"].extend(trimmed_timestamps)
        output["confidences"].extend([0.0] * len(trimmed_tokens))
        output["last_token"] = trimmed_tokens[-1]
        output["last_token_idx"] = trimmed_timestamps[-1]

        for t, token in zip(trimmed_timestamps, trimmed_tokens):
            cur_labels[t % length] = token
        return output, cur_labels, new_offset

    def __call__(self, alignment, compute_confidence: bool = True) -> Dict:
        """
        Greedy decode the RNNT alignment
        Args:
            alignment (List[List[Tuple[torch.Tensor, torch.Tensor]]]):
                where alignment[t][u] is a tuple of tensors indicating log_probs and token_id
            compute_confidence (bool): compute confidence or not
        Returns:
            dict: output dictionary containing tokens, timesteps, and confidences
        """

        compute_confidence = compute_confidence and self.conf_func is not None

        output = {"tokens": [], "timesteps": [], "confidences": [], "last_token": None, "last_token_idx": None}

        log_probs = []
        for t in range(len(alignment)):
            for u in range(len(alignment[t])):
                logprob, token_id = alignment[t][u]  # (logprob, token_id)
                token_id = int(token_id)
                if token_id != self.blank_id:
                    output["tokens"].append(token_id)
                    output["timesteps"].append(t)
                    log_probs.append(logprob)

                    output["last_token"] = token_id
                    output["last_token_idx"] = t

        if compute_confidence and len(log_probs) > 0:
            log_probs = [l.unsqueeze(0) for l in log_probs]
            log_probs = torch.cat(log_probs, dim=0)  # (T, V)
            log_probs = log_probs.unsqueeze(0)  # (1, T, V)
            log_probs = normalize_log_probs(log_probs).cpu()  # 1 x T x N
            confidences = self.conf_func(log_probs[0], v=log_probs.shape[2])  # 1 x T
            if confidences.dim() == 2 and confidences.shape[0] == 1:
                confidences = confidences.squeeze(0).tolist()
            output["confidences"] = confidences.tolist()

        return output


class ClippedRNNTGreedyDecoder:

    def __init__(self, vocabulary: List[str], tokens_per_frame: int, conf_func: Callable = None, endpointer=None):
        """
        Initialize the ClippedRNNTGreedyDecoder
        Args:
            vocabulary (List[str]): list of vocabulary tokens
            tokens_per_frame (int): number of tokens per frame
            conf_func (Callable): function to compute confidence
            endpointer (Any): endpointer to detect EOU
        """
        self.greedy_decoder = RNNTGreedyDecoder(vocabulary, conf_func)
        self.endpointer = endpointer
        self.tokens_per_frame = tokens_per_frame

    def extract_tail_tokens(self, alignment: List, start_idx: int, end_idx: int) -> List[int]:
        """
        Extract non-blank tokens from alignment between start_idx and end_idx
        Args:
            alignment (List): alignment
            start_idx (int): start index
            end_idx (int): end index
        Returns:
            List[int]: list of non-blank tokens
        """
        tokens = []
        for t in range(start_idx, end_idx):
            for u in range(len(alignment[t])):
                _, token_id = alignment[t][u]
                token_id = int(token_id)
                if token_id != self.greedy_decoder.blank_id:
                    tokens.append(token_id)
        return tokens

    def __call__(
        self,
        alignment,
        clip_start: int,
        clip_end: int,
        is_last: bool = False,
        is_start: bool = True,
        return_tail_result: bool = False,
        state_start_idx: int = 0,
        state_end_idx: int = 0,
        stop_history_eou: int = None,
    ) -> Tuple[Dict, Dict, bool, int, int]:
        """
        Decode the alignment within the clip range (clip_start, clip_end)
        Args:
            alignment (List[List[Tuple[torch.Tensor, torch.Tensor]]]):
                where alignment[t][u] is a tuple of tensors indicating log_probs and token_id
            clip_start (int): start index of the clip
            clip_end (int): end index of the clip
            is_last (bool): is the last frame or not.
            is_start (bool): is the first frame for this stream or not.
            return_tail_result (bool): return tail result left after clip_end in the buffer
            state_start_idx (int): start index from stream state
            state_end_idx (int): end index from stream state
            stop_history_eou (int): stop history of EOU, if None then use the default stop history
        Returns:
            Tuple[Dict, Dict, bool, int, int]:
                clipped output, tail output, is_eou, updated start_idx, updated end_idx
        """
        # Initialize end-of-utterance state based on input parameters
        is_eou = is_last
        eou_detected_at = len(alignment)

        # Initialize state tracking variables from input parameters
        start_idx, end_idx = state_start_idx, state_end_idx
        # Update indices for next processing step
        if end_idx > clip_start:
            end_idx -= self.tokens_per_frame
            start_idx = end_idx

        if is_start or end_idx <= clip_start:
            start_idx, end_idx = clip_start, clip_end

        # If not already at end of utterance and endpointer exists, try to detect end of utterance
        if not is_eou and self.endpointer is not None:
            is_eou, eou_detected_at = self.endpointer.detect_eou(
                alignment=alignment,
                pivot_point=start_idx,
                search_start_point=clip_start,
                stop_history_eou=stop_history_eou,
            )

        # If end of utterance is detected beyond current end index, extend end index to include it
        if is_eou and eou_detected_at > end_idx:
            end_idx = eou_detected_at

        # If the end index is within the clip range, set the end index to the clip end
        if clip_start <= end_idx < clip_end:
            end_idx = clip_end
            is_eou = False

        # Create a clipped alignment from start to end index and decode it
        clipped_alignment = alignment[start_idx:end_idx]
        clipped_output = self.greedy_decoder(clipped_alignment, compute_confidence=True)

        # If requested, extract tokens after end index for partial results
        tail_output = {"tokens": []}
        if return_tail_result:
            tail_output = {"tokens": self.extract_tail_tokens(alignment, end_idx, len(alignment))}

        return clipped_output, tail_output, is_eou, start_idx, end_idx

    @staticmethod
    def extract_clipped_and_tail_single_pass(
        timesteps: torch.Tensor, tokens: torch.Tensor, start_idx: int, end_idx: int, return_tail_result: bool
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Extract clipped and tail data using tensor operations - no conversion overhead
        """
        if len(timesteps) == 0:
            return [], [], []
        clipped_mask = (timesteps >= start_idx) & (timesteps < end_idx)
        clipped_timesteps = timesteps[clipped_mask].tolist()
        clipped_tokens = tokens[clipped_mask].tolist()
        tail_tokens = []
        if return_tail_result:
            tail_mask = timesteps >= end_idx
            if tail_mask.any():
                tail_tokens = tokens[tail_mask].tolist()

        return clipped_timesteps, clipped_tokens, tail_tokens

    def __call_with_timestamps__(
        self,
        global_timesteps: torch.Tensor,
        tokens: torch.Tensor,
        clip_start: int,
        clip_end: int,
        alignment_length: int,
        is_last: bool = False,
        is_start: bool = True,
        return_tail_result: bool = False,
        state_start_idx: int = 0,
        state_end_idx: int = 0,
        timestamp_offset: int = 0,
        vad_segments: torch.Tensor = None,
        stop_history_eou: int = None,
    ) -> Tuple[Dict, Dict, bool, int, int]:
        """
        Decode using timestamps instead of dense alignment
        Optimized version with vectorized operations and single-pass processing
        Args:
            global_timesteps (torch.Tensor): global timestamps since the start of the stream
            tokens (torch.Tensor): tokens
            clip_start (int): start index of the clip
            clip_end (int): end index of the clip
            alignment_length (int): length of the alignment
            is_last (bool): is the last frame or not.
            is_start (bool): is the first frame for this stream or not.
            return_tail_result (bool): return tail result left after clip_end in the buffer
            state_start_idx (int): start index from stream state
            state_end_idx (int): end index from stream state
            timestamp_offset (int): offset to apply to the timestamps to make them local
            vad_segments (torch.Tensor): Optional VAD segments to use for end-of-utterance detection
            stop_history_eou (int): stop history of EOU, if None then use the default stop history
        Returns:
            Tuple[Dict, Dict, bool, int, int]:
                clipped output, tail output, is_eou, updated start_idx, updated end_idx
        """
        # Initialize end-of-utterance state based on input parameters
        if timestamp_offset:
            timesteps = global_timesteps - timestamp_offset
        else:
            timesteps = global_timesteps
        is_eou = is_last
        eou_detected_at = alignment_length
        start_idx, end_idx = state_start_idx, state_end_idx
        if end_idx > clip_start:
            end_idx -= self.tokens_per_frame
            start_idx = end_idx
        if is_start:
            start_idx, end_idx = clip_start, clip_start
        elif end_idx <= clip_start:
            start_idx, end_idx = clip_start, clip_end

        if len(timesteps) == 0 or len(tokens) == 0:
            return (
                {"tokens": [], "timesteps": [], "confidences": [], "last_token": None, "last_token_idx": None},
                {"tokens": []},
                True,
                start_idx,
                end_idx,
            )

        mask = timesteps >= start_idx
        timesteps_trimmed = timesteps[mask]
        tokens_trimmed = tokens[mask]
        # If not already at end of utterance and endpointer exists, try to detect end of utterance
        if not is_eou and self.endpointer is not None:
            if vad_segments is not None and len(vad_segments) > 0:
                if vad_segments[-1][1] != 0.0:
                    is_eou, eou_detected_at = self.endpointer.detect_eou_vad(
                        vad_segments=vad_segments, search_start_point=start_idx, stop_history_eou=stop_history_eou
                    )
                else:
                    is_eou = True
                    eou_detected_at = -1
            else:
                is_eou, eou_detected_at = self.endpointer.detect_eou_given_timestamps(
                    timesteps=timesteps_trimmed,
                    tokens=tokens_trimmed,
                    alignment_length=alignment_length,
                    stop_history_eou=stop_history_eou,
                )
        # If EOU is detected beyond current end frame, extend end frame to include it
        if is_eou and eou_detected_at > end_idx:
            end_idx = min(eou_detected_at, alignment_length)

        # If the end frame is within the clip range, set the end frame to the clip end
        if clip_start <= end_idx < clip_end:
            end_idx = clip_end
            is_eou = False
        clipped_timesteps, clipped_tokens, tail_tokens = self.extract_clipped_and_tail_single_pass(
            timesteps, tokens, start_idx, end_idx, return_tail_result
        )
        # Make timestamps global again
        if timestamp_offset:
            clipped_timesteps = [t + timestamp_offset for t in clipped_timesteps]
        # Initialize output with last_token tracking like in __call__ method
        clipped_output = {
            "tokens": clipped_tokens,
            "timesteps": clipped_timesteps,
            "confidences": [0.0] * len(clipped_tokens) if len(clipped_tokens) > 0 else [],
            "last_token": None,
            "last_token_idx": None,
        }

        # Set last_token and last_token_idx if there are tokens
        if len(clipped_tokens) > 0:
            clipped_output["last_token"] = clipped_tokens[-1]
            clipped_output["last_token_idx"] = clipped_timesteps[-1] if len(clipped_timesteps) > 0 else None

        # Create tail output
        tail_output = {"tokens": tail_tokens}
        return clipped_output, tail_output, is_eou, start_idx, end_idx
