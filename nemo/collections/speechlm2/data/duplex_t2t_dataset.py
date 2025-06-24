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
import re
import random

import torch
import torch.utils.data
from lhotse import CutSet, Seconds, compute_num_frames
from lhotse.cut import Cut
from lhotse.dataset.collation import collate_audio, collate_vectors
from lhotse.utils import ifnone

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.utils import logging


class DuplexT2TDataset(torch.utils.data.Dataset):
    """
    A dataset for duplex speech-to-speech models that handles bidirectional conversations.

    This dataset processes Lhotse CutSet objects containing recordings with supervision segments
    from different speakers (roles). It creates aligned representations of audio and text for
    both source (input) and target (output) channels, preserving temporal alignment between
    audio frames and text tokens.

    Args:
        tokenizer (TokenizerSpec):
            Tokenizer for converting text to token IDs and vice versa. Must support BOS and EOS tokens.
            It's expected to support PAD token as well, otherwise we will use 0 as the pad token
            and emit a warning.

        frame_length (Seconds):
            Duration of a single frame in seconds. Used to calculate frame positions for token alignment.

        source_sample_rate (int):
            Sample rate for source audio (e.g., 16000 Hz).

        target_sample_rate (int):
            Sample rate for target audio (e.g., 22050 Hz).

        input_roles (list[str], optional):
            List of speaker roles (cut.supervisions[:].speaker) to consider as inputs. Defaults to ["user"].

        output_roles (list[str], optional):
            List of speaker roles (cut.supervisions[:].speaker) to consider as outputs. Defaults to ["agent"].

    Returns:
        A dictionary with the following keys:
            - source_audio: Tensor of source waveform samples [B, T]
            - source_audio_lens: Tensor of source audio lengths [B]
            - target_audio: Tensor of target waveform samples [B, T]
            - target_audio_lens: Tensor of target audio lengths [B]
            - target_tokens: Tensor of target text tokens [B, T], with special tokens (BOS/EOS/PAD)
                at positions aligned with audio frames
            - target_token_lens: Tensor of target token sequence lengths [B]
            - source_tokens: Tensor of source text tokens [B, T], with special tokens (BOS/EOS/PAD)
                at positions aligned with audio frames
            - source_token_lens: Tensor of source token sequence lengths [B]
            - target_texts: List of full target texts joined from output_roles supervisions [B]

    Notes:
        - The dataset ensures frame-level alignment between audio and text by inserting tokens at
          specific frame positions based on the timing of supervision segments.
        - PAD tokens (typically 0) are used to fill gaps where there's no text.
        - BOS tokens mark the beginning of each speech segment.
        - EOS tokens mark the end of each speech segment.
        - Text tokens from each speaker are placed at frame positions corresponding to their
          timestamp in the original recording, preserving the temporal relationship.
          This is a segment-level alignment only, not word-level alignment.
        - If collate_source_interleaved is True, the source tokens are interleaved with pad
          tokens as per word level timestamps.
    
    Duplex text/token to text/speech model with frozen ASR.
    With cfg.generate_speech=True and cfg.audio_loss_weight > 0, this model can be trained to generate speech.
    
    Text to text model:
      speech → [ASR] → decoded text → [deterministic retokenization] → tokens from LLM’s vocabulary → 
      [LLM’s embed and combine with agent channel] → continuous representation → [LLM]

      CASE 1 input (oracle-EoU):
        <BOS><turn 1 tokens><EOS><PAD tokens to fill user turn 1 duration><PAD tokens to fill agent turn 1>
        <BOS><turn 2 tokens><EOS><PAD tokens to fill user turn 2 duration><PAD tokens to fill agent turn 2>
        ...
      CASE 2 input (oracle-aligned):
        <turn 1 tokens word-aligned><PAD tokens to fill agent turn 1>
        <turn 2 tokens word-aligned><PAD tokens to fill agent turn 2>
        ...

    Token to text model:
      speech → [ASR] → frame-level output tokens → [ASR’s embed and combine with agent channel embed^] → 
      continuous representation → [shallow transformer module*] → continuous representation → [LLM]
    
      Input:
        <turn 1 tokens frame-aligned><PAD tokens to fill agent turn 1>
        <turn 2 tokens frame-aligned><PAD tokens to fill agent turn 2>
        ...

    ^Agent channel can be embedded either via LLM’s tokenize+embedding or ASR’s tokenization+embedding. 

    *transformer so that the self-attention can learn which tokens to combine/split etc to match LLM’s vocabulary space. 
    """

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        frame_length: Seconds,
        source_sample_rate: int,
        # target_sample_rate: int,
        input_roles: list[str] = None,
        output_roles: list[str] = None,
        collate_source_interleaved: bool = False,
        train_retokenizer: bool = False,
    ):
        self.tokenizer = tokenizer
        self.frame_length = frame_length
        self.source_sample_rate = source_sample_rate
        # self.target_sample_rate = target_sample_rate
        self.input_roles = set(ifnone(input_roles, ["user"]))
        self.output_roles = set(ifnone(output_roles, ["agent"]))
        
        self.collate_source_interleaved = collate_source_interleaved
        self.train_retokenizer = train_retokenizer
        
        assert tokenizer.bos is not None, "BOS support in the tokenizer is required for S2S models."
        assert tokenizer.eos is not None, "EOS support in the tokenizer is required for S2S models."

    def __getitem__(self, cuts: CutSet) -> dict:
        stripped_cuts = cuts.transform_text(_strip_timestamps)
        
        source_audio, source_audio_lens = collate_audio(cuts.resample(self.source_sample_rate))
        # target_audio, target_audio_lens = collate_audio(
        #     cuts.resample(self.target_sample_rate), recording_field="target_audio"
        # )

        if self.collate_source_interleaved:
            source_tokens, source_token_lens = collate_token_channel_interleaved(
                cuts, self.tokenizer, self.frame_length, roles=self.input_roles
            )
        else:
            source_tokens, source_token_lens = collate_token_channel(
                stripped_cuts, self.tokenizer, self.frame_length, roles=self.input_roles
            )
        target_tokens, target_token_lens = collate_token_channel(
            stripped_cuts, self.tokenizer, self.frame_length, roles=self.output_roles
        )
        return {
            "source_audio": source_audio,
            "source_audio_lens": source_audio_lens,
            # "target_audio": target_audio,
            # "target_audio_lens": target_audio_lens,
            "target_tokens": target_tokens,
            "target_token_lens": target_token_lens,
            "source_tokens": source_tokens,
            "source_token_lens": source_token_lens,
            "target_texts": [
                " ".join(s.text for s in cut.supervisions if s.speaker in self.output_roles) for cut in stripped_cuts
            ],
            "source_texts": [
                " ".join(s.text for s in cut.supervisions if s.speaker in self.input_roles) for cut in stripped_cuts
            ],
            "data_id": [cut.id for cut in cuts],
        }


def collate_token_channel(
    cuts: CutSet,
    tokenizer: TokenizerSpec,
    frame_length: Seconds,
    roles: set[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id = get_pad_id(tokenizer)
    tokens = [
        build_token_channel(c, tokenizer=tokenizer, frame_length=frame_length, roles=roles, pad_id=pad_id)
        for c in cuts
    ]
    token_lens = torch.tensor([len(tt) for tt in tokens])
    tokens = collate_vectors(tokens, padding_value=pad_id)

    return tokens, token_lens

def collate_token_channel_interleaved(
    cuts: CutSet,
    tokenizer: TokenizerSpec,
    frame_length: Seconds,
    roles: set[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id = get_pad_id(tokenizer)
    tokens = [
        build_token_channel_interleaved(c, tokenizer=tokenizer, frame_length=frame_length, roles=roles, pad_id=pad_id)
        for c in cuts
    ]
    token_lens = torch.tensor([len(tt) for tt in tokens])
    tokens = collate_vectors(tokens, padding_value=pad_id)

    return tokens, token_lens

def build_token_channel(
    cut: Cut,
    tokenizer: TokenizerSpec,
    frame_length: Seconds,
    roles: set[str],
    pad_id: int = -1,
) -> torch.Tensor:
    diagnostic = f"Extra info: {cut.id=}"
    if getattr(cut, "shard_origin", None) is not None:
        diagnostic = f"{diagnostic} {cut.shard_origin=}"

    total = compute_num_frames(cut.duration, frame_length, cut.sampling_rate)
    tokens = torch.ones(total, dtype=torch.long) * pad_id
    for supervision in cut.supervisions:
        if supervision.speaker in roles:
            text_ids = torch.as_tensor([tokenizer.bos] + tokenizer.text_to_ids(supervision.text))

            # Determine the frame offset for the start of the supervision to insert the text tokens.
            pos = compute_num_frames(supervision.start, frame_length, cut.sampling_rate)
            if pos > len(tokens):
                logging.warning(
                    f"Ill-constructed example: the beginning offset of a supervision {pos} is larger than the example's length {len(tokens)}. {diagnostic}"
                )
                continue

            # Determine the frame offset for the last non-EOS text token to form a valid range for insertion;
            # Note that EOS will be placed possibly much later, at the frame that coincides with end of speech,
            # rather than end of text. The gap between last non-EOS token and EOS token will be filled with `pad_id`.
            endpos = pos + len(text_ids)
            if endpos > len(tokens):
                trunc_len = len(tokens) - pos
                logging.warning(
                    f"Truncating training example's text_ids of length {len(text_ids)} by {trunc_len} because {endpos=} > {len(tokens)=}. {diagnostic}"
                )
                text_ids = text_ids[:trunc_len]
            try:
                tokens[pos:endpos] = text_ids
            except Exception as e:
                raise RuntimeError(f"{tokens.shape=} {pos=} {endpos=} {text_ids.shape=} {diagnostic}") from e

            # Insert EOS at the end of the supervision segment.
            eospos = compute_num_frames(supervision.end, frame_length, cut.sampling_rate)
            if eospos < len(tokens):  # skip otherwise - unfinished turn
                tokens[eospos] = tokenizer.eos

    return tokens


def _strip_timestamps(
    text: str, _TIMESTAMP_PATTERN=re.compile(r"<\|\d+\|>"), _SPACE_PATTERN=re.compile(r"\s+")
) -> str:
    """
    Strips timestamp tokens from text, e.g. turns:
      '<|0|> Hey <|3|> <|3|> how <|5|> <|7|> are <|8|> <|8|> <|10|> you? <|12|>'
      into:
      'Hey how are you?'
    """
    # Regexp pattern args are cached compiled patterns (micro-optimization).
    text = _TIMESTAMP_PATTERN.sub("", text)  # strip timestamp tokens if present
    return _SPACE_PATTERN.sub(" ", text).strip()  # strip multi-whitespaces


def parse_timestamped_text(text: str) -> tuple[bool, list[tuple[str, int, int]]]:
    """
    Parses text with timestamp tags and returns a list of (word, start_frame, end_frame) tuples.
    Returns (is_valid, words_with_spans) where is_valid indicates if the text
    contains valid timestamp tags.
    
    Example:
        Input: "<|0|> when <|1|> <|5|> was <|6|> <|7|> it <|8|>"
        Output: (True, [("when", 0, 1), ("was", 5, 6), ("it", 7, 8)])
    """
    # Check if text contains timestamp tags
    if "<|" not in text or "|>" not in text:
        return False, []
        
    # Split text into words and timestamps
    parts = re.split(r'(<\|\d+\|>)', text)
    words_with_spans = []
    timestamps = []
    words = []
    
    for part in parts:
        if not part.strip():
            continue
            
        # Check if this part is a timestamp tag
        timestamp_match = re.match(r'<\|\d+\|>', part)
        if timestamp_match:
            try:
                timestamp = int(part[2:-2])  # Extract number between <| and |>
                timestamps.append(timestamp)
            except ValueError:
                return False, []
        else:
            # This is a word
            words.append(part.strip())
    
    # Assert that we have an even number of timestamps (start and end for each word)
    assert len(timestamps) % 2 == 0, f"Expected even number of timestamps, got {len(timestamps)}"
    
    # Now pair words with their timestamp spans
    # Each word should be paired with consecutive timestamps
    word_idx = 0
    for i in range(0, len(timestamps), 2):  # Step by 2 to get start/end pairs
        if word_idx < len(words):
            start_frame = timestamps[i]
            end_frame = timestamps[i + 1]
            words_with_spans.append((words[word_idx], start_frame, end_frame))
            word_idx += 1
                    
    return True, words_with_spans

def build_token_channel_interleaved(
    cut: Cut,
    tokenizer: TokenizerSpec,
    frame_length: Seconds,
    roles: set[str],
    pad_id: int = -1,
) -> torch.Tensor:
    """
    Similar to build_token_channel but handles timestamped text, placing tokens at specific
    frame positions based on timestamps and filling gaps with pad tokens.
    
    Args:
        cut: Lhotse Cut object containing the audio and supervision segments
        tokenizer: Tokenizer for converting text to tokens
        frame_length: Duration of a single frame in seconds
        roles: Set of speaker roles to process
        pad_id: Token ID to use for padding
        
    Returns:
        torch.Tensor of shape [total_frames] containing token IDs with pad tokens
        in between words based on timestamps
    """
    diagnostic = f"Extra info: {cut.id=}"
    if getattr(cut, "shard_origin", None) is not None:
        diagnostic = f"{diagnostic} {cut.shard_origin=}"

    total = compute_num_frames(cut.duration, frame_length, cut.sampling_rate)
    tokens = torch.ones(total, dtype=torch.long) * pad_id
    
    for supervision in cut.supervisions:
        if supervision.speaker in roles:
            # Validate and parse timestamped text
            is_valid, words_with_spans = parse_timestamped_text(supervision.text)
            if not is_valid:
                logging.warning(
                    f"Invalid timestamp format in supervision text: {supervision.text}. {diagnostic}"
                )
                continue
                
            if not words_with_spans:
                logging.warning(
                    f"No valid words with timestamps found in supervision text: {supervision.text}. {diagnostic}"
                )
                continue
                
            # Process each word with its time span
            for word, start_frame, end_frame in words_with_spans:
                # Convert frame IDs to time (seconds), add supervision.start offset, then convert back to frame IDs
                start_time = start_frame * frame_length
                end_time = end_frame * frame_length
                
                # Add supervision.start offset to get absolute time
                absolute_start_time = start_time + supervision.start
                absolute_end_time = end_time + supervision.start
                
                # Convert back to frame IDs relative to the entire cut
                start_pos = compute_num_frames(absolute_start_time, frame_length, cut.sampling_rate)
                end_pos = compute_num_frames(absolute_end_time, frame_length, cut.sampling_rate)
                
                if start_pos >= len(tokens):
                    logging.warning(
                        f"Start frame {start_frame} (absolute time {absolute_start_time:.3f}s) maps to frame {start_pos} which is beyond total frames {len(tokens)}. {diagnostic}"
                    )
                    continue
                    
                # Tokenize the word and add BOS token
                word_ids = torch.as_tensor(tokenizer.text_to_ids(word))
                
                # Handle truncation if word would exceed total frames
                if end_pos > len(tokens):
                    end_pos = len(tokens)
                    
                # Place tokens within the span, padding if necessary
                span_length = end_pos - start_pos
                if span_length >= len(word_ids):
                    # We have enough space, place tokens and pad the rest
                    tokens[start_pos:start_pos + len(word_ids)] = word_ids
                else:
                    # Span is too short, truncate tokens
                    truncated_ids = word_ids[:span_length]
                    tokens[start_pos:end_pos] = truncated_ids
                    logging.warning(
                        f"Truncating word tokens of length {len(word_ids)} to {span_length} for word '{word}'. {diagnostic}"
                    )

    return tokens

