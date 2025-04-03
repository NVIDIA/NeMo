# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Optional

import torch
from einops import rearrange

from cosmos1.models.autoregressive.configs.base.tokenizer import TokenizerConfig
from cosmos1.utils.lazy_config import instantiate as lazy_instantiate


def update_vocab_size(
    existing_vocab_size,
    to_be_added_vocab_size,
    training_type,
    add_special_tokens,
    video_special_tokens={},
):
    # New vocab size
    if add_special_tokens:
        existing_vocab_size += to_be_added_vocab_size + len(video_special_tokens)
    # For text_to_video, we add one <bov> special token at the beginning of the video
    elif training_type == "text_to_video":
        existing_vocab_size += to_be_added_vocab_size + 1
    else:
        existing_vocab_size += to_be_added_vocab_size
    return existing_vocab_size


class DiscreteMultimodalTokenizer:
    def __init__(self, tokenizer_config: TokenizerConfig):
        self.tokenizer_config = tokenizer_config
        self.vocab_size = 0
        self.total_seq_len = tokenizer_config.seq_len
        self.pad_to_multiple_of = tokenizer_config.pad_to_multiple_of
        self.training_type = tokenizer_config.training_type
        assert self.training_type in [
            "text_only",
            "text_to_video",
            "video_to_video",
            "image_text_interleaved",
        ], f"{self.training_type} not supported"

        self._build_text_tokenizer()
        self._build_video_tokenizer()

    def _build_text_tokenizer(self):
        r"""Function to initialize the text tokenizer model."""
        if self.tokenizer_config.text_tokenizer is not None:
            self.text_tokenizer = lazy_instantiate(self.tokenizer_config.text_tokenizer.config)
            self.vocab_size += self.tokenizer_config.text_tokenizer.vocab_size
        else:
            self.text_tokenizer = None

    def _build_video_tokenizer(self):
        r"""Function to initialize the video tokenizer model."""
        if self.tokenizer_config.video_tokenizer is not None:
            self.video_tokenizer = lazy_instantiate(self.tokenizer_config.video_tokenizer.config)
            self.video_tokenizer = self.video_tokenizer.to("cuda")
            self.video_vocab_size = self.tokenizer_config.video_tokenizer.vocab_size
            special_token_offset = (
                self.tokenizer_config.video_tokenizer.tokenizer_offset
                + self.tokenizer_config.video_tokenizer.vocab_size
            )
            self.video_special_tokens = {
                "<|begin_of_video|>": special_token_offset,
                "<|end_of_video|>": special_token_offset + 1,
                "<|pad_token_video|>": special_token_offset + 2,
            }

            self.vocab_size = update_vocab_size(
                existing_vocab_size=self.vocab_size,
                to_be_added_vocab_size=self.tokenizer_config.video_tokenizer.vocab_size,
                training_type=self.training_type,
                add_special_tokens=self.tokenizer_config.add_special_tokens,
                video_special_tokens=self.video_special_tokens,
            )
        else:
            self.video_tokenizer = None

    @property
    def pad_id(self):
        r"""Returns the pad_id."""

        if self.training_type == "text_only" or self.training_type == "image_text_interleaved":
            pad_id = self.text_tokenizer.pad_id
        elif self.training_type in ["text_to_video", "video_to_video"]:
            pad_id = self.video_special_tokens["<|pad_token_video|>"]
        else:
            raise ValueError(f"training_type {self.training_type} not defined")
        return pad_id

    @property
    def ignore_index(self):
        r"""Returns which token should be ignored during loss computation."""
        if self.training_type == "text_only" or self.training_type == "image_text_interleaved":
            if self.text_tokenizer.pad_id == self.text_tokenizer.eos_id:
                # If the PAD token is the same as the EOS token, we do not ignore it during loss
                # computation, since we want the model to be able to predict EOS tokens in inference.
                # The PyTorch default ignore_index for the cross-entropy loss is -100.
                ignore_index = -100
            else:
                ignore_index = self.text_tokenizer.pad_id
        elif self.training_type in ["text_to_video", "video_to_video"]:
            ignore_index = self.pad_id
        else:
            raise ValueError(f"training_type {self.training_type} not defined")
        return ignore_index

    @property
    def stop_tokens(self):
        r"""Returns the stop tokens."""
        if self.training_type == "text_only" or self.training_type == "image_text_interleaved":
            stop_tokens = self.text_tokenizer.stop_tokens
        elif self.training_type in ["text_to_video", "video_to_video"]:
            stop_tokens = set([self.video_special_tokens["<|end_of_video|>"]])
        else:
            raise ValueError(f"training_type {self.training_type} not defined")
        return stop_tokens

    def _tokenize_text(self, raw_text: list[str], max_text_seq_len: int = -1):
        r"""Function to tokenize text.
        Args:
            raw_text (list[str]): List of input strings
            max_text_seq_len (int): Maximum sequence length returned by text tokenizer
        Returns:
            text_tokens (list[list[int]]): List of text tokens
        """

        batch_size = len(raw_text)
        text_tokens = [self.text_tokenizer.encode(raw_text[i], bos=True, eos=True) for i in range(batch_size)]

        # Clipping the text tokens so that the sequence length does not exceed max_text_seq_len
        if max_text_seq_len > -1:
            for i in range(len(text_tokens)):
                if len(text_tokens[i]) > max_text_seq_len:
                    # Simply clip and add end of seq token
                    text_tokens[i] = text_tokens[i][0 : max_text_seq_len - 1] + [self.text_tokenizer.eos_id]
        return text_tokens

    def _tokenize_class(self, cls_labels: list[str]):
        r"""Function to tokenize the class label.
        Args:
            cls_labels (list[str]): List of class indices
        Returns:
            class_tokens (list[list[int]]): List of class tokens
        """

        # tokenizer_offset tells what offset should be added to the tokens.
        # This is needed for vocab expansion.
        class_tokens = [[int(x) + self.tokenizer_config.class_tokenizer.tokenizer_offset] for x in cls_labels]

        return class_tokens

    def _tokenize_video(self, videos: torch.Tensor, pixel_chunk_duration: Optional[int] = None):
        r"""Function to tokenize video.
        Args:
            videos (torch.Tensor): Input video data tensor
            pixel_chunk_duration (Optional[float]): Pixel chunk duration. If provided, we pass it to the video tokenizer.
        Returns:
            video_tokens (list[list[int]]): List of video tokens
        """

        video_tokens = []
        batch_size = videos.shape[0]

        quantized_out, _ = self.video_tokenizer.encode(videos, pixel_chunk_duration=pixel_chunk_duration)
        indices = self.video_tokenizer.fsq_quantizer.codes_to_indices(quantized_out.permute(0, 2, 3, 4, 1))

        # Flatten the indices
        indices = rearrange(indices, "B T H W -> B (T H W)")

        # tokenizer_offset tells what offset should be added to the tokens.
        # This is needed for vocab expansion.
        indices += self.tokenizer_config.video_tokenizer.tokenizer_offset

        # Add begin and end of video tokens
        bov_token = self.video_special_tokens["<|begin_of_video|>"]
        eov_token = self.video_special_tokens["<|end_of_video|>"]

        # Append bov and eov tokens
        if self.tokenizer_config.add_special_tokens:
            for i in range(batch_size):
                video_tokens.append([bov_token] + indices[i].tolist() + [eov_token])
        else:
            if self.training_type == "text_to_video":
                for i in range(batch_size):
                    video_tokens.append([bov_token] + indices[i].tolist())
            else:
                for i in range(batch_size):
                    video_tokens.append(indices[i].tolist())
                    assert (
                        len(video_tokens[-1]) == self.tokenizer_config.video_tokenizer.max_seq_len
                    ), f"Expected {self.tokenizer_config.video_tokenizer.max_seq_len} tokens, got {len(video_tokens[-1])}; video shape: {videos.shape}"

        return video_tokens

    def tokenize(self, data_batch: dict):
        r"""Function to tokenize data_dict.
        Args:
            data_batch (dict): Input data dict
        Returns:
            tokens (torch.LongTensor): Token tensor dict
        """

        if (
            self.training_type in ["text_only", "image_text_interleaved"]
            and not self.tokenizer_config.text_tokenizer.tokenize_here
        ):
            # In case of pre-computed tokens, just return the data_batch
            return data_batch["tokens"], None

        # Online tokenization
        tokens = []
        token_boundaries = defaultdict(list)

        # Obtain maximum sequence length
        max_text_seq_len = -1
        max_visual_seq_len = -1

        if self.training_type in ["text_to_video", "video_to_video"]:
            max_visual_seq_len = self.tokenizer_config.video_tokenizer.max_seq_len

        # If max visual sequence length is specified, make sure that text is clipped so that
        # the full video/image is always seen.
        if max_visual_seq_len > -1:
            if self.tokenizer_config.add_special_tokens:
                max_visual_seq_len = max_visual_seq_len + 2  # Two special tokens is for [bov, eov] or [boi, eoi] token
            elif self.training_type == "text_to_video":
                max_visual_seq_len = max_visual_seq_len + 1
            else:
                max_visual_seq_len = max_visual_seq_len
            assert (
                max_visual_seq_len <= self.total_seq_len
            ), f"max_visual_seq_len ({max_visual_seq_len}) is greater that total sequence length ({self.total_seq_len})"
            max_text_seq_len = self.total_seq_len - max_visual_seq_len

        # Tokenize the text
        if (
            "text" in self.training_type
            and self.text_tokenizer is not None
            and self.tokenizer_config.text_tokenizer.tokenize_here
        ):
            key = self.tokenizer_config.text_tokenizer.data_key
            batch_size = len(data_batch[key])
            assert key in data_batch, f"Key {key} should be present in data for text tokenizer"
            tokens = self._tokenize_text(data_batch["caption"], max_text_seq_len)

            for i in range(batch_size):
                token_boundaries["text"].append((0, len(tokens[i])))
        else:
            tokens = []
            batch_size = None

        # Tokenize the class label
        if "class" in self.training_type and self.tokenizer_config.class_tokenizer is not None:
            key = self.tokenizer_config.class_tokenizer.data_key
            assert key in data_batch, f"Key {key} should be present in data for class tokenizer"
            batch_size = len(data_batch[key]) if batch_size is None else batch_size
            tokens_class = self._tokenize_class(data_batch[key])
            if len(tokens) == 0:
                tokens = tokens_class
                for i in range(batch_size):
                    token_boundaries["class"].append((0, len(tokens[i])))
            else:
                for i in range(batch_size):
                    token_boundaries["class"].append((len(tokens[i]), len(tokens[i]) + len(tokens_class[i])))
                    tokens[i] = tokens[i] + tokens_class[i]

        # Tokenize the video
        if self.video_tokenizer is not None and self.tokenizer_config.video_tokenizer.tokenize_here:
            key = self.tokenizer_config.video_tokenizer.data_key
            assert key in data_batch, f"Key {key} should be present in data for video tokenizer"
            batch_size = len(data_batch[key]) if batch_size is None else batch_size

            pixel_chunk_duration = (
                None  # If not specified, we assume it's a video dataset and use the default chunk duration
            )
            dataset_name = data_batch.get("dataset_name", None)
            if dataset_name is not None and dataset_name.startswith("image"):
                # If it's an image dataset, we use a pixel chunk duration of 1
                pixel_chunk_duration = 1
            tokens_video = self._tokenize_video(data_batch[key], pixel_chunk_duration=pixel_chunk_duration)
            if len(tokens) == 0:
                tokens = tokens_video
                for i in range(batch_size):
                    token_boundaries["video"].append((0, len(tokens[i])))
                    # [B,] each entry is ((0, len(tokens[i])))
            else:
                for i in range(batch_size):
                    token_boundaries["video"].append((len(tokens[i]), len(tokens[i]) + len(tokens_video[i])))
                    tokens[i] = tokens[i] + tokens_video[i]

        # Combine the tokens and do padding
        max_seq_len_in_batch = max([len(token) for token in tokens])
        if self.pad_to_multiple_of is not None:
            # Pad the sequence length to the nearest multiple of pad_to_multiple_of
            max_seq_len_in_batch = ((max_seq_len_in_batch - 1) // self.pad_to_multiple_of + 1) * self.pad_to_multiple_of
        pad_to_len = min(max_seq_len_in_batch, self.total_seq_len)
        for i in range(len(tokens)):
            if len(tokens[i]) < pad_to_len:
                tokens[i] = tokens[i] + [self.pad_id] * (pad_to_len - len(tokens[i]))
            else:
                tokens[i] = tokens[i][0:pad_to_len]

        # Convert it to long tensor
        tokens = torch.LongTensor(tokens)
        return tokens, token_boundaries
