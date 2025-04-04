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

from typing import List, Tuple, Union

import torch
import transformers
from transformers import T5EncoderModel, T5TokenizerFast

from cosmos1.utils import log

transformers.logging.set_verbosity_error()


class CosmosT5TextEncoder(torch.nn.Module):
    """Handles T5 text encoding operations."""

    def __init__(self, model_name: str = "google-t5/t5-11b", device: str = "cuda", cache_dir: str = "~/.cache"):
        """Initializes the T5 tokenizer and encoder.

        Args:
            model_name: The name of the T5 model to use.
            device: The device to use for computations.
        """
        super().__init__()
        try:
            self.tokenizer = T5TokenizerFast.from_pretrained(model_name, cache_dir=cache_dir)
            self.text_encoder = T5EncoderModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)
        except Exception as e:
            log.warning(f"Failed to load T5 model using cache_dir '{cache_dir}', falling back to default location: {e}")
            self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
            self.text_encoder = T5EncoderModel.from_pretrained(model_name).to(device)
        self.text_encoder.eval()
        self.device = device

    @torch.inference_mode()
    def encode_prompts(
        self, prompts: Union[str, List[str]], max_length: int = 512
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes text prompts into hidden state representations using a T5 encoder.

        This function tokenizes the input prompts, processes them through a T5 text encoder,
        and returns the last hidden states. The encoded outputs beyond the actual sequence
        length are zero-padded. All prompts in a batch are padded to max_length.

        Args:
            prompts: Input text to encode. Can be a single string or a list of strings.
            max_length: Maximum sequence length for tokenization and padding. Longer
                sequences will be truncated. Defaults to 512.
            return_mask: If True, returns the attention mask along with encoded text.
                Defaults to False.

        Returns:
            If return_mask is False:
                torch.Tensor: Encoded text embeddings of shape (batch_size, max_length, hidden_size).
            If return_mask is True:
                tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                    - Encoded text embeddings of shape (batch_size, max_length, hidden_size)
                    - Attention mask of shape (batch_size, max_length) as boolean tensor

        Raises:
            ValueError: If the input prompts list is empty.

        Example:
            >>> encoder = CosmosT5TextEncoder()
            >>> prompts = ["Hello world", "Another example"]
            >>> embeddings = encoder.encode_prompts(prompts, max_length=128)
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        if not prompts:
            raise ValueError("The input prompt list is empty.")

        batch_encoding = self.tokenizer.batch_encode_plus(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_length=True,
            return_offsets_mapping=False,
        )

        input_ids = batch_encoding.input_ids.to(self.device)
        attn_mask = batch_encoding.attention_mask.to(self.device)

        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)

        encoded_text = outputs.last_hidden_state
        lengths = attn_mask.sum(dim=1).cpu()

        for batch_id in range(encoded_text.shape[0]):
            encoded_text[batch_id][lengths[batch_id] :] = 0

        return encoded_text, attn_mask
