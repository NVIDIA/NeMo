# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass, field
from typing import Dict, List

import torch
from megatron.energon import (
    VQASample,
    batch_list,
    batch_pad_stack,
)
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.multimodal.data.energon.sample_encoder import SampleEncoder
from nemo.collections.multimodal.data.energon.task_encoder import MultiModalTaskEncoder
from nemo.collections.vlm.llama.data.sample_encoder import Llama3SampleEncoder, LlamaImageTextSample


@dataclass
class LlamaImageTextRawBatch:
    __keys__: List[str] = field(default_factory=list)

    tokens: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    labels: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    loss_mask: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float))

    batch_images: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    batch_masks: torch.Tensor = field(default_factory=lambda: torch.empty(0))

    aspect_ratio_ids: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float))
    aspect_ratio_mask: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float))
    num_tiles: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float))


class LlamaTaskEncoder(MultiModalTaskEncoder):
    def __init__(self, tokenizer, image_processor, multimodal_sample_config):
        super().__init__(tokenizer, image_processor, multimodal_sample_config)
        self.encoders: Dict[str, SampleEncoder] = {
            VQASample.__name__: Llama3SampleEncoder(tokenizer, image_processor, multimodal_sample_config)
        }

    def batch(self, samples: List[LlamaImageTextSample]) -> LlamaImageTextRawBatch:
        keys, images, tokens, labels, loss_mask, vision_mask = [], [], [], [], [], []
        aspect_ratio_ids, aspect_ratio_mask, num_tiles = [], [], []
        for sample in samples:
            keys.append(sample.__key__)
            images.append(sample.images)
            tokens.append(sample.tokens)
            labels.append(sample.labels)
            loss_mask.append(sample.loss_mask)
            vision_mask.append(sample.vision_mask)
            aspect_ratio_ids.append(sample.aspect_ratio_ids)
            aspect_ratio_mask.append(sample.aspect_ratio_mask)
            num_tiles.append(sample.num_tiles)

        batch_keys = batch_list(keys)
        batch_images = batch_pad_stack(images)

        batch_tokens = pad_sequence(tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch_labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # Hard-coded ignore index.

        batch_loss_mask = batch_pad_stack(loss_mask)
        batch_vision_mask = batch_pad_stack(vision_mask)
        batch_aspect_ratio_ids = batch_pad_stack(aspect_ratio_ids)
        batch_aspect_ratio_mask = batch_pad_stack(aspect_ratio_mask)
        batch_num_tiles = torch.tensor(num_tiles)
        return LlamaImageTextRawBatch(
            __keys__=batch_keys,
            batch_images=batch_images,
            batch_masks=batch_vision_mask,
            tokens=batch_tokens,
            labels=batch_labels,
            loss_mask=batch_loss_mask,
            aspect_ratio_ids=batch_aspect_ratio_ids,
            aspect_ratio_mask=batch_aspect_ratio_mask,
            num_tiles=batch_num_tiles,
        )
