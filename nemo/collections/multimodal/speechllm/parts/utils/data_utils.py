# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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


def maybe_cast_to_list(x):
    if isinstance(x, np.ndarray):
        return [item.tolist() for item in x]
    return x


def ceil_to_nearest(n, m):
    return (n + m - 1) // m * m


def get_num_samples_from_files(file_list):
    if isinstance(file_list, str):
        file_list = file_list.split(',')
    num_samples = []
    for file in file_list:
        with open(file, 'r') as f:
            lines = list(f.readlines())
            num = len(lines)
            if lines[-1] == '\n':
                num -= 1
            num_samples.append(num)
    return num_samples


def to_cuda(inputs, non_blocking=True):
    """Recursively move inputs to cuda."""
    if isinstance(inputs, torch.Tensor):
        return inputs.cuda(non_blocking=non_blocking)
    elif isinstance(inputs, dict):
        return {k: to_cuda(v, non_blocking) for k, v in inputs.items()}
    elif isinstance(inputs, (list, tuple, set)):
        return inputs.__class__([to_cuda(x, non_blocking) for x in inputs])
    else:
        return inputs


def shift_tokens_by_multi_audios(
    context_tokens, context_lengths, audio_feat_lens, context_start_idx, encoder_max_length
):
    """
        split and shift the context tokens by the audio segments, then concatenate them back. This function assumes that the whole context 
        starts and ends with text tokens, and the audio segments are in between the text tokens. The audio segments are not allowed to be adjacent to each other.
        Args:
            context_tokens: tensor of shape [batch, max_context_len]
            context_lengths: tensor of shape [batch,]
            audio_feat_lens: List[List[int]]
            context_start_idx: List[List[int]]
            encoder_max_length: int
        """
    new_context_tokens = []
    for i in range(context_tokens.shape[0]):
        start_idx_list_i = context_start_idx[i] + [context_lengths[i]]
        input_len_list = [start_idx_list_i[j + 1] - start_idx_list_i[j] for j in range(len(start_idx_list_i) - 1)]
        context_tokens_list = context_tokens[i][: context_lengths[i]].split(input_len_list)
        context_tokens_i = [context_tokens_list[0]]
        for j in range(1, len(context_tokens_list)):
            context_tokens_i.append(
                torch.zeros(audio_feat_lens[i][j - 1], dtype=torch.long, device=context_tokens.device)
            )
            context_tokens_i.append(context_tokens_list[j])
        context_tokens_i = torch.cat(context_tokens_i)
        context_tokens_i = torch.nn.functional.pad(
            context_tokens_i, (0, encoder_max_length - context_tokens_i.shape[0])
        )
        new_context_tokens.append(context_tokens_i)
    new_context_tokens = torch.stack(new_context_tokens)
    return new_context_tokens
