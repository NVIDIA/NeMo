# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from x_transformers import ContinuousTransformerWrapper
from x_transformers.x_transformers import exists


# had to override the implementation to fix the memory implementation.
class TransformerXL(ContinuousTransformerWrapper):
    def __init__(self, *args, max_mem_len, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_mem_len = max_mem_len

    def forward(
        self, x, return_embeddings=False, mask=None, return_attn=False, mems=None, pos=None, **kwargs,
    ):
        b, n, _, device = *x.shape, x.device

        x = self.project_in(x)
        x = x + self.pos_emb(x, pos=pos)

        x = self.post_emb_norm(x)
        x = self.emb_dropout(x)

        x, intermediates = self.attn_layers(x, mask=mask, mems=mems, return_hiddens=True, **kwargs)
        x = self.norm(x)

        out = self.project_out(x) if not return_embeddings else x

        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        if self.max_mem_len > 0:
            hiddens = intermediates.hiddens
            new_mems = list(map(lambda pair: torch.cat(pair, dim=-2), zip(mems, hiddens))) if exists(mems) else hiddens
            new_mems = list(map(lambda t: t[..., -self.max_mem_len :, :].detach(), new_mems))
            return out, new_mems

        return out
