# adapted from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/FastPitch/fastpitch/transformer.py
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from nemo.collections.tts.helpers.common import get_mask_from_lengths, PositionalEmbedding


class PositionwiseConvFF(nn.Module):
    def __init__(self, n_text_dim, d_inner, kernel_size):
        super(PositionwiseConvFF, self).__init__()
        self.CoreNet = nn.Sequential(
            nn.Conv1d(n_text_dim, d_inner, kernel_size, 1, (kernel_size // 2)),
            nn.ReLU(),
            nn.Conv1d(d_inner, n_text_dim, kernel_size, 1, (kernel_size // 2)),
        )
        self.layer_norm = nn.LayerNorm(n_text_dim)

    def forward(self, inp):
        # positionwise feed-forward
        core_out = inp.transpose(1, 2)
        core_out = self.CoreNet(core_out)
        core_out = core_out.transpose(1, 2)

        # residual connection + layer normalization
        output = self.layer_norm(inp + core_out)

        return output


class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, n_text_dim, d_head):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.n_text_dim = n_text_dim
        self.d_head = d_head
        self.scale = 1 / (d_head ** 0.5)

        self.qkv_net = nn.Linear(n_text_dim, 3 * n_head * d_head)
        self.o_net = nn.Linear(n_head * d_head, n_text_dim, bias=False)
        self.layer_norm = nn.LayerNorm(n_text_dim)

    def forward(self, inp, attn_mask=None):
        residual = inp

        n_head, d_head = self.n_head, self.d_head

        head_q, head_k, head_v = torch.chunk(self.qkv_net(inp), 3, dim=-1)
        head_q = head_q.view(inp.size(0), inp.size(1), n_head, d_head)
        head_k = head_k.view(inp.size(0), inp.size(1), n_head, d_head)
        head_v = head_v.view(inp.size(0), inp.size(1), n_head, d_head)

        q = head_q.permute(0, 2, 1, 3).reshape(-1, inp.size(1), d_head)
        k = head_k.permute(0, 2, 1, 3).reshape(-1, inp.size(1), d_head)
        v = head_v.permute(0, 2, 1, 3).reshape(-1, inp.size(1), d_head)

        attn_score = torch.bmm(q, k.transpose(1, 2))
        attn_score.mul_(self.scale)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
            attn_mask = attn_mask.repeat(n_head, attn_mask.size(2), 1)
            attn_score.masked_fill_(~attn_mask, -float('inf'))

        attn_prob = F.softmax(attn_score, dim=2)
        attn_vec = torch.bmm(attn_prob, v)

        attn_vec = attn_vec.view(n_head, inp.size(0), inp.size(1), d_head)
        attn_vec = attn_vec.permute(1, 2, 0, 3).contiguous().view(
            inp.size(0), inp.size(1), n_head * d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)

        # residual connection + layer normalization
        output = self.layer_norm(attn_out + residual)

        return output


class TransformerLayer(nn.Module):
    def __init__(self, n_head, n_text_dim, d_head, d_inner, kernel_size, **kwargs):
        super(TransformerLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, n_text_dim, d_head, **kwargs)
        self.pos_ff = PositionwiseConvFF(n_text_dim, d_inner, kernel_size)

    def forward(self, dec_inp, mask=None):
        output = self.dec_attn(dec_inp, attn_mask=mask)
        output = output * mask[..., None] if mask is not None else output
        output = self.pos_ff(output)
        output = output * mask[..., None] if mask is not None else output
        return output


class FFTransformer(nn.Module):
    def __init__(self, n_text_dim, n_layers=6, n_head=2, d_head=64,
                 d_inner=1024, kernel_size=3):
        super(FFTransformer, self).__init__()
        self.pos_emb = PositionalEmbedding(n_text_dim)
        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(
                TransformerLayer(n_head, n_text_dim, d_head, d_inner, kernel_size))

    def forward(self, token_embedding, in_lens):
        if in_lens is not None:
            mask = get_mask_from_lengths(in_lens)
            text_embedding = self.pos_emb(token_embedding) * mask[:, None]
        else:
            mask = None
            text_embedding = self.pos_emb(token_embedding)

        text_embedding = text_embedding.permute(0, 2, 1)

        for layer in self.layers:
            text_embedding = layer(text_embedding, mask=mask)

        if in_lens is not None:
            text_embedding = text_embedding * mask[..., None]

        return text_embedding

    def infer(self, token_embedding):
        text_embedding = self.pos_emb(token_embedding)
        text_embedding = text_embedding.permute(0, 2, 1)

        for layer in self.layers:
            text_embedding = layer(text_embedding)

        return text_embedding
