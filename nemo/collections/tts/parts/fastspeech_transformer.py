# Copyright 2020 NVIDIA. All Rights Reserved.
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
from torch import nn

from nemo.collections.nlp.nm.trainables.common.transformer import transformer_modules


def get_non_pad_mask(seq, pad_id):
    assert seq.dim() == 2
    return seq.ne(pad_id).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """Sinusoid position encoding table."""

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.tensor(sinusoid_table).float()


def get_attn_key_pad_mask(seq_k, seq_q, pad_id):
    """For masking out the padding part of key sequence."""

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad_id)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, fft_conv1d_kernel, fft_conv1d_padding, dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = transformer_modules.MultiHeadAttention(d_model, n_head, attn_layer_dropout=dropout)
        self.n_head = n_head
        self.pos_ffn = transformer_modules.PositionWiseFF(
            hidden_size=d_model, inner_size=d_inner, ffn_dropout=dropout,
        )

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        slf_attn_mask = slf_attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)
        enc_output = self.slf_attn(enc_input, enc_input, enc_input, attention_mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output


class FastSpeechTransformerEncoder(nn.Module):
    """Encoder."""

    def __init__(
        self,
        len_max_seq,
        d_word_vec,
        n_layers,
        n_head,
        d_k,
        d_v,
        d_model,
        d_inner,
        fft_conv1d_kernel,
        fft_conv1d_padding,
        dropout,
        n_src_vocab,
        pad_id,
    ):

        super(FastSpeechTransformerEncoder, self).__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_id)
        self.pad_id = pad_id

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0), freeze=True
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model,
                    d_inner,
                    n_head,
                    d_k,
                    d_v,
                    fft_conv1d_kernel=fft_conv1d_kernel,
                    fft_conv1d_padding=fft_conv1d_padding,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, src_pos):
        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq, pad_id=self.pad_id)
        non_pad_mask = get_non_pad_mask(src_seq, pad_id=self.pad_id)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for i, enc_layer in enumerate(self.layer_stack):
            enc_output = enc_layer(enc_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)

        return enc_output, non_pad_mask


class FastSpeechTransformerDecoder(nn.Module):
    """Decoder."""

    def __init__(
        self,
        len_max_seq,
        d_word_vec,
        n_layers,
        n_head,
        d_k,
        d_v,
        d_model,
        d_inner,
        fft_conv1d_kernel,
        fft_conv1d_padding,
        dropout,
        pad_id,
    ):

        super(FastSpeechTransformerDecoder, self).__init__()

        n_position = len_max_seq + 1

        self.position_dec = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0), freeze=True
        )
        self.pad_id = pad_id

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model,
                    d_inner,
                    n_head,
                    d_k,
                    d_v,
                    fft_conv1d_kernel=fft_conv1d_kernel,
                    fft_conv1d_padding=fft_conv1d_padding,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, dec_seq, dec_pos):
        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=dec_pos, seq_q=dec_pos, pad_id=self.pad_id)
        non_pad_mask = get_non_pad_mask(dec_pos, pad_id=self.pad_id)

        # -- Forward
        dec_output = dec_seq + self.position_dec(dec_pos)

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)

        return dec_output, non_pad_mask
