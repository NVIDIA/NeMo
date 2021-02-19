import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from nemo.collections.tts.helpers.helpers import get_mask_from_lengths


class FFTBlocks(nn.Module):
    def __init__(
        self,
        name,
        max_seq_len,
        n_layers=4,
        n_head=2,
        d_k=64,
        d_v=64,
        d_model=256,
        d_inner=1024,
        d_word_vec=256,
        fft_conv1d_kernel_1=9,
        fft_conv1d_kernel_2=1,
        fft_conv1d_padding_1=4,
        fft_conv1d_padding_2=0,
        dropout=0.2,
        fused_layernorm=False,
        use_amp=False,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_word_vec = d_word_vec
        self.d_inner = d_inner
        self.fft_conv1d_kernel_1 = fft_conv1d_kernel_1
        self.fft_conv1d_kernel_2 = fft_conv1d_kernel_2
        self.fft_conv1d_padding_1 = fft_conv1d_padding_1
        self.fft_conv1d_padding_2 = fft_conv1d_padding_2
        self.droupout = dropout
        self.fused_layernorm = fused_layernorm
        self.name = name

        n_position = max_seq_len + 1
        self.position = nn.Embedding.from_pretrained(
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
                    fft_conv1d_kernel_1=fft_conv1d_kernel_1,
                    fft_conv1d_kernel_2=fft_conv1d_kernel_2,
                    fft_conv1d_padding_1=fft_conv1d_padding_1,
                    fft_conv1d_padding_2=fft_conv1d_padding_2,
                    dropout=dropout,
                    fused_layernorm=fused_layernorm,
                    use_amp=use_amp,
                    name="{}.layer_stack.{}".format(self.name, i),
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, seq, lengths, return_attns=False, acts=None):

        slf_attn_list = []
        non_pad_mask = get_mask_from_lengths(lengths, max_len=seq.size(1))

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=non_pad_mask, seq_q=non_pad_mask)  # (b, t, t)
        non_pad_mask = non_pad_mask.unsqueeze(-1)

        # -- Forward
        seq_length = seq.size(1)
        if seq_length > self.max_seq_len:
            raise ValueError(
                f"Input sequence is longer than maximum allowed sequence length for positional encoding. "
                f"Got {seq_length} and {self.max_seq_len}"
            )
        position_ids = torch.arange(start=0, end=0 + seq_length, dtype=torch.long, device=seq.device)
        position_ids = position_ids.unsqueeze(0).expand(seq.size(0), -1)
        pos_enc = self.position(position_ids) * non_pad_mask
        output = seq + pos_enc

        if acts is not None:
            acts["act.{}.add_pos_enc".format(self.name)] = output

        for i, layer in enumerate(self.layer_stack):
            output, slf_attn = layer(output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask, acts=acts)
            if return_attns:
                slf_attn_list += [slf_attn]

            if acts is not None:
                acts['act.{}.layer_stack.{}'.format(self.name, i)] = output

        return output, non_pad_mask


class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(
        self,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        fft_conv1d_kernel_1,
        fft_conv1d_kernel_2,
        fft_conv1d_padding_1,
        fft_conv1d_padding_2,
        dropout,
        name,
        fused_layernorm=False,
        use_amp=False,
    ):
        super(FFTBlock, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.fft_conv1d_kernel_1 = fft_conv1d_kernel_1
        self.fft_conv1d_kernel_2 = fft_conv1d_kernel_2
        self.fft_conv1d_padding_1 = fft_conv1d_padding_1
        self.fft_conv1d_padding_2 = fft_conv1d_padding_2
        self.droupout = dropout
        self.name = name
        self.fused_layernorm = fused_layernorm

        self.slf_attn = MultiHeadAttention(
            n_head=n_head,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            name="{}.slf_attn".format(name),
            fused_layernorm=fused_layernorm,
            use_amp=use_amp,
        )

        self.pos_ffn = PositionwiseFeedForward(
            d_in=d_model,
            d_hid=d_inner,
            fft_conv1d_kernel_1=fft_conv1d_kernel_1,
            fft_conv1d_kernel_2=fft_conv1d_kernel_2,
            fft_conv1d_padding_1=fft_conv1d_padding_1,
            fft_conv1d_padding_2=fft_conv1d_padding_2,
            dropout=dropout,
            name="{}.pos_ffn".format(name),
            fused_layernorm=fused_layernorm,
            use_amp=use_amp,
        )

    def forward(self, _input, non_pad_mask=None, slf_attn_mask=None, acts=None):
        output, slf_attn = self.slf_attn(_input, mask=slf_attn_mask, acts=acts)

        output *= non_pad_mask.to(output.dtype)

        output = self.pos_ffn(output, acts=acts)
        output *= non_pad_mask.to(output.dtype)

        return output, slf_attn


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

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

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)  # (b, t)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # (b, t, t)

    return padding_mask


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout, name, fused_layernorm=False, use_amp=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.name = name
        self.fused_layernorm = fused_layernorm
        self.use_amp = use_amp

        d_out = d_k + d_k + d_v
        self.linear = nn.Linear(d_model, n_head * d_out)
        nn.init.xavier_normal_(self.linear.weight)

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5), name="{}.scaled_dot".format(self.name)
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, acts=None):
        bs, seq_len, _ = x.size()

        residual = x

        d_out = self.d_k + self.d_k + self.d_v
        x = self.linear(x)  # (b, t, n_heads * h)

        if acts is not None:
            acts['act.{}.linear'.format(self.name)] = x

        x = x.view(bs, seq_len, self.n_head, d_out)  # (b, t, n_heads, h)
        x = x.permute(2, 0, 1, 3).contiguous().view(self.n_head * bs, seq_len, d_out)  # (n * b, t, h)

        q = x[..., : self.d_k]  # (n * b, t, d_k)
        k = x[..., self.d_k : 2 * self.d_k]  # (n * b, t, d_k)
        v = x[..., 2 * self.d_k :]  # (n * b, t, d_k)

        mask = mask.repeat(self.n_head, 1, 1)  # (b, t, h) -> (n * b, t, h)

        output, attn = self.attention(q, k, v, mask=mask, acts=acts)

        output = output.view(self.n_head, bs, seq_len, self.d_v)  # (n, b, t, d_k)
        output = output.permute(1, 2, 0, 3).contiguous().view(bs, seq_len, self.n_head * self.d_v)  # (b, t, n * d_k)

        if acts is not None:
            acts['act.{}.scaled_dot'.format(self.name)] = output

        output = self.fc(output)

        output = self.dropout(output)

        output += residual

        if acts is not None:
            acts['act.{}.residual'.format(self.name)] = output

        output = self.layer_norm(output)

        if acts is not None:
            acts['act.{}.ln'.format(self.name)] = output

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(
        self,
        d_in,
        d_hid,
        fft_conv1d_kernel_1,
        fft_conv1d_kernel_2,
        fft_conv1d_padding_1,
        fft_conv1d_padding_2,
        dropout,
        name,
        fused_layernorm=False,
        use_amp=False,
    ):
        super().__init__()

        self.name = name
        self.fused_layernorm = fused_layernorm
        self.use_amp = use_amp

        self.w_1 = nn.Conv1d(d_in, d_hid, kernel_size=fft_conv1d_kernel_1, padding=fft_conv1d_padding_1)

        self.w_2 = nn.Conv1d(d_hid, d_in, kernel_size=fft_conv1d_kernel_2, padding=fft_conv1d_padding_2)

        self.layer_norm = nn.LayerNorm(d_in)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, acts=None):
        residual = x

        output = x.transpose(1, 2)
        output = self.w_1(output)

        if acts is not None:
            acts['act.{}.conv1'.format(self.name)] = output

        output = F.relu(output)
        output = self.w_2(output)
        if acts is not None:
            acts['act.{}.conv2'.format(self.name)] = output

        output = output.transpose(1, 2)
        output = self.dropout(output)

        output += residual

        if acts is not None:
            acts['act.{}.residual'.format(self.name)] = output

        if self.fused_layernorm and self.use_amp:
            from torch.cuda import amp

            with amp.autocast(enabled=False):
                output = output.float()
                output = self.layer_norm(output)
                output = output.half()
        else:
            output = self.layer_norm(output)

        if acts is not None:
            acts['act.{}.ln'.format(self.name)] = output

        return output


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1, name=None):
        super().__init__()

        self.temperature = temperature
        self.name = name

        self.bmm1 = Bmm()
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.bmm2 = Bmm()

    def forward(self, q, k, v, mask=None, acts=None):

        attn = self.bmm1(q, k.transpose(1, 2))

        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -65504)

        attn = self.softmax(attn)

        attn = self.dropout(attn)

        output = self.bmm2(attn, v)

        return output, attn


class Bmm(nn.Module):
    """ Required for manual fp16 casting. If not using amp_opt_level='O2', just use torch.bmm.
    """

    def forward(self, a, b):
        return torch.bmm(a, b)
