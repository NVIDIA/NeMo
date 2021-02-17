import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm


import numpy as np
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


class LengthRegulator(nn.Module):
    def forward(self, x, dur):
        output = []
        for x_i, dur_i in zip(x, dur):
            expanded = self.expand(x_i, dur_i)
            output.append(expanded)
        output = self.pad(output)
        return output

    def expand(self, x, dur):
        output = []
        for i, frame in enumerate(x):
            expanded_len = int(dur[i] + 0.5)
            expanded = frame.expand(expanded_len, -1)
            output.append(expanded)
        output = torch.cat(output, 0)
        return output

    def pad(self, x):
        output = []
        max_len = max([x[i].size(0) for i in range(len(x))])
        for i, seq in enumerate(x):
            padded = F.pad(seq, [0, 0, 0, max_len - seq.size(0)], 'constant', 0.0)
            output.append(padded)
        output = torch.stack(output)
        return output


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


LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.02):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class GaussianBlurAugmentation(nn.Module):
    def __init__(self, kernel_size, sigmas, p_blurring):
        super(GaussianBlurAugmentation, self).__init__()
        self.kernel_size = kernel_size
        self.sigmas = sigmas
        kernels = self.initialize_kernels(kernel_size, sigmas)
        self.register_buffer('kernels', kernels)
        self.p_blurring = p_blurring
        self.conv = F.conv2d

    def initialize_kernels(self, kernel_size, sigmas):
        mesh_grids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        kernels = []
        for sigma in sigmas:
            kernel = 1
            sigma = [sigma] * len(kernel_size)
            for size, std, mgrid in zip(kernel_size, sigma, mesh_grids):
                mean = (size - 1) / 2
                kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)

            # Make sure sum of values in gaussian kernel equals 1.
            kernel = kernel / torch.sum(kernel)

            # Reshape to depthwise convolutional weight
            kernel = kernel.view(1, 1, *kernel.size())
            kernel = kernel.repeat(1, *[1] * (kernel.dim() - 1))
            kernels.append(kernel[None])

        kernels = torch.cat(kernels)
        return kernels

    def forward(self, x):
        if np.random.random() > self.p_blurring:
            return x
        else:
            i = np.random.randint(len(self.kernels))
            kernel = self.kernels[i]
            pad = int((self.kernel_size[0] - 1) / 2)
            x = F.pad(x[:, None], (pad, pad, pad, pad), mode='reflect')
            x = self.conv(x, weight=kernel)[:, 0]
            return x


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(
        self,
        gaussian_blur,
        upsample_kernel_sizes,
        upsample_rates,
        upsample_initial_channel,
        # resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(Conv1d(256, upsample_initial_channel, 7, 1, padding=3))
        self.gaussian_blur = gaussian_blur
        if gaussian_blur['p_blurring'] > 0.0:
            self.gaussian_blur_fn = GaussianBlurAugmentation(**gaussian_blur)

        # resblock = ResBlock1 if resblock == '1' else ResBlock2
        resblock = ResBlock1

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

        self.conv_pre.apply(init_weights)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        if self.gaussian_blur['p_blurring'] > 0.0:
            x = self.gaussian_blur_fn(x)
        x = self.conv_pre(x)
        # print(f"x: {x.shape}")
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            # print(f"x: {x.shape}")
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        # print(f"xin: {x.shape}")
        for i, l in enumerate(self.convs):
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            # print(f"x{i}: {x.shape}")
            fmap.append(x)
        x = self.conv_post(x)
        # print(f"x{i}: {x.shape}")
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorP(2), DiscriminatorP(3), DiscriminatorP(5), DiscriminatorP(7), DiscriminatorP(11),]
            # [DiscriminatorP(2), DiscriminatorP(4), DiscriminatorP(8), DiscriminatorP(12)]
        )

    def forward(self, signal):
        score = []
        features = []
        for disc in self.discriminators:
            score_i, features_i = disc(signal)
            score.append(score_i)
            features.append(features_i)

        return score, features


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for i, l in enumerate(self.convs):
            # print(f"x{i}: {x.shape}")
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        # print(f"x2pre {x.shape}")
        x = self.conv_post(x)
        # print(f"x2post {x.shape}")
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorS(use_spectral_norm=True), DiscriminatorS(), DiscriminatorS(),]
        )
        self.meanpools = nn.ModuleList([AvgPool1d(4, 2, padding=1), AvgPool1d(4, 2, padding=1)])

    def forward(self, signal):
        score = []
        features = []
        for i, disc in enumerate(self.discriminators):
            if i != 0:
                signal = self.meanpools[i - 1](signal)
            score_i, features_i = disc(signal)
            score.append(score_i)
            features.append(features_i)
        return score, features


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    # r_losses = []
    # g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += r_loss + g_loss
        # r_losses.append(r_loss.item())
        # g_losses.append(g_loss.item())

    # return loss / len(r_losses), r_losses, g_losses
    return loss / len(disc_real_outputs)


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss / len(gen_losses), gen_losses


def compute_generator_loss(y, y_hat, y_mel, y_hat_mel, mpd, msd):
    loss_mel = F.l1_loss(y_mel, y_hat_mel) * 45
    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_hat)
    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_hat)
    loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
    loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
    loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
    loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
    return loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
