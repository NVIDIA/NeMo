# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from torch import nn
from torch.nn import functional as F

from nemo.collections.asr.data.audio_to_text import AudioToCharWithDursF0Dataset


class GaussianEmbedding(nn.Module):
    """Gaussian embedding layer.."""

    EPS = 1e-6

    def __init__(
        self, vocab, d_emb, sigma_c=2.0, merge_blanks=False,
    ):
        super().__init__()

        self.embed = nn.Embedding(len(vocab.labels), d_emb)
        self.pad = vocab.pad
        self.sigma_c = sigma_c
        self.merge_blanks = merge_blanks

    def forward(self, text, durs):
        """See base class."""
        # Fake padding
        text = F.pad(text, [0, 2, 0, 0], value=self.pad)
        durs = F.pad(durs, [0, 2, 0, 0], value=0)

        repeats = AudioToCharWithDursF0Dataset.repeat_merge(text, durs, self.pad)
        total_time = repeats.shape[-1]

        # Centroids: [B,T,N]
        c = (durs / 2.0) + F.pad(torch.cumsum(durs, dim=-1)[:, :-1], [1, 0, 0, 0], value=0)
        c = c.unsqueeze(1).repeat(1, total_time, 1)

        # Sigmas: [B,T,N]
        sigmas = durs
        sigmas = sigmas.float() / self.sigma_c
        sigmas = sigmas.unsqueeze(1).repeat(1, total_time, 1) + self.EPS
        assert c.shape == sigmas.shape

        # Times at indexes
        t = torch.arange(total_time, device=c.device).view(1, -1, 1).repeat(durs.shape[0], 1, durs.shape[-1]).float()
        t = t + 0.5

        ns = slice(None)
        if self.merge_blanks:
            ns = slice(1, None, 2)

        # Weights: [B,T,N]
        d = torch.distributions.normal.Normal(c, sigmas)
        w = d.log_prob(t).exp()[:, :, ns]  # [B,T,N]
        pad_mask = (text == self.pad)[:, ns].unsqueeze(1).repeat(1, total_time, 1)
        w.masked_fill_(pad_mask, 0.0)  # noqa
        w = w / (w.sum(-1, keepdim=True) + self.EPS)
        pad_mask = (repeats == self.pad).unsqueeze(-1).repeat(1, 1, text[:, ns].size(1))  # noqa
        w.masked_fill_(pad_mask, 0.0)  # noqa
        pad_mask[:, :, :-1] = False
        w.masked_fill_(pad_mask, 1.0)  # noqa

        # Embeds
        u = torch.bmm(w, self.embed(text)[:, ns, :])  # [B,T,E]

        return u


class MaskedInstanceNorm1d(nn.Module):
    """Instance norm + masking."""

    MAX_CNT = 1e5

    def __init__(self, d_channel: int, unbiased: bool = True, affine: bool = False):
        super().__init__()

        self.d_channel = d_channel
        self.unbiased = unbiased

        self.affine = affine
        if self.affine:
            gamma = torch.ones(d_channel, dtype=torch.float)
            beta = torch.zeros_like(gamma)
            self.register_parameter('gamma', nn.Parameter(gamma))
            self.register_parameter('beta', nn.Parameter(beta))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:  # noqa
        """`x`: [B,C,T], `x_mask`: [B,T] => [B,C,T]."""
        x_mask = x_mask.unsqueeze(1).type_as(x)  # [B,1,T]
        cnt = x_mask.sum(dim=-1, keepdim=True)  # [B,1,1]

        # Mean: [B,C,1]
        cnt_for_mu = cnt.clamp(1.0, self.MAX_CNT)
        mu = (x * x_mask).sum(dim=-1, keepdim=True) / cnt_for_mu

        # Variance: [B,C,1]
        sigma = (x - mu) ** 2
        cnt_fot_sigma = (cnt - int(self.unbiased)).clamp(1.0, self.MAX_CNT)
        sigma = (sigma * x_mask).sum(dim=-1, keepdim=True) / cnt_fot_sigma
        sigma = (sigma + 1e-8).sqrt()

        y = (x - mu) / sigma

        if self.affine:
            gamma = self.gamma.unsqueeze(0).unsqueeze(-1)
            beta = self.beta.unsqueeze(0).unsqueeze(-1)
            y = y * gamma + beta

        return y


class StyleResidual(nn.Module):
    """Styling."""

    def __init__(self, d_channel: int, d_style: int, kernel_size: int = 1):
        super().__init__()

        self.rs = nn.Conv1d(
            in_channels=d_style, out_channels=d_channel, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
        )

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """`x`: [B,C,T], `s`: [B,S,T] => [B,C,T]."""
        return x + self.rs(s)
