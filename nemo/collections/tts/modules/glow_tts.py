import math

import torch
import torch.nn.functional as F
from torch import nn

from nemo.collections.tts.parts import glow_tts
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import (
    AcousticEncodedRepresentation,
    IntType,
    LengthsType,
    LogDeterminantType,
    MaskType,
    MelSpectrogramType,
    NormalDistributionLogVarianceType,
    NormalDistributionMeanType,
    NormalDistributionSamplesType,
    SequenceToSequenceAlignmentType,
    TokenIndex,
    TokenLogDurationType,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils.decorators import experimental


@experimental
class TextEncoder(NeuralModule):
    def __init__(
        self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        filter_channels_dp,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        window_size,
        mean_only=False,
        prenet=False,
        gin_channels=0,
    ):
        """
        GlowTTS text encoder. Takes in the input text tokens and produces prior distribution statistics for the latent
        representation corresponding to each token, as well as the log durations (the Duration Predictor is also part of
         this module).
        Architecture is similar to Transformer TTS with slight modifications.
        Args:
            n_vocab: Number of tokens in the vocabulary
            out_channels: Latent representation channels
            hidden_channels: Number of channels in the intermediate representations
            filter_channels: Number of channels for the representations in the feed-forward layer
            filter_channels_dp: Number of channels for the representations in the duration predictor
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            kernel_size: Kernels size for the feed-forward layer
            p_dropout: Dropout probability
            mean_only: Return zeros for logs if true
            prenet: Use an additional network before the transformer modules
            gin_channels: Number of channels in speaker embeddings
        """
        super().__init__()

        self.n_layers = n_layers
        self.hidden_channels = hidden_channels
        self.prenet = prenet
        self.mean_only = mean_only

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        if prenet:
            self.pre = glow_tts.ConvReluNorm(
                hidden_channels, hidden_channels, hidden_channels, kernel_size=5, n_layers=3, p_dropout=0.5,
            )

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for i in range(self.n_layers):
            self.attn_layers.append(
                glow_tts.AttentionBlock(
                    hidden_channels, hidden_channels, n_heads, window_size=window_size, p_dropout=p_dropout,
                )
            )
            self.norm_layers_1.append(glow_tts.LayerNorm(hidden_channels))
            self.ffn_layers.append(
                glow_tts.FeedForwardNetwork(
                    hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout
                )
            )
            self.norm_layers_2.append(glow_tts.LayerNorm(hidden_channels))

        self.proj_m = nn.Conv1d(hidden_channels, out_channels, 1)
        if not mean_only:
            self.proj_s = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj_w = glow_tts.DurationPredictor(
            hidden_channels + gin_channels, filter_channels_dp, kernel_size, p_dropout
        )

    @property
    def input_types(self):
        return {
            "text": NeuralType(('B', 'T'), TokenIndex()),
            "text_lengths": NeuralType(('B',), LengthsType()),
            "speaker_embeddings": NeuralType(('B', 'D'), AcousticEncodedRepresentation(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "x_m": NeuralType(('B', 'D', 'T'), NormalDistributionMeanType()),
            "x_logs": NeuralType(('B', 'D', 'T'), NormalDistributionLogVarianceType()),
            "logw": NeuralType(('B', 'T'), TokenLogDurationType()),
            "x_mask": NeuralType(('B', 'D', 'T'), MaskType()),
        }

    @typecheck()
    def forward(self, text, text_lengths, speaker_embeddings=None):

        x = self.emb(text) * math.sqrt(self.hidden_channels)  # [b, t, h]

        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(glow_tts.sequence_mask(text_lengths, x.size(2)), 1).to(x.dtype)

        if self.prenet:
            x = self.pre(x, x_mask)

        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.n_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask

        if speaker_embeddings is not None:
            g_exp = speaker_embeddings.expand(-1, -1, x.size(-1))
            x_dp = torch.cat([torch.detach(x), g_exp], 1)
        else:
            x_dp = torch.detach(x)

        x_m = self.proj_m(x) * x_mask
        if not self.mean_only:
            x_logs = self.proj_s(x) * x_mask
        else:
            x_logs = torch.zeros_like(x_m)

        logw = self.proj_w(spect=x_dp, mask=x_mask)

        return x_m, x_logs, logw, x_mask

    def save_to(self, save_path: str):
        """TODO: Implement"""
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        """TODO: Implement"""
        pass


@experimental
class FlowSpecDecoder(NeuralModule):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_blocks,
        n_layers,
        p_dropout=0.0,
        n_split=4,
        n_sqz=2,
        sigmoid_scale=False,
        gin_channels=0,
    ):
        """
        Flow-based invertible decoder for GlowTTS. Converts spectrograms to latent representations and back.
        Args:
            in_channels:
            hidden_channels:
            kernel_size:
            dilation_rate:
            n_blocks:
            n_layers:
            p_dropout:
            n_split:
            n_sqz:
            sigmoid_scale:
        """
        super().__init__()

        self.n_sqz = n_sqz

        self.flows = nn.ModuleList()
        for b in range(n_blocks):
            self.flows.append(glow_tts.ActNorm(channels=in_channels * n_sqz))
            self.flows.append(glow_tts.InvConvNear(channels=in_channels * n_sqz, n_split=n_split))
            self.flows.append(
                glow_tts.CouplingBlock(
                    in_channels * n_sqz,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    n_layers=n_layers,
                    p_dropout=p_dropout,
                    sigmoid_scale=sigmoid_scale,
                    gin_channels=gin_channels,
                )
            )

    @property
    def input_types(self):
        return {
            "spect": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "spect_mask": NeuralType(('B', 'D', 'T'), MaskType()),
            "speaker_embeddings": NeuralType(('B', 'D'), AcousticEncodedRepresentation(), optional=True),
            "reverse": NeuralType(elements_type=IntType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "z": NeuralType(('B', 'D', 'T'), NormalDistributionSamplesType()),
            "logdet_tot": NeuralType(('B',), LogDeterminantType()),
        }

    @typecheck()
    def forward(self, spect, spect_mask, speaker_embeddings=None, reverse=False):
        if not reverse:
            flows = self.flows
            logdet_tot = 0
        else:
            flows = reversed(self.flows)
            logdet_tot = None

        x = spect
        x_mask = spect_mask

        if self.n_sqz > 1:
            x, x_mask = self.squeeze(x, x_mask, self.n_sqz)
        for f in flows:
            if not reverse:
                x, logdet = f(x, x_mask, g=speaker_embeddings, reverse=reverse)
                logdet_tot += logdet
            else:
                x, logdet = f(x, x_mask, g=speaker_embeddings, reverse=reverse)
        if self.n_sqz > 1:
            x, x_mask = self.unsqueeze(x, x_mask, self.n_sqz)
        return x, logdet_tot

    def squeeze(self, x, x_mask=None, n_sqz=2):
        b, c, t = x.size()

        t = (t // n_sqz) * n_sqz
        x = x[:, :, :t]
        x_sqz = x.view(b, c, t // n_sqz, n_sqz)
        x_sqz = x_sqz.permute(0, 3, 1, 2).contiguous().view(b, c * n_sqz, t // n_sqz)

        if x_mask is not None:
            x_mask = x_mask[:, :, n_sqz - 1 :: n_sqz]
        else:
            x_mask = torch.ones(b, 1, t // n_sqz).to(device=x.device, dtype=x.dtype)
        return x_sqz * x_mask, x_mask

    def unsqueeze(self, x, x_mask=None, n_sqz=2):
        b, c, t = x.size()

        x_unsqz = x.view(b, n_sqz, c // n_sqz, t)
        x_unsqz = x_unsqz.permute(0, 2, 3, 1).contiguous().view(b, c // n_sqz, t * n_sqz)

        if x_mask is not None:
            x_mask = x_mask.unsqueeze(-1).repeat(1, 1, 1, n_sqz).view(b, 1, t * n_sqz)
        else:
            x_mask = torch.ones(b, 1, t * n_sqz).to(device=x.device, dtype=x.dtype)
        return x_unsqz * x_mask, x_mask

    def store_inverse(self):
        for f in self.flows:
            f.store_inverse()

    def save_to(self, save_path: str):
        """TODO: Implement"""
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        """TODO: Implement"""
        pass


class GlowTTSModule(NeuralModule):
    def __init__(self, encoder_module, decoder_module, n_speakers=1, gin_channels=0):
        super().__init__()

        self.encoder = encoder_module
        self.decoder = decoder_module

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
            nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)

    @property
    def input_types(self):
        return {
            "text": NeuralType(('B', 'T'), TokenIndex()),
            "text_lengths": NeuralType(('B',), LengthsType()),
            "spect": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "spect_lengths": NeuralType(('B',), LengthsType()),
            "speaker": NeuralType(('B',), IntType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "z": NeuralType(('B', 'D', 'T'), NormalDistributionSamplesType()),
            "y_m": NeuralType(('B', 'D', 'T'), NormalDistributionMeanType()),
            "y_logs": NeuralType(('B', 'D', 'T'), NormalDistributionLogVarianceType()),
            "logdet": NeuralType(('B',), LogDeterminantType()),
            "log_durs_predicted": NeuralType(('B', 'T'), TokenLogDurationType()),
            "log_durs_extracted": NeuralType(('B', 'T'), TokenLogDurationType()),
            "spect_lengths": NeuralType(('B',), LengthsType()),
            "attn": NeuralType(('B', 'T', 'T'), SequenceToSequenceAlignmentType()),
        }

    @typecheck()
    def forward(self, text, text_lengths, spect, spect_lengths, g=None):

        if g is not None:
            g = F.normalize(self.emb_g(g)).unsqueeze(-1)  # [b, h]

        x_m, x_logs, log_durs_predicted, x_mask = self.encoder(
            text=text, text_lengths=text_lengths, speaker_embeddings=g
        )

        y_max_length = spect.size(2)
        y_max_length = (y_max_length // self.decoder.n_sqz) * self.decoder.n_sqz
        spect = spect[:, :, :y_max_length]

        spect_lengths = (spect_lengths // self.decoder.n_sqz) * self.decoder.n_sqz

        y_mask = torch.unsqueeze(glow_tts.sequence_mask(spect_lengths, y_max_length), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)

        z, logdet = self.decoder(spect=spect, spect_mask=y_mask, speaker_embeddings=g, reverse=False)

        with torch.no_grad():
            x_s_sq_r = torch.exp(-2 * x_logs)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(-1)  # [b, t, 1]
            logp2 = torch.matmul(x_s_sq_r.transpose(1, 2), -0.5 * (z ** 2))  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp3 = torch.matmul((x_m * x_s_sq_r).transpose(1, 2), z)  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp4 = torch.sum(-0.5 * (x_m ** 2) * x_s_sq_r, [1]).unsqueeze(-1)  # [b, t, 1]
            logp = logp1 + logp2 + logp3 + logp4  # [b, t, t']

            attn = (glow_tts.maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()).squeeze(1)

        y_m = torch.matmul(x_m, attn)
        y_logs = torch.matmul(x_logs, attn)

        log_durs_extracted = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask.squeeze()

        return z, y_m, y_logs, logdet, log_durs_predicted, log_durs_extracted, spect_lengths, attn

    def generate_spect(self, text, text_lengths, g=None, noise_scale=0.3, length_scale=1.0):

        if g is not None:
            g = F.normalize(self.emb_g(g)).unsqueeze(-1)  # [b, h]

        x_m, x_logs, log_durs_predicted, x_mask = self.encoder(
            text=text, text_lengths=text_lengths, speaker_embeddings=g
        )

        w = torch.exp(log_durs_predicted) * x_mask.squeeze() * length_scale
        w_ceil = torch.ceil(w)
        spect_lengths = torch.clamp_min(torch.sum(w_ceil, [1]), 1).long()
        y_max_length = None

        spect_lengths = (spect_lengths // self.decoder.n_sqz) * self.decoder.n_sqz

        y_mask = torch.unsqueeze(glow_tts.sequence_mask(spect_lengths, y_max_length), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)

        attn = glow_tts.generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1))

        y_m = torch.matmul(x_m, attn)
        y_logs = torch.matmul(x_logs, attn)

        z = (y_m + torch.exp(y_logs) * torch.randn_like(y_m) * noise_scale) * y_mask
        y, _ = self.decoder(spect=z, spect_mask=y_mask, speaker_embeddings=g, reverse=True)

        return y, attn

    def save_to(self, save_path: str):
        """TODO: Implement"""
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        """TODO: Implement"""
        pass
