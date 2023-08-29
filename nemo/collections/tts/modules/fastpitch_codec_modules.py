# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from einops import rearrange
from torch import nn
from typing import Optional

from nemo.collections.tts.modules.submodules import ConditionalInput, ConditionalLayerNorm
from nemo.collections.tts.parts.utils.helpers import (
    binarize_attention_parallel,
    get_mask_from_lengths,
    mask_sequence_tensor,
    regulate_len
)
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import (
    EncodedRepresentation,
    FloatType,
    Index,
    LengthsType,
    LogitsType,
    LogprobsType,
    PredictionsType,
    ProbsType,
    TokenDurationType,
    TokenIndex,
    VoidType
)
from nemo.core.neural_types.neural_type import NeuralType


def log_to_duration(log_dur, min_dur, max_dur, mask):
    dur = torch.clamp(torch.exp(log_dur) - 1.0, min_dur, max_dur)
    dur = dur * mask
    return dur


class Conv1d(NeuralModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None
    ):
        super().__init__()
        if not padding:
            padding = kernel_size // 2

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'C', 'T'), VoidType()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'C', 'T'), VoidType()),
        }

    @typecheck()
    def forward(self, inputs, input_len):
        out = self.conv(inputs)
        out = mask_sequence_tensor(out, input_len)
        return out


class ConvReLUNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dropout=0.0, condition_dim=384, condition_types=[]):
        super(ConvReLUNorm, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size // 2))
        self.norm = ConditionalLayerNorm(out_channels, condition_dim=condition_dim, condition_types=condition_types)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, signal, conditioning=None):
        out = torch.nn.functional.relu(self.conv(signal))
        out = self.norm(out.transpose(1, 2), conditioning).transpose(1, 2)
        out = self.dropout(out)

        return out


class TemporalPredictor(NeuralModule):
    """Predicts a single float per each temporal location"""

    def __init__(self, input_size, filter_size, kernel_size, dropout, n_layers=2, condition_types=[]):
        super(TemporalPredictor, self).__init__()
        self.cond_input = ConditionalInput(input_size, input_size, condition_types)
        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                ConvReLUNorm(
                    input_size if i == 0 else filter_size,
                    filter_size,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    condition_dim=input_size,
                    condition_types=condition_types,
                )
            )
        self.fc = torch.nn.Linear(filter_size, 1, bias=True)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "mask": NeuralType(('B', 'T'), TokenDurationType()),
            "conditioning": NeuralType(('B', 'T', 'D'), EncodedRepresentation(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T'), EncodedRepresentation()),
        }

    def forward(self, inputs, mask, conditioning=None):
        mask = rearrange(mask, 'B T -> B T 1')

        enc = self.cond_input(inputs, conditioning=conditioning)
        out = enc * mask
        out = rearrange(out, 'B T D -> B D T')

        for layer in self.layers:
            out = layer(out, conditioning=conditioning)

        out = rearrange(out, 'B D T -> B T D')
        out = self.fc(out)
        out = out * mask
        out = rearrange(out, 'B T 1 -> B T')
        return out


def fused_tanh_sigmoid_activation(inputs):
    t_act = torch.tanh(inputs)
    s_act = torch.sigmoid(inputs)
    out = t_act * s_act
    return out


class WaveNetBlock(NeuralModule):
    def __init__(
        self,
        hidden_channels=192,
        filters=384,
        kernel_size=3,
        dropout_rate=0.1,
        activation="fused_tanh_sigmoid",
        norm_type=None

    ):
        super(WaveNetBlock, self).__init__()

        self.d_model = hidden_channels
        self.dropout = torch.nn.Dropout(dropout_rate)
        if activation == "fused_tanh_sigmoid":
            self.activation = fused_tanh_sigmoid_activation
        elif activation == "elu":
            self.activation = torch.nn.ELU()
        else:
            raise ValueError(f"Unsupported activation {activation}")

        if norm_type is None:
            self.norm = None
        elif norm_type == "layer_norm":
            self.norm = torch.nn.LayerNorm(hidden_channels)
        else:
            raise ValueError(f"Unsupported norm {norm_type}")

        self.input_conv = Conv1d(
            in_channels=hidden_channels,
            out_channels=filters,
            kernel_size=kernel_size
        )
        self.res_conv = Conv1d(
            in_channels=filters,
            out_channels=hidden_channels,
            kernel_size=kernel_size
        )

    @property
    def input_types(self):
        return {
            "input": NeuralType(('B', 'D', 'T'), VoidType()),
            "seq_lens": NeuralType(tuple('B'), LengthsType())
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'C', 'T'), EncodedRepresentation())
        }

    @typecheck()
    def forward(self, inputs, input_len):
        res_input = self.input_conv(inputs=inputs, input_len=input_len)
        res_input = self.activation(res_input)
        res = self.res_conv(inputs=res_input, input_len=input_len)
        res = self.dropout(res)
        out = inputs + res

        if self.norm:
            out = rearrange(out, 'B C T -> B T C')
            out = self.norm(out)
            out = rearrange(out, 'B T C -> B C T')

        return out


class WaveNetDecoder(NeuralModule):
    def __init__(
        self,
        in_channels,
        n_layers=16,
        hidden_channels=192,
        filters=384,
        kernel_size=3,
        dropout_rate=0.1,
        activation="fused_tanh_sigmoid",
        norm_type=None

    ):
        super(WaveNetDecoder, self).__init__()

        self.d_model = hidden_channels
        self.pre_conv = Conv1d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size
        )
        self.wn_layers = torch.nn.ModuleList(
            [
                WaveNetBlock(
                    hidden_channels=hidden_channels,
                    filters=filters,
                    kernel_size=kernel_size,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    norm_type=norm_type
                )
                for _ in range(n_layers)
            ]
        )

    @property
    def input_types(self):
        return {
            "input": NeuralType(('B', 'T', 'D'), VoidType()),
            "seq_lens": NeuralType(tuple('B'), LengthsType()),
            "conditioning": NeuralType(('B', 'T', 'D'), EncodedRepresentation(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "encoded": NeuralType(('B', 'T', 'C'), EncodedRepresentation()),
            "mask": NeuralType(('B', 'T', 'D'), VoidType()),
        }

    @typecheck()
    def forward(self, inputs, input_len):
        out = rearrange(inputs, 'B T D -> B D T')
        out = self.pre_conv(inputs=out, input_len=input_len)
        for layer in self.wn_layers:
            out = layer(input=out, input_len=input_len)

        out = rearrange(out, 'B C T -> B T C')
        return out, None


class FastPitchCodecModule(NeuralModule):
    def __init__(
        self,
        aligner_module: NeuralModule,
        encoder_module: NeuralModule,
        decoder_module: NeuralModule,
        duration_predictor: NeuralModule,
        pitch_predictor: NeuralModule,
        energy_predictor: NeuralModule,
        speaker_encoder: NeuralModule,
        n_codebooks: int,
        codebook_size: int,
        min_token_duration: int = 1,
        max_token_duration: int = 75
    ):
        super().__init__()

        self.aligner = aligner_module
        self.encoder = encoder_module
        self.decoder = decoder_module
        self.duration_predictor = duration_predictor
        self.pitch_predictor = pitch_predictor
        self.energy_predictor = energy_predictor
        self.speaker_encoder = speaker_encoder

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.n_logits = self.n_codebooks * self.codebook_size
        self.hidden_dim = self.encoder.d_model

        self.min_token_duration = min_token_duration
        self.max_token_duration = max_token_duration

        self.pitch_layer = torch.nn.Conv1d(1, self.hidden_dim, kernel_size=3, padding=1)
        self.energy_layer = torch.nn.Conv1d(1, self.hidden_dim, kernel_size=3, padding=1)
        self.audio_token_layer = torch.nn.Linear(self.hidden_dim, self.n_logits)

    def _get_conditioning(self, speaker):
        if speaker is None:
            if self.speaker_encoder is not None:
                raise ValueError("Must provided speaker for speaker encoder.")
            return None

        if self.speaker_encoder is None:
            if speaker is not None:
                raise ValueError(f"Must provide speaker encoder. Received speaker {speaker}.")
            return None

        batch_size = speaker.shape[0]
        spk_emb = self.speaker_encoder(
            batch_size=batch_size, speaker=speaker, reference_spec=None, reference_spec_lens=None
        )
        conditioning = rearrange(spk_emb, "B D -> B 1 D")
        return conditioning

    def _predict_prosody(self, encoder_output, conditioning, mask):
        pitch_pred = self.pitch_predictor(encoder_output, mask=mask, conditioning=conditioning)
        energy_pred = self.energy_predictor(encoder_output, mask=mask, conditioning=conditioning)
        log_durs_pred = self.duration_predictor(encoder_output, mask=mask, conditioning=conditioning)
        return pitch_pred, energy_pred, log_durs_pred

    def _condition_on_pitch(self, inputs, pitch, mask):
        pitch = rearrange(pitch, "B T -> B 1 T")
        pitch_emb = self.pitch_layer(pitch)
        pitch_emb = rearrange(pitch_emb, "B D T -> B T D")

        out = inputs + pitch_emb
        out = out * rearrange(mask, 'B T -> B T 1')
        return out

    def _condition_on_energy(self, inputs, energy, mask):
        energy = rearrange(energy, "B T -> B 1 T")
        energy_emb = self.energy_layer(energy)
        energy_emb = rearrange(energy_emb, "B D T -> B T D")

        out = inputs + energy_emb
        out = out * rearrange(mask, 'B T -> B T 1')
        return out

    def _predict_audio_tokens(self, decoder_output, mask):
        mask = rearrange(mask, 'B T -> B T 1')
        # [batch_size, audio_len, num_codebook * codebook_size]
        audio_logits = self.audio_token_layer(decoder_output)
        audio_logits = audio_logits * mask

        # [batch_size, audio_len, num_codebook, codebook_size]
        logit_shape = (audio_logits.shape[0], audio_logits.shape[1], self.n_codebooks, self.codebook_size)
        audio_logits = torch.reshape(audio_logits, logit_shape)

        # [batch_size, audio_len, num_codebook]
        audio_tokens = audio_logits.max(dim=3).indices
        audio_tokens = audio_tokens * mask

        audio_logits = rearrange(audio_logits, 'B T C W -> B C W T')
        audio_tokens = rearrange(audio_tokens, 'B T C -> B C T')

        return audio_tokens, audio_logits

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_lens": NeuralType(tuple('B'), LengthsType()),
            "audio_codes": NeuralType(('B', 'D', 'T_audio'), EncodedRepresentation()),
            "audio_code_lens": NeuralType(tuple('B'), LengthsType()),
            "attn_prior": NeuralType(('B', 'T_audio', 'T_text'), ProbsType()),
            "speaker": NeuralType(tuple('B'), Index()),
        },
        output_types={
            "durations": NeuralType(('B', 'T_text'), TokenDurationType()),
            "attn_hard": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
            "attn_soft": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
            "attn_logprob": NeuralType(('B', 'S', 'T_audio', 'T_text'), LogprobsType())
        }
    )
    def get_alignments(self, text, text_lens, audio_codes, audio_code_lens, attn_prior=None, speaker=None):
        text_mask = get_mask_from_lengths(text_lens)
        text_mask = rearrange(text_mask, "B T_text -> B T_text 1")
        # Aligner requires an inverted mask
        aligner_text_mask = text_mask == 0
        # [batch_size, 1, hidden_dim]
        cond = self._get_conditioning(speaker=speaker)

        # [batch_size, text_len, hidden_dim]
        text_emb = self.encoder.word_emb(text)
        text_emb = rearrange(text_emb, "B T_text D -> B D T_text")

        # [batch_size, 1, audio_len, text_len]
        attn_soft, attn_logprob = self.aligner(
            queries=audio_codes, keys=text_emb, mask=aligner_text_mask, attn_prior=attn_prior, conditioning=cond
        )
        attn_hard = binarize_attention_parallel(attn=attn_soft, in_lens=text_lens, out_lens=audio_code_lens)

        durations = attn_hard.sum(2)
        durations = rearrange(durations, 'B 1 T_text -> B T_text')
        return durations, attn_hard, attn_soft, attn_logprob

    @property
    def input_types(self):
        return {
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_lens": NeuralType(tuple('B'), LengthsType()),
            "durs": NeuralType(('B', 'T_text'), TokenDurationType()),
            "pitch": NeuralType(('B', 'T_text'), FloatType()),
            "energy": NeuralType(('B', 'T_text'), FloatType()),
            "speaker": NeuralType(tuple('B'), Index(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "audio_tokens": NeuralType(('B', 'C', 'T_token'), TokenIndex()),
            "audio_logits": NeuralType(('B', 'C', 'W', 'T_token'), LogitsType()),
            "log_durs_pred": NeuralType(('B', 'T_text'), PredictionsType()),
            "pitch_pred": NeuralType(('B', 'T_text'), PredictionsType()),
            "energy_pred": NeuralType(('B', 'T_text'), PredictionsType()),
        }

    @typecheck()
    def forward(
        self,
        text,
        text_lens,
        durs,
        pitch,
        energy,
        speaker=None
    ):
        text_mask = get_mask_from_lengths(text_lens)

        # [batch_size, text_len, hidden_dim]
        cond = self._get_conditioning(speaker=speaker)
        enc_out, _ = self.encoder(input=text, conditioning=cond)

        # [batch_size, text_len]
        pitch_pred, energy_pred, log_durs_pred = self._predict_prosody(
            encoder_output=enc_out, conditioning=cond, mask=text_mask
        )

        enc_out = self._condition_on_pitch(inputs=enc_out, pitch=pitch, mask=text_mask)
        enc_out = self._condition_on_energy(inputs=enc_out, energy=energy, mask=text_mask)

        # [batch_size, token_len, hidden_dim], [batch_size]
        enc_out_regulated, audio_token_lens = regulate_len(durs, enc_out, pace=1.0)
        #dec_out, _ = self.decoder(inputs=enc_out_regulated, input_len=audio_token_lens)
        dec_out, _ = self.decoder(input=enc_out_regulated, seq_lens=audio_token_lens, conditioning=cond)

        # [batch_size, audio_len]
        audio_token_mask = get_mask_from_lengths(audio_token_lens)
        # [batch_size, num_codebook, audio_len], [batch_size, num_codebook, codebook_size, audio_len]
        audio_tokens, audio_logits = self._predict_audio_tokens(decoder_output=dec_out, mask=audio_token_mask)

        return audio_tokens, audio_logits, log_durs_pred, pitch_pred, energy_pred

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_lens": NeuralType(tuple('B'), LengthsType()),
            "speaker": NeuralType(tuple('B'), Index(), optional=True),
        },
        output_types={
            "audio_tokens": NeuralType(('B', 'C', 'T_token'), TokenIndex()),
            "audio_token_lens": NeuralType(tuple('B'), LengthsType())
        }
    )
    def infer(self, text, text_lens, speaker=None):
        # [batch_size, text_len]
        text_mask = get_mask_from_lengths(text_lens)

        # [batch_size, text_len, hidden_dim]
        cond = self._get_conditioning(speaker=speaker)
        enc_out, _ = self.encoder(input=text, conditioning=cond)

        # [batch_size, text_len, 1]
        pitch_pred, energy_pred, log_durs_pred = self._predict_prosody(
            encoder_output=enc_out, conditioning=cond, mask=text_mask
        )
        durs_pred = log_to_duration(
            log_dur=log_durs_pred, min_dur=self.min_token_duration, max_dur=self.max_token_duration, mask=text_mask
        )

        # [batch_size, text_len, hidden_dim]
        enc_out = self._condition_on_pitch(inputs=enc_out, pitch=pitch_pred, mask=text_mask)
        enc_out = self._condition_on_energy(inputs=enc_out, energy=energy_pred, mask=text_mask)

        # [batch_size, token_len, hidden_dim], [batch_size]
        enc_out_regulated, audio_token_lens = regulate_len(durs_pred, enc_out, pace=1.0)
        #dec_out, _ = self.decoder(inputs=enc_out_regulated, input_len=audio_token_lens)
        dec_out, _ = self.decoder(input=enc_out_regulated, seq_lens=audio_token_lens, conditioning=cond)

        # [batch_size, audio_len]
        audio_token_mask = get_mask_from_lengths(audio_token_lens)
        # [batch_size, num_codebook, audio_len]
        audio_tokens, _ = self._predict_audio_tokens(decoder_output=dec_out, mask=audio_token_mask)

        return audio_tokens, audio_token_lens