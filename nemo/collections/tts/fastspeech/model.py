import collections
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

from nemo.backends.pytorch import nm as nemo_nm
from nemo.collections.tts.fastspeech import transformer
from nemo.core.neural_types import EmbeddedTextType, LengthsType, MaskType, MelSpectrogramType, NeuralType


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, encoder_output_size, duration_predictor_filter_size, duration_predictor_kernel_size, dropout):
        super(LengthRegulator, self).__init__()

        self.duration_predictor = DurationPredictor(
            input_size=encoder_output_size,
            filter_size=duration_predictor_filter_size,
            kernel=duration_predictor_kernel_size,
            conv_output_size=duration_predictor_filter_size,
            dropout=dropout,
        )

    def forward(self, encoder_output, encoder_output_mask, target=None, alpha=1.0, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(encoder_output, encoder_output_mask)

        if self.training:
            output, dec_pos = self.get_output(encoder_output, target, alpha, mel_max_length)
        else:
            duration_predictor_output = torch.clamp_min(torch.exp(duration_predictor_output) - 1, 0)

            output, dec_pos = self.get_output(encoder_output, duration_predictor_output, alpha)

        return output, dec_pos, duration_predictor_output

    @staticmethod
    def get_output(encoder_output, duration_predictor_output, alpha, mel_max_length=None):
        output = list()
        dec_pos = list()

        for i in range(encoder_output.size(0)):
            repeats = duration_predictor_output[i].float() * alpha
            repeats = torch.round(repeats).long()
            output.append(torch.repeat_interleave(encoder_output[i], repeats, dim=0))
            dec_pos.append(torch.from_numpy(np.indices((output[i].shape[0],))[0] + 1))

        output = torch.nn.utils.rnn.pad_sequence(output, batch_first=True)
        dec_pos = torch.nn.utils.rnn.pad_sequence(dec_pos, batch_first=True)

        dec_pos = dec_pos.to(output.device, non_blocking=True)

        if mel_max_length:
            output = output[:, :mel_max_length]
            dec_pos = dec_pos[:, :mel_max_length]

        return output, dec_pos


class DurationPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, input_size, filter_size, kernel, conv_output_size, dropout):
        super(DurationPredictor, self).__init__()

        self.input_size = input_size
        self.filter_size = filter_size
        self.kernel = kernel
        self.conv_output_size = conv_output_size
        self.dropout = dropout

        self.conv_layer = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "conv1d_1",
                        transformer.Conv(self.input_size, self.filter_size, kernel_size=self.kernel, padding=1),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        transformer.Conv(self.filter_size, self.filter_size, kernel_size=self.kernel, padding=1),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1, bias=True)

    def forward(self, encoder_output, encoder_output_mask):
        encoder_output = encoder_output * encoder_output_mask

        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out * encoder_output_mask
        out = out.squeeze(-1)

        return out


class FastSpeech(nemo_nm.TrainableNM):
    """FastSpeech Model.

    Attributes:
        decoder_output_size: Output size for decoder.
        n_mels: Number of features for mel spectrogram.
        max_seq_len: Maximum length of input sequence.
        word_vec_dim: Dimensionality of word embedding vector.
        encoder_n_layer: Number of layers for encoder.
        encoder_head: Number of heads for encoder.
        encoder_conv1d_filter_size: Filter size for encoder convolutions.
        decoder_n_layer: Number of layers for decoder.
        decoder_head: Number of heads for decoder.
        decoder_conv1d_filter_size: Filter size for decoder convolutions.
        fft_conv1d_kernel: Kernel size for FFT.
        fft_conv1d_padding: Padding for FFT.
        encoder_output_size: Output size for encoder.
        duration_predictor_filter_size: Predictor filter size.
        duration_predictor_kernel_size: Predictor kernel size.
        dropout: Dropout probability.
        Alpha: Predictor loss coeficient.

    """

    # noinspection DuplicatedCode
    @property
    def input_ports(self) -> Optional[Dict[str, NeuralType]]:
        return dict(
            text=NeuralType(('B', 'T'), EmbeddedTextType()),
            text_pos=NeuralType(('B', 'T'), MaskType()),
            mel_true=NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            dur_true=NeuralType(('B', 'T'), LengthsType()),
        )

    @property
    def output_ports(self) -> Optional[Dict[str, NeuralType]]:
        return dict(
            mel_pred=NeuralType(('B', 'D', 'T'), MelSpectrogramType()), dur_pred=NeuralType(('B', 'T'), LengthsType()),
        )

    def __init__(
        self,
        decoder_output_size: int,
        n_mels: int,
        max_seq_len: int,
        word_vec_dim: int,
        encoder_n_layer: int,
        encoder_head: int,
        encoder_conv1d_filter_size: int,
        decoder_n_layer: int,
        decoder_head: int,
        decoder_conv1d_filter_size: int,
        fft_conv1d_kernel: int,
        fft_conv1d_padding: int,
        encoder_output_size: int,
        duration_predictor_filter_size: int,
        duration_predictor_kernel_size: int,
        dropout: float,
        alpha: float,
        n_src_vocab: int,
        pad_id: int,
    ):
        super().__init__()

        self.encoder = transformer.Encoder(
            len_max_seq=max_seq_len,
            d_word_vec=word_vec_dim,
            n_layers=encoder_n_layer,
            n_head=encoder_head,
            d_k=64,
            d_v=64,
            d_model=word_vec_dim,
            d_inner=encoder_conv1d_filter_size,
            fft_conv1d_kernel=fft_conv1d_kernel,
            fft_conv1d_padding=fft_conv1d_padding,
            dropout=dropout,
            n_src_vocab=n_src_vocab,
            pad_id=pad_id,
        ).to(self._device)
        self.length_regulator = LengthRegulator(
            encoder_output_size, duration_predictor_filter_size, duration_predictor_kernel_size, dropout
        ).to(self._device)

        self.decoder = transformer.Decoder(
            len_max_seq=max_seq_len,
            d_word_vec=word_vec_dim,
            n_layers=decoder_n_layer,
            n_head=decoder_head,
            d_k=64,
            d_v=64,
            d_model=word_vec_dim,
            d_inner=decoder_conv1d_filter_size,
            fft_conv1d_kernel=fft_conv1d_kernel,
            fft_conv1d_padding=fft_conv1d_padding,
            dropout=dropout,
            pad_id=pad_id,
        ).to(self._device)

        self.mel_linear = nn.Linear(decoder_output_size, n_mels, bias=True).to(self._device)
        self.alpha = alpha

    def forward(self, text, text_pos, mel_true=None, dur_true=None):
        encoder_output, encoder_mask = self.encoder(text, text_pos)

        if self.training:
            mel_max_length = mel_true.shape[2]
            length_regulator_output, decoder_pos, dur_pred = self.length_regulator(
                encoder_output, encoder_mask, dur_true, self.alpha, mel_max_length
            )

            assert length_regulator_output.shape[1] <= mel_max_length

        else:
            length_regulator_output, decoder_pos, dur_pred = self.length_regulator(
                encoder_output, encoder_mask, alpha=self.alpha
            )

        decoder_output, decoder_mask = self.decoder(length_regulator_output, decoder_pos)
        mel_pred = self.mel_linear(decoder_output).transpose(1, 2)

        assert mel_pred.shape[2] == dur_true.sum(-1).max()
        assert mel_true.shape[2] == dur_true.sum(-1).max(), f'{mel_true.shape} {dur_true.sum(-1).max()}'

        return mel_pred, dur_pred
