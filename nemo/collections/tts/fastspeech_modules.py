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

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

import nemo
from nemo.backends.pytorch import nm as nemo_nm
from nemo.backends.pytorch.nm import DataLayerNM, LossNM
from nemo.collections.asr.parts import AudioDataset, WaveformFeaturizer
from nemo.collections.tts.parts import fastspeech, fastspeech_transformer
from nemo.core.neural_types import AudioSignal, EmbeddedTextType, LengthsType, MaskType, MelSpectrogramType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['FastSpeechDataLayer', 'FastSpeech', 'FastSpeechLoss']


class FastSpeechDataLayer(DataLayerNM):
    """Data Layer for Fast Speech model.

    Basically, replicated behavior from AudioToText Data Layer, zipped with ground truth durations for additional loss.

    Args:
        manifest_filepath (str): Dataset parameter.
            Path to JSON containing data.
        durs_dir (str): Path to durations arrays directory.
        labels (list): Dataset parameter.
            List of characters that can be output by the ASR model.
            For Jasper, this is the 28 character set {a-z '}. The CTC blank
            symbol is automatically added later for models using ctc.
        batch_size (int): batch size
        sample_rate (int): Target sampling rate for data. Audio files will be
            resampled to sample_rate if it is not already.
            Defaults to 16000.
        int_values (bool): Bool indicating whether the audio file is saved as
            int data or float data.
            Defaults to False.
        eos_id (id): Dataset parameter.
            End of string symbol id used for seq2seq models.
            Defaults to None.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        normalize_transcripts (bool): Dataset parameter.
            Whether to use automatic text cleaning.
            It is highly recommended to manually clean text for best results.
            Defaults to True.
        trim_silence (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        load_audio (bool): Dataset parameter.
            Controls whether the dataloader loads the audio signal and
            transcript or just the transcript.
            Defaults to True.
        drop_last (bool): See PyTorch DataLoader.
            Defaults to False.
        shuffle (bool): See PyTorch DataLoader.
            Defaults to True.
        num_workers (int): See PyTorch DataLoader.
            Defaults to 0.
        perturb_config (dict): Currently disabled.

    """

    @property
    @add_port_docs
    def output_ports(self):
        """Returns definitions of module output ports."""
        return dict(
            audio=NeuralType(('B', 'T'), AudioSignal(freq=self.sample_rate)),
            audio_len=NeuralType(tuple('B'), LengthsType()),
            text=NeuralType(('B', 'T'), EmbeddedTextType()),
            text_pos=NeuralType(('B', 'T'), MaskType()),
            dur_true=NeuralType(('B', 'T'), LengthsType()),
        )

    def __init__(
        self,
        manifest_filepath,
        durs_dir,
        labels,
        batch_size,
        sample_rate=16000,
        int_values=False,
        bos_id=None,
        eos_id=None,
        pad_id=None,
        min_duration=0.1,
        max_duration=None,
        normalize_transcripts=True,
        trim_silence=False,
        load_audio=True,
        drop_last=False,
        shuffle=True,
        num_workers=0,
    ):
        super().__init__()

        # Set up dataset.
        self._featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=None)
        dataset_params = {
            'manifest_filepath': manifest_filepath,
            'labels': labels,
            'featurizer': self._featurizer,
            'max_duration': max_duration,
            'min_duration': min_duration,
            'normalize': normalize_transcripts,
            'trim': trim_silence,
            'bos_id': bos_id,
            'eos_id': eos_id,
            'load_audio': load_audio,
        }
        audio_dataset = AudioDataset(**dataset_params)
        self._dataset = fastspeech.FastSpeechDataset(audio_dataset, durs_dir)
        self._pad_id = pad_id
        self.sample_rate = sample_rate

        sampler = None
        if self._placement == nemo.core.DeviceType.AllGpu:
            sampler = torch.utils.data.distributed.DistributedSampler(self._dataset)

        self._dataloader = torch.utils.data.DataLoader(
            dataset=self._dataset,
            batch_size=batch_size,
            collate_fn=self._collate,
            drop_last=drop_last,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
        )

    def _collate(self, batch):
        def merge(tensors, value=0.0, dtype=torch.float):
            max_len = max(tensor.shape[0] for tensor in tensors)
            new_tensors = []
            for tensor in tensors:
                pad = (2 * len(tensor.shape)) * [0]
                pad[-1] = max_len - tensor.shape[0]
                new_tensors.append(F.pad(tensor, pad=pad, value=value))
            return torch.stack(new_tensors).to(dtype=dtype)

        def make_pos(lengths):
            return merge([torch.arange(length) + 1 for length in lengths], value=0, dtype=torch.int64)

        batch = {key: [example[key] for example in batch] for key in batch[0]}

        audio = merge(batch['audio'])
        audio_len = torch.tensor(batch['audio_len'])
        text = merge(batch['text'], value=self._pad_id or 0, dtype=torch.long)
        text_pos = make_pos(batch.pop('text_len'))
        dur_true = merge(batch['dur_true'])

        assert text.shape == text_pos.shape
        assert text.shape == dur_true.shape

        return audio, audio_len, text, text_pos, dur_true

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def dataset(self) -> Optional[torch.utils.data.Dataset]:
        return None

    @property
    def data_iterator(self) -> Optional[torch.utils.data.DataLoader]:
        return self._dataloader


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

    @property
    @add_port_docs
    def input_ports(self):
        """Returns definitions of module input ports."""
        return dict(
            text=NeuralType(('B', 'T'), EmbeddedTextType()),
            text_pos=NeuralType(('B', 'T'), MaskType()),
            mel_true=NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            dur_true=NeuralType(('B', 'T'), LengthsType()),
        )

    @property
    @add_port_docs
    def output_ports(self):
        """Returns definitions of module output ports."""
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

        self.encoder = fastspeech_transformer.FastSpeechTransformerEncoder(
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
        self.length_regulator = fastspeech.LengthRegulator(
            encoder_output_size, duration_predictor_filter_size, duration_predictor_kernel_size, dropout
        ).to(self._device)

        self.decoder = fastspeech_transformer.FastSpeechTransformerDecoder(
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
        assert mel_true.shape[2] == dur_true.sum(-1).max()

        return mel_pred, dur_pred


class FastSpeechLoss(LossNM):
    """Neural Module Wrapper for Fast Speech Loss.

    Calculates final loss as sum of two: MSE for mel spectrograms and MSE for durations.

    """

    @property
    @add_port_docs
    def input_ports(self):
        """Returns definitions of module input ports."""
        return dict(
            mel_true=NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            mel_pred=NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            dur_true=NeuralType(('B', 'T'), LengthsType()),
            dur_pred=NeuralType(('B', 'T'), LengthsType()),
            text_pos=NeuralType(('B', 'T'), MaskType()),
        )

    @property
    @add_port_docs
    def output_ports(self):
        """Returns definitions of module output ports."""
        return dict(loss=NeuralType(None))

    def _loss_function(self, **kwargs):
        return self._loss(*(kwargs.values()))

    @staticmethod
    def _loss(
        mel_true, mel_pred, dur_true, dur_pred, text_pos,
    ):
        """Do the actual math in FastSpeech loss calculation.

        Args:
            mel_true: Ground truth mel spectrogram features (BTC, float).
            mel_pred: Predicted mel spectrogram features (BTC, float).
            dur_true: Ground truth durations (BQ, float).
            dur_pred: Predicted log-normalized durations (BQ, float).

        Returns:
            Single 0-dim loss tensor.

        """

        mel_loss = F.mse_loss(mel_pred, mel_true, reduction='none')
        mel_loss *= mel_true.ne(0).float()
        mel_loss = mel_loss.mean()

        dur_loss = F.mse_loss(dur_pred, (dur_true + 1).log(), reduction='none')
        dur_loss *= text_pos.ne(0).float()
        dur_loss = dur_loss.mean()

        loss = mel_loss + dur_loss

        return loss
