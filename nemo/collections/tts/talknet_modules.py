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

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

import nemo
from nemo.backends.pytorch import nm as nemo_nm
from nemo.backends.pytorch.nm import DataLayerNM, LossNM, NonTrainableNM
from nemo.collections import asr as nemo_asr
from nemo.collections import tts as nemo_tts
from nemo.collections.asr.parts import AudioDataset, WaveformFeaturizer
from nemo.core.neural_types import (
    AudioSignal,
    ChannelType,
    EmbeddedTextType,
    EncodedRepresentation,
    LengthsType,
    MaskType,
    MelSpectrogramType,
    NeuralType,
)
from nemo.utils.decorators import add_port_docs

__all__ = [
    'TalkNetDataLayer',
    'TalkNet',
    'LenSampler',
    'TalkNetDursLoss',
    'TalkNetMelsLoss',
]


class Ops:
    """Bunch of PyTorch useful function for tensors manipulation."""

    @staticmethod
    def merge(tensors, value=0, dtype=None):
        tensors = [tensor if isinstance(tensor, torch.Tensor) else torch.tensor(tensor) for tensor in tensors]
        dtype = tensors[0].dtype if dtype is None else dtype
        max_len = max(tensor.shape[0] for tensor in tensors)
        new_tensors = []
        for tensor in tensors:
            pad = (2 * len(tensor.shape)) * [0]
            pad[-1] = max_len - tensor.shape[0]
            new_tensors.append(F.pad(tensor, pad=pad, value=value))
        return torch.stack(new_tensors).to(dtype=dtype)

    @staticmethod
    def make_mask(lengths):
        device = lengths.device if torch.is_tensor(lengths) else 'cpu'
        return Ops.merge([torch.ones(length, device=device) for length in lengths], value=0, dtype=torch.bool)

    @staticmethod
    def interleave(x, y):
        xy = torch.stack([x[:-1], y], dim=1).view(-1)
        xy = F.pad(xy, pad=[0, 1], value=x[-1])
        return xy

    @staticmethod
    def pad(x, to=16, dim=1, value=0):
        pads = [0, 0] * len(x.shape)
        pads[-(2 * dim + 1)] = (to - (x.shape[dim] % to)) % to
        return F.pad(x, pads, value=value)


class TalkNetDataset:
    def __init__(
        self, audio_dataset, durs_file, durs_type='full-pad', speakers=None, speaker_table=None, speaker_embs=None,
    ):
        """TalkNet dataset with indexing.

        Args:
            audio_dataset: Instance of AudioDataset with basic audio/text info.
            durs_file: Durations list file.
            durs_type: Type of durations to use.
            speakers: Speakers list file.
            speaker_table: Table of speakers ids.
            speaker_embs: Matrix of speakers embeddings.
        """

        self._audio_dataset = audio_dataset
        self._durs = np.load(durs_file, allow_pickle=True)
        self._durs_type = durs_type

        self._speakers, self._speakers_table, self._speaker_embs = None, None, None

        if speakers is not None:
            self._speakers = np.load(speakers, allow_pickle=True)

        if speaker_table is not None:
            self._speakers_table = {sid: i for i, sid in enumerate(pd.read_csv(speaker_table, sep='\t').index)}

        if speaker_embs is not None:
            self._speaker_embs = np.load(speaker_embs, allow_pickle=True)

    def __getitem__(self, index):
        (audio, audio_len, text, text_len), misc = self._audio_dataset[index]
        id_, text_raw, speaker = misc['id'], misc['text_raw'], misc['speaker']
        example = dict(audio=audio, audio_len=audio_len, text=text, text_len=text_len, text_raw=text_raw)

        if self._durs_type == 'pad':
            dur = self._durs[id_]
            example['dur'] = torch.tensor(dur, dtype=torch.long)
        elif self._durs_type == 'full-pad':
            blank, dur = self._durs[id_]
            example['blank'] = torch.tensor(blank, dtype=torch.long)
            example['dur'] = torch.tensor(dur, dtype=torch.long)
        else:
            raise ValueError("Wrong durations handling type.")

        if self._speakers_table is not None:
            example['speaker'] = self._speakers_table[speaker]

        if self._speakers is not None:
            example['speaker_emb'] = torch.tensor(self._speakers[id_])

        if self._speaker_embs is not None:
            example['speaker_emb'] = torch.tensor(self._speaker_embs[example['speaker']])

        return example

    def __len__(self):
        return len(self._audio_dataset)


class LengthsAwareSampler(torch.utils.data.distributed.DistributedSampler):  # noqa
    def __init__(self, *args, **kwargs):
        """Assignees samples with similar lengths to reduce padding.

        Args:
            *args: Args to propagate to `DistributedSampler` constructor.
            **kwargs: Kwargs to propagate to `DistributedSampler` constructor. Additional
                keys should be 'lengths' and 'batch_size'.
        """

        self.lengths = kwargs.pop('lengths')
        self.batch_size = kwargs.pop('batch_size')

        super().__init__(*args, **kwargs)

    @staticmethod
    def _local_shuffle(values, keys, window, seed):
        g = np.random.Generator(np.random.PCG64(seed))

        keys = np.array(keys)
        kam_middle = keys[2 * window :] - keys[: -(2 * window)]
        kam_prefix = keys[window : 2 * window] - keys[:window]
        kam_suffix = keys[-window:] - keys[-2 * window : -window]
        kam = np.hstack([kam_prefix, kam_middle, kam_suffix])
        new_keys = keys + g.uniform(-kam, kam, size=keys.shape)

        kvs = list(zip(new_keys, values))
        kvs.sort()

        return [v for _, v in kvs]

    def __iter__(self):
        indices = list(super().__iter__())

        indices.sort(key=lambda i: self.lengths[i])

        # Local shuffle (to introduce a little bit of randomness)
        lens = [self.lengths[i] for i in indices]
        # For lengths over workers to be notably different, coef should be around 1000. But smaller values works for
        # shuffling as well.
        new_indices = self._local_shuffle(indices, lens, 10 * self.batch_size, hash((self.rank, self.epoch)))
        assert len(indices) == len(new_indices)
        indices = new_indices

        batches = []
        for i in range(0, len(indices), self.batch_size):
            batches.append(indices[i : i + self.batch_size])

        g = torch.Generator()
        g.manual_seed(self.epoch)  # noqa
        b_indices = torch.randperm(len(batches), generator=g).tolist()

        for b_i in b_indices:
            yield from batches[b_i]


class AllSampler(torch.utils.data.distributed.DistributedSampler):  # noqa
    def __iter__(self):
        return iter(list(range(len(self.dataset))))


class BlanksDurationAugmentation:
    """Different blanks/durs augs."""

    @staticmethod
    def _split(d):
        return np.array(d[::2]), np.array(d[1::2])

    @staticmethod
    def _merge(b, d):
        result = []
        for b1, d1 in zip(b, d):
            result.extend([b1, d1])

        result.append(b[-1])
        return np.array(result)

    @classmethod
    def shake_biased(cls, b, d, p=0.1):
        """Biased binomial shake."""

        b, d, total = b.copy(), d.copy(), sum(b) + sum(d)

        def split2(x):
            xl = np.random.binomial(x, p, size=x.shape)
            return xl, x - xl

        def split3(x):
            xl, xm = split2(x)
            xr, xm = split2(xm)
            return xl, xm, xr

        bdl, bdm, bdr = split3(cls._merge(b, d))
        bd = bdm + np.roll(bdl, -1) + np.roll(bdr, +1)
        b, d = cls._split(bd)

        assert sum(b) + sum(d) == total

        return b, d

    @classmethod
    def shake_unbiased(cls, b, d, p=0.1):
        b, d, total = b.copy(), d.copy(), sum(b) + sum(d)

        def split2(x, mm):
            xl = np.random.binomial(np.minimum(x, mm), p, size=x.shape)
            return xl, x - xl

        def split3(x):
            xl, xm = split2(x, np.roll(x, +1))
            xr, xm = split2(xm, np.roll(x, -1))
            return xl, xm, xr

        bdl, bdm, bdr = split3(cls._merge(b, d))
        bd = bdm + np.roll(bdl, -1) + np.roll(bdr, +1)
        b, d = cls._split(bd)

        assert sum(b) + sum(d) == total

        return b, d

    @staticmethod
    def zero_out(b, d, p=0.1):
        """Zeros out some of the durs."""

        b, d, total = b.copy(), d.copy(), sum(b) + sum(d)

        mask = np.random.binomial(1, size=d.size, p=p).astype(bool)
        b[:-1][mask] += d[mask]
        d[mask] -= d[mask]

        assert sum(b) + sum(d) == total

        return b, d

    @staticmethod
    def compose(augs):
        """Composes augs."""

        def pipe(b, d, p=0.1):
            for aug in augs:
                b, d = aug(b, d, p=p)

            return b, d

        return pipe


class TalkNetDataLayer(DataLayerNM):
    """Data Layer for TalkNet model.

    Basically, replicated behavior from AudioToText Data Layer, zipped with ground truth durations for additional loss.

    """

    @property
    @add_port_docs
    def output_ports(self):
        """Returns definitions of module output ports."""
        return dict(
            audio=NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            audio_len=NeuralType(tuple('B'), LengthsType()),
            text=NeuralType(('B', 'T'), EmbeddedTextType()),
            text_mask=NeuralType(('B', 'T'), MaskType()),
            dur=NeuralType(('B', 'T'), LengthsType()),
            text_rep=NeuralType(('B', 'T'), LengthsType()),
            text_rep_mask=NeuralType(('B', 'T'), MaskType()),
            text_raw=NeuralType(),
            speaker=NeuralType(('B',), EmbeddedTextType(), optional=True),
            speaker_emb=NeuralType(('B', 'T'), EncodedRepresentation(), optional=True),
        )

    def __init__(
        self,
        data: str,
        durs: str,
        labels: List[str],
        durs_type: str = 'full-pad',
        speakers: str = None,
        speaker_table: str = None,
        speaker_embs: str = None,
        batch_size: int = 32,
        sample_rate: int = 16000,
        int_values: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: Optional[int] = None,
        blank_id: Optional[int] = None,
        min_duration: Optional[float] = 0.1,
        max_duration: Optional[float] = None,
        normalize_transcripts: bool = True,
        trim_silence: bool = False,
        load_audio: bool = True,
        drop_last: bool = False,
        shuffle: bool = True,
        num_workers: int = 0,
        sampler_type: str = 'default',
        bd_aug: bool = False,
    ):
        """Creates TalkNet data iterator.

        Args:
            data: Path to dataset manifest file.
            durs: Path to pickled durations file.
            labels: List strings of labels to use.
            durs_type: String id of durations type to use.
            speakers: Speakers list file.
            speaker_table: Speakers ids mapping.
            speaker_embs: Speakers embeddings file.
            batch_size: Number of sample in batch.
            sample_rate: Target sampling rate for data. Audio files will be resampled to sample_rate if it is not
                already.
            int_values: Bool indicating whether the audio file is saved as int data or float data.
            bos_id: Beginning of string symbol id used for seq2seq models.
            eos_id: End of string symbol id used for seq2seq models.
            pad_id: Token used to pad when collating samples in batches.
            blank_id: Int id of blank symbol.
            min_duration: All training files which have a duration less than min_duration are dropped.
            max_duration: All training files which have a duration more than max_duration are dropped.
            normalize_transcripts: Whether to use automatic text cleaning.
            trim_silence: Whether to use trim silence from beginning and end of audio signal using
                librosa.effects.trim().
            load_audio: Controls whether the dataloader loads the audio signal and transcript or just the transcript.
            drop_last: See PyTorch DataLoader.
            shuffle: See PyTorch DataLoader.
            num_workers: See PyTorch DataLoader.
            sampler_type: String id of sampler type to use.
            bd_aug: True if use augmentation for blanks/durs.
        """

        super().__init__()

        # Set up dataset.
        self._featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=None)
        dataset_params = {
            'manifest_filepath': data,
            'labels': labels,
            'featurizer': self._featurizer,
            'max_duration': max_duration,
            'min_duration': min_duration,
            'normalize': normalize_transcripts,
            'trim': trim_silence,
            'bos_id': bos_id,
            'eos_id': eos_id,
            'load_audio': load_audio,
            'add_misc': True,
        }
        audio_dataset = AudioDataset(**dataset_params)
        self._dataset = TalkNetDataset(audio_dataset, durs, durs_type, speakers, speaker_table, speaker_embs)
        self._durs_type = durs_type
        self._pad_id = pad_id
        self._blank_id = blank_id
        self._space_id = labels.index(' ')
        self._sample_rate = sample_rate
        self._load_audio = load_audio
        self._bd_aug = bd_aug

        sampler = None
        if self._placement == nemo.core.DeviceType.AllGpu:
            if sampler_type == 'all':
                sampler = AllSampler(self._dataset)
            elif sampler_type == 'default':
                sampler = torch.utils.data.distributed.DistributedSampler(self._dataset)  # noqa
            elif sampler_type == 'super-smart':
                sampler = LengthsAwareSampler(
                    dataset=self._dataset,
                    lengths=[e.duration for e in audio_dataset.collection],
                    batch_size=batch_size,
                )
            else:
                raise ValueError("Invalid sample type.")

        self._dataloader = torch.utils.data.DataLoader(  # noqa
            dataset=self._dataset,
            batch_size=batch_size,
            collate_fn=self._collate,
            drop_last=drop_last,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
        )

    def _collate(self, batch):
        batch = {key: [example[key] for example in batch] for key in batch[0]}

        if self._load_audio:
            audio = Ops.merge(batch['audio'])
            audio_len = torch.tensor(batch['audio_len'], dtype=torch.long)
        else:
            audio, audio_len = None, None

        if self._durs_type == 'pad':
            text = [F.pad(text, pad=[1, 1], value=self._space_id) for text in batch['text']]
            text = Ops.merge(text, value=self._pad_id, dtype=torch.long)
            # noinspection PyTypeChecker
            text_mask = Ops.make_mask([text_len + 2 for text_len in batch['text_len']])
            dur = Ops.merge(batch['dur'], dtype=torch.long)
        elif self._durs_type == 'full-pad':
            # `text`
            text = [
                Ops.interleave(x=torch.empty(len(text) + 1, dtype=torch.long).fill_(self._blank_id), y=text)
                for text in batch['text']
            ]
            text = Ops.merge(text, value=self._pad_id, dtype=torch.long)

            # `text_mask`
            text_mask = Ops.make_mask([text_len * 2 + 1 for text_len in batch['text_len']])  # noqa

            # `dur`
            blank, dur = batch['blank'], batch['dur']

            # Aug
            if self._bd_aug:
                if isinstance(self._bd_aug, str):
                    name, p = self._bd_aug.split('|')
                else:
                    name, p = 'shake_biased', '0.1'
                aug = lambda b, d: getattr(BlanksDurationAugmentation, name)(b, d, p)  # noqa

                new_blank, new_dur = [], []
                for b, d in zip(blank, dur):
                    new_b, new_d = aug(b.numpy(), d.numpy())
                    new_blank.append(torch.tensor(new_b))
                    new_dur.append(torch.tensor(new_d))
                blank, dur = new_blank, new_dur

            dur = Ops.merge([Ops.interleave(b, d) for b, d in zip(blank, dur)], dtype=torch.long)
        else:
            raise ValueError("Wrong durations handling type.")

        text_rep = Ops.merge(
            tensors=[torch.repeat_interleave(text1, dur1) for text1, dur1 in zip(text, dur)], dtype=torch.long,
        )
        text_rep_mask = Ops.make_mask(dur.sum(-1))

        text_raw = batch['text_raw']

        speaker, speaker_emb = None, None
        if 'speaker' in batch:
            speaker = torch.tensor(batch['speaker'], dtype=torch.long)
        if 'speaker_emb' in batch:
            speaker_emb = Ops.merge(batch['speaker_emb'], dtype=torch.float)

        assert audio is None or audio.shape[-1] == audio_len.max()
        assert text.shape == text_mask.shape, f'{text.shape} vs {text_mask.shape}'
        assert text.shape == dur.shape, f'{text.shape} vs {dur.shape}'

        return audio, audio_len, text, text_mask, dur, text_rep, text_rep_mask, text_raw, speaker, speaker_emb

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def dataset(self) -> Optional[torch.utils.data.Dataset]:  # noqa
        return None

    @property
    def data_iterator(self) -> Optional[torch.utils.data.DataLoader]:  # noqa
        return self._dataloader


class PolySpanEmb(nn.Module):
    def __init__(self, emb):
        """Wraps chars embeddings with assigning polynomial span embs for blanks.

        Args:
            emb: Character embeddings.
        """

        super().__init__()

        self._emb = emb

    def forward(self, text, dur):
        lefts, rights = self._generate_sides(text)
        lefts = self._emb(self._generate_text_rep(lefts, dur))
        rights = self._emb(self._generate_text_rep(rights, dur))

        left_c = self._generate_left_c(dur).unsqueeze_(-1)  # noqa

        x = left_c * lefts + (1 - left_c) * rights  # noqa

        return x

    def _generate_sides(self, text):
        lefts = F.pad(text[:, :-1], [1, 0, 0, 0], value=self._emb.padding_idx)
        lefts[:, 1::2] = text[:, 1::2]
        rights = F.pad(text[:, 1:], [0, 1, 0, 0], value=self._emb.padding_idx)
        rights[:, 1::2] = text[:, 1::2]

        return lefts, rights

    @staticmethod
    def _generate_text_rep(text, dur):
        text_rep = []
        for t, d in zip(text, dur):
            text_rep.append(torch.repeat_interleave(t, d))

        text_rep = Ops.merge(text_rep)

        return text_rep

    def _generate_left_c(self, dur):
        x = F.pad(torch.cumsum(dur, dim=-1)[:, :-1], [1, 0], value=0)
        pos_cm = self._generate_text_rep(x, dur)
        mask = self._generate_text_rep(torch.ones_like(dur), dur)
        ones_cm = torch.cumsum(mask, dim=1)
        totals = self._generate_text_rep(dur, dur) + 1

        left_c = (ones_cm - pos_cm) * mask
        left_c = left_c.to(dtype=self._emb.weight.dtype)
        left_c = 1.0 - left_c / totals  # noqa

        return left_c


class TalkNet(nemo_nm.TrainableNM):
    """TalkNet TTS Model"""

    @property
    @add_port_docs
    def input_ports(self):
        """Returns definitions of module input ports."""
        return dict(
            text=NeuralType(('B', 'T'), EmbeddedTextType(), optional=True),
            text_mask=NeuralType(('B', 'T'), MaskType(), optional=True),
            text_rep=NeuralType(('B', 'T'), LengthsType(), optional=True),
            text_rep_mask=NeuralType(('B', 'T'), MaskType(), optional=True),
            speaker_emb=NeuralType(('B', 'T'), EncodedRepresentation(), optional=True),
            durs=NeuralType(('B', 'T'), LengthsType(), optional=True),
        )

    @property
    @add_port_docs
    def output_ports(self):
        """Returns definitions of module output ports."""
        return dict(pred=NeuralType(('B', 'T', 'D'), ChannelType()), len=NeuralType(('B',), LengthsType()))

    def __init__(
        self,
        n_vocab: int,
        d_char: int,
        pad_id: int,
        jasper_kwargs: Dict[str, Any],
        d_out: int,
        d_speaker_emb: Optional[int] = None,
        d_speaker_x: Optional[int] = None,
        d_speaker_o: Optional[int] = None,
        pad16: bool = False,
        poly_span: bool = False,
        doubling: bool = False,
    ):
        """Creates TalkNet backbone instance.

        Args:
            n_vocab: Size of input vocabulary.
            d_char: Dimension of char embedding.
            pad_id: Id of padding symbol.
            jasper_kwargs: Kwargs to instantiate QN encoder.
            d_out: Dimension of output.
            d_speaker_emb: Dimension of speaker embedding.
            d_speaker_x: Dimension of pre speaker embedding.
            d_speaker_o: Dimension of post speaker embedding.
            pad16: True if pad tensors to 16.
            poly_span: True if assign polynomial span embeddings for blanks.
            doubling: True if using mel channels doubling trick.
        """

        super().__init__()

        # Embedding for input text
        self.text_emb = nn.Embedding(n_vocab, d_char, padding_idx=pad_id).to(self._device)
        self.text_emb.weight.data.uniform_(-1, 1)

        # PolySpan
        self.ps = PolySpanEmb(self.text_emb)
        self._poly_span = poly_span

        # Embedding for speaker
        if d_speaker_emb is not None:
            self.speaker_in = nn.Linear(d_speaker_emb, d_speaker_x).to(self._device)
            self.speaker_out = nn.Linear(d_speaker_emb, d_speaker_o).to(self._device)
        else:
            self.speaker_in, self.speaker_out = None, None

        jasper_params = jasper_kwargs['jasper']
        d_enc_out = jasper_params[-1]["filters"]
        d_x = d_char + (int(d_speaker_x or 0) if d_speaker_emb else 0)
        self.jasper = nemo_asr.JasperEncoder(feat_in=d_x, **jasper_kwargs).to(self._device)

        d_o = d_enc_out + (int(d_speaker_o or 0) if d_speaker_emb else 0)
        self.out = nn.Conv1d(d_o, d_out * (1 + int(doubling)), kernel_size=1, bias=True).to(self._device)

        self._pad16 = pad16
        self._doubling = doubling

    def forward(self, text=None, text_mask=None, text_rep=None, text_rep_mask=None, speaker_emb=None, durs=None):
        if self._poly_span:
            # Do not support pad16 yet. Do we really need it though?
            x, x_len = self.ps(text, durs), durs.sum(-1)
        else:
            if text_rep is not None:
                text, text_mask = text_rep, text_rep_mask

            if self._pad16:
                text = Ops.pad(text, value=self.text_emb.padding_idx)

            x, x_len = self.text_emb(text), text_mask.sum(-1)

        if speaker_emb is not None:
            speaker_x = speaker_emb
            if self.speaker_in is not None:
                speaker_x = self.speaker_in(speaker_x)

            speaker_x = speaker_x.unsqueeze(1).repeat([1, x.shape[1], 1])  # BS => BTS
            x = torch.cat([x, speaker_x], dim=-1)  # stack([BTE, BTS]) = BT(E + S)

        o, o_len = self.jasper(x.transpose(-1, -2), x_len, force_pt=True)
        assert x.shape[1] == o.shape[-1]  # Time
        assert torch.equal(x_len, o_len)

        if speaker_emb is not None:
            speaker_o = speaker_emb
            if self.speaker_out is not None:
                speaker_o = self.speaker_out(speaker_o)

            speaker_o = speaker_o.unsqueeze(-1).repeat([1, 1, o.shape[-1]])  # BS => BST
            o = torch.cat([o, speaker_o], dim=1)  # stack([BOT, BST]) = B(O + S)T

        o = o.to(dtype=self.out.weight.dtype)
        o = self.out(o).transpose(-1, -2)  # BTO

        if self._doubling:
            o = o.reshape(o.shape[0], -1, o.shape[-1] // 2)  # Time doubles.

        return o, o_len


class LenSampler(NonTrainableNM):
    """Subsample data based on length to fit in bigger batches."""

    @property
    @add_port_docs
    def input_ports(self):
        """Returns definitions of module input ports."""
        return dict(
            text_rep=NeuralType(('B', 'T'), LengthsType()),
            text_rep_mask=NeuralType(('B', 'T'), MaskType()),
            mel_true=NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            mel_len=NeuralType(('B',), LengthsType()),
        )

    @property
    @add_port_docs
    def output_ports(self):
        """Returns definitions of module input ports."""
        return dict(
            text_rep=NeuralType(('B', 'T'), LengthsType()),
            text_rep_mask=NeuralType(('B', 'T'), MaskType()),
            mel_true=NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            mel_len=NeuralType(('B',), LengthsType()),
        )

    def __init__(self, max_len=1200):
        super().__init__()

        self._max_len = max_len

    def forward(self, text_rep, text_rep_mask, mel_true, mel_len):
        # Optimization
        if mel_len.max().item() <= self._max_len:
            return text_rep, text_rep_mask, mel_true, mel_len

        inds = np.random.randint(np.maximum(mel_len.cpu().numpy() - self._max_len, 0) + 1)

        text_rep = torch.stack([b1[i : i + self._max_len] for i, b1 in zip(inds, text_rep)])
        text_rep_mask = torch.stack([b1[i : i + self._max_len] for i, b1 in zip(inds, text_rep_mask)])
        mel_true = torch.stack([b1[:, i : i + self._max_len] for i, b1 in zip(inds, mel_true)])
        mel_len = torch.clamp(mel_len, max=self._max_len)

        return text_rep, text_rep_mask, mel_true, mel_len


class TalkNetDursLoss(LossNM):
    """Neural Module Wrapper for TalkNet dur loss."""

    @property
    @add_port_docs
    def input_ports(self):
        """Returns definitions of module input ports."""
        return dict(
            dur_true=NeuralType(('B', 'T'), LengthsType()),
            dur_pred=NeuralType(('B', 'T', 'D'), ChannelType()),
            text_mask=NeuralType(('B', 'T'), MaskType()),
        )

    @property
    @add_port_docs
    def output_ports(self):
        """Returns definitions of module output ports."""
        return dict(loss=NeuralType(None))

    def __init__(
        self,
        method='l2-log',
        num_classes=32,
        dmld_hidden=5,
        reduction='all',
        max_dur=500,
        xe_steps_coef=1.5,
        pad16=False,
    ):
        """Creates duration loss instance.

        Args:
            method: Method for duration loss calculation.
            num_classes: Number of classes to predict for classification methods.
            dmld_hidden: Dimension of hidden vector for DMLD method.
            reduction: Final loss tensor reduction type.
            max_dur: Maximum duration value to cover.
            xe_steps_coef: Ratio of adjusted steps for 'xe-steps' method.
            pad16: True if pad tensors to 16.
        """

        super().__init__()

        self._method = method
        self._num_classes = num_classes
        self._dmld_hidden = dmld_hidden
        self._reduction = reduction
        self._pad16 = pad16

        # Creates XE Steps classes.
        classes = np.arange(num_classes).tolist()
        k, c = 1, num_classes - 1
        while c < max_dur:
            k *= xe_steps_coef
            c += k
            classes.append(int(c))  # noqa
        self._xe_steps_classes = classes
        if self._method == 'xe-steps':
            nemo.logging.info('XE Steps Classes: %s', str(classes))

        # w = torch.arange(num_classes, dtype=torch.float, device=self._device)
        # w = (w + 1).log() + 1
        # w /= w.sum()
        self._weights = None

    def _loss_function(self, dur_true, dur_pred, text_mask):
        if self._method.startswith('l2'):
            if dur_pred.shape[-1] != 1:
                raise ValueError("Wrong `dur_pred` shape.")
            dur_pred = dur_pred.squeeze(-1)

        if self._method == 'l2-log':
            loss = F.mse_loss(dur_pred, (dur_true + 1).float().log(), reduction='none')
        elif self._method == 'l2':
            loss = F.mse_loss(dur_pred, dur_true.float(), reduction='none')
        elif self._method == 'dmld-log':
            # [0, inf] => [0, num_classes - 1]
            dur_true = torch.clamp(dur_true, max=self._num_classes - 1)
            # [0, num_classes - 1] => [0, log(num_classes)]
            dur_true = (dur_true + 1).float().log()
            # [0, log(num_classes)] => [-1, 1]
            dur_true = (torch.clamp(dur_true / math.log(self._num_classes), max=1.0) - 0.5) * 2

            loss = nemo_tts.parts.dmld_loss(dur_pred, dur_true, self._num_classes)
        elif self._method == 'dmld':
            # [0, inf] => [0, num_classes - 1]
            dur_true = torch.clamp(dur_true, max=self._num_classes - 1)
            # [0, num_classes - 1] => [-1, 1]
            dur_true = (dur_true / (self._num_classes - 1) - 0.5) * 2

            loss = nemo_tts.parts.dmld_loss(dur_pred, dur_true, self._num_classes)
        elif self._method == 'xe':
            # [0, inf] => [0, num_classes - 1]
            dur_true = torch.clamp(dur_true, max=self._num_classes - 1)

            loss = F.cross_entropy(
                input=dur_pred.transpose(1, 2), target=dur_true, reduction='none', weight=self._weights,
            )
        elif self._method == 'xe-steps':
            # [0, inf] => [0, xe-steps-num-classes - 1]
            classes = torch.tensor(self._xe_steps_classes, device=dur_pred.device)
            a = dur_true.unsqueeze(-1).repeat(1, 1, *classes.shape)
            b = classes.unsqueeze(0).unsqueeze(0).repeat(*dur_true.shape, 1)
            dur_true = (a - b).abs().argmin(-1)

            loss = F.cross_entropy(input=dur_pred.transpose(1, 2), target=dur_true, reduction='none')
        else:
            raise ValueError("Wrong Method")

        loss *= text_mask.float()
        if self._reduction == 'all':
            loss = loss.sum() / text_mask.sum()
        elif self._reduction == 'batch':
            loss = loss.sum(-1) / text_mask.sum(-1)
            loss = loss.mean()
        else:
            raise ValueError("Wrong Reduction")

        return loss

    @property
    def d_out(self):
        if self._method == 'l2-log':
            return 1
        elif self._method == 'l2':
            return 1
        elif self._method == 'dmld-log':
            return 3 * self._dmld_hidden
        elif self._method == 'dmld':
            return 3 * self._dmld_hidden
        elif self._method == 'xe':
            return self._num_classes
        elif self._method == 'xe-steps':
            # noinspection PyTypeChecker
            return len(self._xe_steps_classes)
        else:
            raise ValueError("Wrong Method")

    def preprocessing(self, tensors):
        if self._method == 'l2-log':
            dur_pred = tensors.dur_pred.squeeze(-1).exp() - 1
        elif self._method == 'l2':
            dur_pred = tensors.dur_pred.squeeze(-1)
        elif self._method == 'dmld-log':
            dur_pred = nemo_tts.parts.dmld_sample(tensors.dur_pred)

            # [-1, 1] => [0, log(num_classes)]
            dur_pred = (dur_pred + 1) / 2 * math.log(self._num_classes)
            # [0, log(num_classes)] => [0, num_classes - 1]
            dur_pred = torch.clamp(dur_pred.exp() - 1, max=self._num_classes - 1)
        elif self._method == 'dmld':
            dur_pred = nemo_tts.parts.dmld_sample(tensors.dur_pred)

            # [-1, 1] => [0, num_classes - 1]
            dur_pred = (dur_pred + 1) / 2 * (self._num_classes - 1)
        elif self._method == 'xe':
            dur_pred = tensors.dur_pred.argmax(-1)
        elif self._method == 'xe-steps':
            dur_pred = tensors.dur_pred.argmax(-1)
            classes = torch.tensor(self._xe_steps_classes, device=dur_pred.device)
            b = classes.unsqueeze(0).unsqueeze(0).repeat(*dur_pred.shape, 1)
            dur_pred = b.gather(-1, dur_pred.unsqueeze(-1)).squeeze(-1)
        else:
            raise ValueError("Wrong Method")

        dur_pred[dur_pred < 0.0] = 0.0
        dur_pred = dur_pred.float().round().long()
        tensors.dur_pred = dur_pred

        return tensors


class TalkNetMelsLoss(LossNM):
    """Neural Module Wrapper for TalkNet mel loss."""

    @property
    @add_port_docs
    def input_ports(self):
        """Returns definitions of module input ports."""
        return dict(
            true=NeuralType(('B', 'D', 'T'), ChannelType()),  # 'BDT' - to fit mels directly.
            pred=NeuralType(('B', 'T', 'D'), ChannelType()),
            mask=NeuralType(('B', 'T'), MaskType()),
            mel_len=NeuralType(('B',), LengthsType(), optional=True),
            dur_true=NeuralType(('B', 'T'), LengthsType(), optional=True),
        )

    @property
    @add_port_docs
    def output_ports(self):
        """Returns definitions of module output ports."""
        return dict(loss=NeuralType(None))

    def __init__(self, reduction='all', pad16=False, doubling=False):
        """Creates instance of TalkNet mels loss calculation.

        Args:
            reduction: Final loss tensor reduction type.
            pad16: True if pad tensors to 16.
            doubling: True if using mel channels doubling trick.
        """

        super().__init__()

        self._reduction = reduction
        self._pad16 = pad16
        self._doubling = doubling

    def _loss_function(self, true, pred, mask, mel_len=None, dur_true=None):
        if mel_len is not None and dur_true is not None:
            if not torch.equal(mel_len, dur_true.sum(-1)) or not torch.equal(mel_len, mask.sum(-1)):
                raise ValueError("Wrong mel length calculation.")

        true = true.transpose(-1, -2)
        if self._pad16:
            true = Ops.pad(true)
        if self._doubling:
            true = Ops.pad(true, to=2)

        loss = F.mse_loss(pred, true, reduction='none').mean(-1)

        if self._pad16:
            mask = Ops.pad(mask)
        if self._doubling:
            mask = Ops.make_mask(mel_len)
            mask = Ops.pad(mask, to=2)
        loss *= mask.float()

        if self._reduction == 'all':
            loss = loss.sum() / mask.sum()
        elif self._reduction == 'batch':
            loss = loss.sum(-1) / mask.sum(-1)
            loss = loss.mean()
        else:
            raise ValueError("Wrong reduction.")

        return loss
