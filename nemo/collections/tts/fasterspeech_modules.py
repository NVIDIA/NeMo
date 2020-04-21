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

import argparse
import math
import os
import sys
import time
from typing import Any, Dict, Optional

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
    'FasterSpeechDataLayer',
    'FasterSpeech',
    'LenSampler',
    'FasterSpeechDurLoss',
    'FasterSpeechMelLoss',
    'WaveGlowInference',
]


class _Ops:
    @staticmethod
    def merge(tensors, value=0.0, dtype=torch.float):
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
        return _Ops.merge([torch.ones(length, device=device) for length in lengths], value=0, dtype=torch.bool)

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


class FasterSpeechDataset:
    def __init__(
        self, audio_dataset, durs_file, durs_type='full-pad', speakers=None, speaker_table=None, speaker_embs=None,
    ):
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
        id_, audio, audio_len, text, text_len, speaker = self._audio_dataset[index]
        example = dict(audio=audio, audio_len=audio_len, text=text, text_len=text_len)

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


class SuperSmartSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, *args, **kwargs):
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
        g.manual_seed(self.epoch)
        b_indices = torch.randperm(len(batches), generator=g).tolist()

        for b_i in b_indices:
            yield from batches[b_i]


class AllSampler(torch.utils.data.distributed.DistributedSampler):
    def __iter__(self):
        return iter(list(range(self.total_size)))


class BDAugs:
    """Different blanks/durs augs."""

    @staticmethod
    def shake(b, d, p=0.1):
        """Changes blanks/durs balance sightly."""

        b, d, total = b.copy(), d.copy(), sum(b) + sum(d)

        def split2(x):
            xl = np.random.binomial(x, p)
            return xl, x - xl

        def split3(x):
            xl, xm = split2(x)
            xr, xm = split2(xm)
            return xl, xm, xr

        n, m = len(b), len(d)
        nb = np.zeros_like(b)
        for i in range(len(b)):
            bl, bm, br = split3(b[i])

            nb[i] += bm

            if i > 0:
                d[i - 1] += bl
            else:
                nb[i] += bl

            if i < m:
                d[i] += br
            else:
                nb[i] += br

        b = nb
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


class FasterSpeechDataLayer(DataLayerNM):
    """Data Layer for Faster Speech model.

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
            speaker=NeuralType(('B',), EmbeddedTextType(), optional=True),
            speaker_emb=NeuralType(('B', 'T'), EncodedRepresentation(), optional=True),
        )

    def __init__(
        self,
        data,
        durs,
        labels,
        durs_type='full-pad',
        speakers=None,
        speaker_table=None,
        speaker_embs=None,
        batch_size=32,
        sample_rate=16000,
        int_values=False,
        bos_id=None,
        eos_id=None,
        pad_id=None,
        blank_id=None,
        min_duration=0.1,
        max_duration=None,
        normalize_transcripts=True,
        trim_silence=False,
        load_audio=True,
        drop_last=False,
        shuffle=True,
        num_workers=0,
        sampler_type='default',
    ):
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
            'add_id': True,
            'add_speaker': True,
        }
        audio_dataset = AudioDataset(**dataset_params)
        self._dataset = FasterSpeechDataset(audio_dataset, durs, durs_type, speakers, speaker_table, speaker_embs)
        self._durs_type = durs_type
        self._pad_id = pad_id
        self._blank_id = blank_id
        self._space_id = labels.index(' ')
        self._sample_rate = sample_rate
        self._load_audio = load_audio

        sampler = None
        if self._placement == nemo.core.DeviceType.AllGpu:
            if sampler_type == 'all':
                sampler = AllSampler(self._dataset)
            elif sampler_type == 'default':
                sampler = torch.utils.data.distributed.DistributedSampler(self._dataset)
            elif sampler_type == 'super-smart':
                sampler = SuperSmartSampler(
                    dataset=self._dataset,
                    lengths=[e.duration for e in audio_dataset.collection],
                    batch_size=batch_size,
                )
            else:
                raise ValueError("Invalid sample type.")

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
        batch = {key: [example[key] for example in batch] for key in batch[0]}

        if self._load_audio:
            audio = _Ops.merge(batch['audio'])
            audio_len = torch.tensor(batch['audio_len'], dtype=torch.long)
        else:
            audio, audio_len = None, None

        if self._durs_type == 'pad':
            text = [F.pad(text, pad=[1, 1], value=self._space_id) for text in batch['text']]
            text = _Ops.merge(text, value=self._pad_id, dtype=torch.long)
            # noinspection PyTypeChecker
            text_mask = _Ops.make_mask([text_len + 2 for text_len in batch['text_len']])
            dur = _Ops.merge(batch['dur'], dtype=torch.long)
        elif self._durs_type == 'full-pad':
            # `text`
            text = [
                _Ops.interleave(x=torch.empty(len(text) + 1, dtype=torch.long).fill_(self._blank_id), y=text)
                for text in batch['text']
            ]
            text = _Ops.merge(text, value=self._pad_id, dtype=torch.long)

            # `text_mask`
            # noinspection PyTypeChecker
            text_mask = _Ops.make_mask([text_len * 2 + 1 for text_len in batch['text_len']])

            # `dur`
            blank, dur = batch['blank'], batch['dur']

            # # Aug
            # new_blank, new_dur = [], []
            # for b, d in zip(blank, dur):
            #     new_b, new_d = BDAugs.shake(b.numpy(), d.numpy(), p=0.4)
            #     new_blank.append(torch.tensor(new_b))
            #     new_dur.append(torch.tensor(new_d))
            # blank, dur = new_blank, new_dur

            dur = _Ops.merge([_Ops.interleave(b, d) for b, d in zip(blank, dur)], dtype=torch.long)
        else:
            raise ValueError("Wrong durations handling type.")

        text_rep = _Ops.merge(
            tensors=[torch.repeat_interleave(text1, dur1) for text1, dur1 in zip(text, dur)], dtype=torch.long,
        )
        text_rep_mask = _Ops.make_mask(dur.sum(-1))

        speaker, speaker_emb = None, None
        if 'speaker' in batch:
            speaker = torch.tensor(batch['speaker'], dtype=torch.long)
        if 'speaker_emb' in batch:
            speaker_emb = _Ops.merge(batch['speaker_emb'], dtype=torch.float)

        assert audio is None or audio.shape[-1] == audio_len.max()
        assert text.shape == text_mask.shape, f'{text.shape} vs {text_mask.shape}'
        assert text.shape == dur.shape, f'{text.shape} vs {dur.shape}'

        return audio, audio_len, text, text_mask, dur, text_rep, text_rep_mask, speaker, speaker_emb

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def dataset(self) -> Optional[torch.utils.data.Dataset]:
        return None

    @property
    def data_iterator(self) -> Optional[torch.utils.data.DataLoader]:
        return self._dataloader


class FasterSpeech(nemo_nm.TrainableNM):
    """FasterSpeech TTS Model"""

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
    ):
        super().__init__()

        # Embedding for input text
        self.text_emb = nn.Embedding(n_vocab, d_char, padding_idx=pad_id).to(self._device)
        self.text_emb.weight.data.uniform_(-1, 1)

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
        self.out = nn.Conv1d(d_o, d_out, kernel_size=1, bias=True).to(self._device)

        self._pad16 = pad16

    def forward(self, text=None, text_mask=None, text_rep=None, text_rep_mask=None, speaker_emb=None):
        if text_rep is not None:
            text, text_mask = text_rep, text_rep_mask

        if self._pad16:
            text = _Ops.pad(text, value=self.text_emb.padding_idx)

        x = self.text_emb(text)  # BT => BTE
        # x = torch.cat([x, torch.flip(x, dims=[1])], dim=-1)  # BTE => BT[2E]
        x_len = text_mask.sum(-1)

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

        o = self.out(o).transpose(-1, -2)  # BTO

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


class FasterSpeechDurLoss(LossNM):
    """Neural Module Wrapper for Faster Speech Dur Loss."""

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
        self, method='l2-log', num_classes=32, dmld_hidden=5, reduction='all', max_dur=500, xe_steps_coef=1.5,
    ):
        super().__init__()

        self._method = method
        self._num_classes = num_classes
        self._dmld_hidden = dmld_hidden
        self._reduction = reduction

        # Creates XE Steps classes.
        classes = np.arange(num_classes).tolist()
        k, c = 1, num_classes - 1
        while c < max_dur:
            k *= xe_steps_coef
            c += k
            classes.append(int(c))
        self._xe_steps_classes = classes
        if self._method == 'xe-steps':
            nemo.logging.info('XE Steps Classes: %s', str(classes))

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

            loss = F.cross_entropy(input=dur_pred.transpose(1, 2), target=dur_true, reduction='none')
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
            return 3 * self._args.loss_dmld_hidden
        elif self._method == 'dmld':
            return 3 * self._args.loss_dmld_hidden
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
            dur_pred = (dur_pred + 1) / 2 * math.log(self._loss_dmld_num_classes)
            # [0, log(num_classes)] => [0, num_classes - 1]
            dur_pred = torch.clamp(dur_pred.exp() - 1, max=self._loss_dmld_num_classes - 1)
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


class FasterSpeechMelLoss(LossNM):
    """Neural Module Wrapper for Faster Speech Mel Loss."""

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

    def __init__(self, reduction='all', pad16=False):
        super().__init__()

        self._reduction = reduction
        self._pad16 = pad16

    def _loss_function(self, true, pred, mask, mel_len=None, dur_true=None):
        if mel_len is not None and dur_true is not None:
            if not torch.equal(mel_len, dur_true.sum(-1)) or not torch.equal(mel_len, mask.sum(-1)):
                raise ValueError("Wrong mel length calculation.")

        true = true.transpose(-1, -2)
        if self._pad16:
            true = _Ops.pad(true)

        loss = F.mse_loss(pred, true, reduction='none').mean(-1)

        if self._pad16:
            mask = _Ops.pad(mask)
        loss *= mask.float()

        if self._reduction == 'all':
            loss = loss.sum() / mask.sum()
        elif self._reduction == 'batch':
            loss = loss.sum(-1) / mask.sum(-1)
            loss = loss.mean()
        else:
            raise ValueError("Wrong reduction.")

        return loss


class WaveGlowInference:
    def __init__(self, code, checkpoint, sigma=1.0):
        # One nasty little hack
        sys.path.append(code)

        nemo.logging.info("Loading WaveGlow from %s", checkpoint)
        from convert_model import update_model

        model = update_model(torch.load(checkpoint)['model'])
        model = model.remove_weightnorm(model).cuda()
        model.eval()
        self._model = model
        self._sigma = sigma

        from denoiser import Denoiser

        denoiser = Denoiser(self._model).cuda()
        denoiser.eval()
        self._denoiser = denoiser

    def __call__(self, mel, denoiser=0.1):
        mel = torch.tensor(mel, device='cuda').unsqueeze(0)

        with torch.no_grad():
            audio = self._model.infer(mel, sigma=self._sigma)
            if denoiser > 0.0:
                audio = self._denoiser(audio, denoiser)

            audio /= audio.max()

        audio = audio.squeeze().cpu().numpy()

        return audio
