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
import io
import os
from typing import Callable, Dict, List, Optional, Union

import braceexpand
import torch
import webdataset as wd
from torch.nn import functional as F

from nemo.collections.asr.data import vocabs
from nemo.collections.asr.parts import collections, parsers
from nemo.collections.asr.parts.features import WaveformFeaturizer
from nemo.core.classes import Dataset, IterableDataset
from nemo.core.neural_types import *
from nemo.utils import logging
from nemo.utils.decorators import experimental

__all__ = [
    'AudioToCharDataset',
    'AudioToCharWithDursDataset',
    'AudioToBPEDataset',
    'AudioLabelDataset',
    'TarredAudioToCharDataset',
    'TarredAudioToBPEDataset',
]


def _speech_collate_fn(batch, pad_id):
    """collate batch of audio sig, audio len, tokens, tokens len
    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
               LongTensor):  A tuple of tuples of signal, signal lengths,
               encoded tokens, and encoded tokens length.  This collate func
               assumes the signals are 1d torch tensors (i.e. mono audio).
    """
    _, audio_lengths, _, tokens_lengths = zip(*batch)
    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()
    max_tokens_len = max(tokens_lengths).item()

    audio_signal, tokens = [], []
    for sig, sig_len, tokens_i, tokens_i_len in batch:
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)
        tokens_i_len = tokens_i_len.item()
        if tokens_i_len < max_tokens_len:
            pad = (0, max_tokens_len - tokens_i_len)
            tokens_i = torch.nn.functional.pad(tokens_i, pad, value=pad_id)
        tokens.append(tokens_i)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None
    tokens = torch.stack(tokens)
    tokens_lengths = torch.stack(tokens_lengths)

    return audio_signal, audio_lengths, tokens, tokens_lengths


class _AudioTextDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio files, transcripts, and durations (in seconds).
    Each new line is a different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath": "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}
    Args:
        manifest_filepath: Path to manifest json as described above. Can be comma-separated paths.
        labels: String containing all the possible characters to map to
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor object used to augment loaded
            audio
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include in dataset
        max_utts: Limit number of utterances
        blank_index: blank character index, default = -1
        unk_index: unk_character index, default = -1
        normalize: whether to normalize transcript text (default): True
        bos_id: Id of beginning of sequence symbol to append if not None
        eos_id: Id of end of sequence symbol to append if not None
        load_audio: Boolean flag indicate whether do or not load audio
        add_misc: True if add additional info dict.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'audio_signal': NeuralType(
                ('B', 'T'),
                AudioSignal(freq=self._sample_rate)  # TODO: self._sample_rate is not defined anywhere
                if self is not None and hasattr(self, '_sample_rate')
                else AudioSignal(),
            ),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self,
        manifest_filepath: str,
        parser: Union[str, Callable],
        sample_rate: int,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_utts: int = 0,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        load_audio: bool = True,
        add_misc: bool = False,
    ):
        self.parser = parser

        self.collection = collections.ASRAudioText(
            manifests_files=manifest_filepath.split(','),
            parser=parser,
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
        )

        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)
        self.trim = trim
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.pad_id = pad_id
        self.load_audio = load_audio
        self._add_misc = add_misc

    def __getitem__(self, index):
        sample = self.collection[index]
        if self.load_audio:
            offset = sample.offset

            if offset is None:
                offset = 0

            features = self.featurizer.process(
                sample.audio_file, offset=offset, duration=sample.duration, trim=self.trim, orig_sr=sample.orig_sr,
            )
            f, fl = features, torch.tensor(features.shape[0]).long()
        else:
            f, fl = None, None

        t, tl = sample.text_tokens, len(sample.text_tokens)
        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1

        output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

        if self._add_misc:
            misc = dict()
            misc['id'] = sample.id
            misc['text_raw'] = sample.text_raw
            misc['speaker'] = sample.speaker
            output = (output, misc)

        return output

    def __len__(self):
        return len(self.collection)

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch, pad_id=self.pad_id)


@experimental
class AudioToCharDataset(_AudioTextDataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, transcripts, and durations (in seconds). Each new line is a
    different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath":
    "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the
    transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}
    Args:
        manifest_filepath: Path to manifest json as described above. Can
            be comma-separated paths.
        labels: String containing all the possible characters to map to
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor
            object used to augment loaded audio
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include
            in dataset
        max_utts: Limit number of utterances
        blank_index: blank character index, default = -1
        unk_index: unk_character index, default = -1
        normalize: whether to normalize transcript text (default): True
        bos_id: Id of beginning of sequence symbol to append if not None
        eos_id: Id of end of sequence symbol to append if not None
        load_audio: Boolean flag indicate whether do or not load audio
        add_misc: True if add additional info dict.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'audio_signal': NeuralType(
                ('B', 'T'),
                AudioSignal(freq=self._sample_rate)
                if self is not None and hasattr(self, '_sample_rate')
                else AudioSignal(),
            ),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self,
        manifest_filepath: str,
        labels: Union[str, List[str]],
        sample_rate: int,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: int = 0,
        blank_index: int = -1,
        unk_index: int = -1,
        normalize: bool = True,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        load_audio: bool = True,
        parser: Union[str, Callable] = 'en',
        add_misc: bool = False,
    ):
        self.labels = labels

        parser = parsers.make_parser(
            labels=labels, name=parser, unk_id=unk_index, blank_id=blank_index, do_normalize=normalize,
        )

        super().__init__(
            manifest_filepath=manifest_filepath,
            parser=parser,
            sample_rate=sample_rate,
            int_values=int_values,
            augmentor=augmentor,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            trim=trim,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            load_audio=load_audio,
            add_misc=add_misc,
        )


@experimental
class AudioToCharWithDursDataset(AudioToCharDataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, transcripts, and durations (in seconds). Each new line is a
    different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath":
    "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the
    transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}

    Additionally, user provides path to precomputed durations.

    Args:
        **kwargs: Passed to AudioToCharDataset constructor.
        vocab_notation: Vocabulary to be used for text preprocessing.
        vocab_punct: True if use punctuation for vocab.
        vocab_spaces: True if use spaces for vocab.
        vocab_stresses: True if use stresses for vocab.
        durs_path: String path to pickled durations location.
        rep: True if repeat text according to durs.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports."""
        return {
            'audio': NeuralType(
                ('B', 'T'),
                AudioSignal(freq=self._sample_rate)
                if self is not None and hasattr(self, '_sample_rate')
                else AudioSignal(),
            ),
            'audio_len': NeuralType(('B',), LengthsType()),
            'text': NeuralType(('B', 'T'), LabelsType()),
            'text_len': NeuralType(('B',), LengthsType()),
            'durs': NeuralType(('B', 'T'), LengthsType()),
        }

    @staticmethod
    def make_vocab(**kwargs):
        notation = kwargs.get('vocab_notation', 'chars')
        punct = kwargs.get('vocab_punct', True)
        spaces = kwargs.get('vocab_spaces', False)
        stresses = kwargs.get('vocab_stresses', False)
        if notation == 'chars':
            vocab = vocabs.Chars(punct=punct, spaces=spaces)
        elif notation == 'phonemes':
            vocab = vocabs.Phonemes(punct=punct, stresses=stresses, spaces=spaces)
        else:
            raise ValueError("Unsupported vocab type.")
        return vocab

    def __init__(self, **kwargs):
        self.vocab = self.make_vocab(**kwargs)
        for k in ['vocab_notation', 'vocab_punct', 'vocab_spaces', 'vocab_stresses']:
            kwargs.pop(k, None)
        durs_path = kwargs.pop('durs_path')
        rep = kwargs.pop('rep', False)
        kwargs.setdefault('labels', [])

        super().__init__(**kwargs)

        pth = torch.load(durs_path)
        tag2d = dict(zip(pth['tags'], pth['durs']))
        durs = []
        for i, e in enumerate(self.collection):
            tag = os.path.splitext(os.path.basename(e.audio_file))[0]
            durs.append(tag2d[tag])
        self.durs = durs
        self.rep = rep

    def __getitem__(self, item):
        sample = self.collection[item]
        audio, audio_len, _, _ = super().__getitem__(item)
        text = self.vocab.encode(sample.text_raw)
        text, text_len = torch.tensor(text).long(), torch.tensor(len(text)).long()
        blanks_durs, graphemes_durs = self.durs[item]

        return (
            audio,
            audio_len,
            text,
            text_len,
            blanks_durs,
            graphemes_durs,
        )

    @staticmethod
    def _merge(tensors, dim=0, value=0, dtype=None):
        """Merges list of tensors into one."""
        tensors = [tensor if isinstance(tensor, torch.Tensor) else torch.tensor(tensor) for tensor in tensors]
        dim = dim if dim != -1 else len(tensors[0].shape) - 1
        dtype = tensors[0].dtype if dtype is None else dtype
        max_len = max(tensor.shape[dim] for tensor in tensors)
        new_tensors = []
        for tensor in tensors:
            pad = (2 * len(tensor.shape)) * [0]
            pad[-2 * dim - 1] = max_len - tensor.shape[dim]
            new_tensors.append(F.pad(tensor, pad=pad, value=value))
        return torch.stack(new_tensors).to(dtype=dtype)

    @staticmethod
    def _interleave(x, y):
        """Interleave two tensors."""
        xy = torch.stack([x[:-1], y], dim=1).view(-1)
        xy = F.pad(xy, pad=[0, 1], value=x[-1])
        return xy

    def _collate_fn(self, batch):
        batch = list(zip(*batch))

        asr_batch = _speech_collate_fn(list(zip(*batch[:4])), pad_id=self.vocab.pad)
        audio, audio_len, text, text_len = asr_batch

        text = [
            self._interleave(
                x=torch.empty(len(t) + 1, dtype=torch.long, device=t.device,).fill_(self.vocab.blank), y=t,
            )
            for t in text
        ]
        text = self._merge(text, value=self.vocab.pad, dtype=torch.long)
        text_len = text_len * 2 + 1

        blanks_durs, graphemes_durs = batch[4:]
        durs = [self._interleave(b, c) for b, c in zip(blanks_durs, graphemes_durs)]
        durs = self._merge(durs, dtype=torch.long).to(text.device)

        if self.rep:
            text = self._merge(
                tensors=[torch.repeat_interleave(text1, durs1) for text1, durs1 in zip(text, durs)], dtype=torch.long,
            )
            text_len = durs.sum(-1)

        return (
            audio,
            audio_len,
            text,
            text_len,
            durs,
        )


@experimental
class AudioToBPEDataset(_AudioTextDataset):
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'audio_signal': NeuralType(
                ('B', 'T'),
                AudioSignal(freq=self._sample_rate)
                if self is not None and hasattr(self, '_sample_rate')
                else AudioSignal(),
            ),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self,
        manifest_filepath: str,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        sample_rate: int,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_utts: int = 0,
        trim: bool = False,
        load_audio: bool = True,
        add_misc: bool = False,
    ):
        if hasattr(tokenizer, 'bos_token'):
            bos_id = tokenizer.bos_id
        else:
            bos_id = None

        if hasattr(tokenizer, 'eos_token'):
            eos_id = tokenizer.eos_id
        else:
            eos_id = None

        if hasattr(tokenizer, 'pad_token'):
            pad_id = tokenizer.pad_id
        else:
            pad_id = 0

        class TokenizerWrapper:
            def __init__(self, tokenizer):
                self._tokenizer = tokenizer

            def __call__(self, text):
                t = self._tokenizer.text_to_ids(text)
                return t

        super().__init__(
            manifest_filepath=manifest_filepath,
            parser=TokenizerWrapper(tokenizer),
            sample_rate=sample_rate,
            int_values=int_values,
            augmentor=augmentor,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            trim=trim,
            load_audio=load_audio,
            add_misc=add_misc,
        )


# Ported from https://github.com/NVIDIA/OpenSeq2Seq/blob/master/open_seq2seq/data/speech2text/speech_commands.py
@experimental
class AudioLabelDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, command class, and durations (in seconds). Each new line is a
    different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "label":
    "label", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "label": "label",
    "offset": 301.75, "duration": 0.82}
    Args:
        manifest_filepath: Path to manifest json as described above. Can
            be comma-separated paths.
        labels (Optional[list]): String containing all the possible labels to map to
            if None then automatically picks from ASRSpeechLabel collection.
        featurizer: Initialized featurizer class that converts paths of
            audio to feature tensors
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include
            in dataset
        trim: Boolean flag whether to trim the audio
        load_audio: Boolean flag indicate whether do or not load audio
    """

    def __init__(
        self,
        manifest_filepath,
        featurizer,
        labels=None,
        max_duration=None,
        min_duration=None,
        trim=False,
        load_audio=True,
    ):
        self.collection = collections.ASRSpeechLabel(
            manifests_files=manifest_filepath.split(','), min_duration=min_duration, max_duration=max_duration,
        )

        self.featurizer = featurizer
        self.trim = trim
        self.load_audio = load_audio

        self.labels = labels if labels else self.collection.uniq_labels
        self.num_commands = len(self.labels)

        self.label2id, self.id2label = {}, {}
        for label_id, label in enumerate(self.labels):
            self.label2id[label] = label_id
            self.id2label[label_id] = label

    def __getitem__(self, index):
        sample = self.collection[index]
        if self.load_audio:
            offset = sample.offset

            if offset is None:
                offset = 0

            features = self.featurizer.process(
                sample.audio_file, offset=offset, duration=sample.duration, trim=self.trim,
            )
            f, fl = features, torch.tensor(features.shape[0]).long()
        else:
            f, fl = None, None

        t = self.label2id[sample.label]
        tl = 1  # For compatibility with collate_fn used later

        return f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

    def __len__(self):
        return len(self.collection)

    def _collate_fn(self, batch):
        """collate batch of audio sig, audio len, tokens (single token), tokens len (1)
        Args:
            batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
                   LongTensor):  A tuple of tuples of signal, signal lengths,
                   encoded tokens, and encoded tokens length.  This collate func
                   assumes the signals are 1d torch tensors (i.e. mono audio).
        """
        return _speech_collate_fn(batch, pad_id=0)


@experimental
class _TarredAudioToTextDataset(IterableDataset):
    """
    A similar Dataset to the AudioToCharDataset/AudioToBPEDataset, but which loads tarred audio files.

    Accepts a single comma-separated JSON manifest file (in the same style as for the AudioToCharDataset/AudioToBPEDataset),
    as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
    contain the information for one audio file, including at least the transcript and name of the audio
    file within the tarball.

    Valid formats for the audio_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].

    Note: For brace expansion in (1), there may be cases where `{x..y}` syntax cannot be used due to shell interference.
    This occurs most commonly inside SLURM scripts. Therefore we provide a few equivalent replacements.
    Supported opening braces - { <=> (, [, < and the special tag _OP_.
    Supported closing braces - } <=> ), ], > and the special tag _CL_.
    For SLURM based tasks, we suggest the use of the special tags for ease of use.

    See the WebDataset documentation for more information about accepted data and input formats.

    If using multiple processes the number of shards should be divisible by the number of workers to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    In addition, if using mutiprocessing, each shard MUST HAVE THE SAME NUMBER OF ENTRIES after filtering
    is applied. We currently do not check for this, but your program may hang if the shards are uneven!

    Notice that a few arguments are different from the AudioToCharDataset; for example, shuffle (bool) has been
    replaced by shuffle_n (int).

    Additionally, please note that the len() of this DataLayer is assumed to be the length of the manifest
    after filtering. An incorrect manifest length may lead to some DataLoader issues down the line.

    Args:
        audio_tar_filepaths: Either a list of audio tarball filepaths, or a
            string (can be brace-expandable).
        manifest_filepath (str): Path to the manifest.
        parser (callable): A callable which is used to pre-process the text output.
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor
            object used to augment loaded audio
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        max_utts (int): Limit number of utterances. 0 means no maximum.
        blank_index (int): Blank character index, defaults to -1.
        unk_index (int): Unknown character index, defaults to -1.
        normalize (bool): Dataset parameter.
            Whether to use automatic text cleaning.
            It is highly recommended to manually clean text for best results.
            Defaults to True.
        trim (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        bos_id (id): Dataset parameter.
            Beginning of string symbol id used for seq2seq models.
            Defaults to None.
        eos_id (id): Dataset parameter.
            End of string symbol id used for seq2seq models.
            Defaults to None.
        pad_id (id): Token used to pad when collating samples in batches.
            If this is None, pads using 0s.
            Defaults to None.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 0.
    """

    def __init__(
        self,
        audio_tar_filepaths: Union[str, List[str]],
        manifest_filepath: str,
        parser: Callable,
        sample_rate: int,
        int_values: bool = False,
        augmentor: Optional['nemo.collections.asr.parts.perturb.AudioAugmentor'] = None,
        shuffle_n: int = 0,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_utts: int = 0,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        add_misc: bool = False,
        pad_id: int = 0,
        global_rank: int = 0,
        world_size: int = 0,
    ):
        self.collection = collections.ASRAudioText(
            manifests_files=manifest_filepath.split(','),
            parser=parser,
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
            index_by_file_id=True,  # Must set this so the manifest lines can be indexed by file ID
        )

        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)
        self.trim = trim
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.pad_id = pad_id
        self._add_misc = add_misc

        if isinstance(audio_tar_filepaths, str):
            # Replace '(' and '[' with '{'
            brace_keys_open = ['(', '[', '<', '_OP_']
            for bkey in brace_keys_open:
                if bkey in audio_tar_filepaths:
                    audio_tar_filepaths = audio_tar_filepaths.replace(bkey, "{")

            # Replace ')' and ']' with '}'
            brace_keys_close = [')', ']', '>', '_CL_']
            for bkey in brace_keys_close:
                if bkey in audio_tar_filepaths:
                    audio_tar_filepaths = audio_tar_filepaths.replace(bkey, "}")

        # Check for distributed and partition shards accordingly
        if world_size > 1:
            if isinstance(audio_tar_filepaths, str):
                # Brace expand
                audio_tar_filepaths = list(braceexpand.braceexpand(audio_tar_filepaths))

            if len(audio_tar_filepaths) % world_size != 0:
                logging.warning(
                    f"Number of shards in tarred dataset ({len(audio_tar_filepaths)}) is not divisible "
                    f"by number of distributed workers ({world_size})."
                )

            begin_idx = (len(audio_tar_filepaths) // world_size) * global_rank
            end_idx = begin_idx + (len(audio_tar_filepaths) // world_size)
            audio_tar_filepaths = audio_tar_filepaths[begin_idx:end_idx]
            logging.info(
                "Partitioning tarred dataset: process (%d) taking shards [%d, %d)", global_rank, begin_idx, end_idx
            )

        # Put together WebDataset
        self._dataset = (
            wd.Dataset(audio_tar_filepaths)
            .shuffle(shuffle_n)
            .rename(audio='wav', key='__key__')
            .to_tuple('audio', 'key')
            .pipe(self._filter)
            .map(f=self._build_sample)
        )

    def _filter(self, iterator):
        """This function is used to remove samples that have been filtered out by ASRAudioText already.
        Otherwise, we would get a KeyError as _build_sample attempts to find the manifest entry for a sample
        that was filtered out (e.g. for duration).
        Note that if using multi-GPU training, filtering may lead to an imbalance in samples in each shard,
        which may make your code hang as one process will finish before the other.
        """

        class TarredAudioFilter:
            def __init__(self, collection):
                self.iterator = iterator
                self.collection = collection

            def __iter__(self):
                return self

            def __next__(self):
                while True:
                    audio_bytes, audio_filename = next(self.iterator)
                    file_id, _ = os.path.splitext(os.path.basename(audio_filename))
                    if file_id in self.collection.mapping:
                        return audio_bytes, audio_filename

        return TarredAudioFilter(self.collection)

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch, self.pad_id)

    def _build_sample(self, tup):
        """Builds the training sample by combining the data from the WebDataset with the manifest info.
        """
        audio_bytes, audio_filename = tup

        # Grab manifest entry from self.collection
        file_id, _ = os.path.splitext(os.path.basename(audio_filename))
        manifest_idx = self.collection.mapping[file_id]
        manifest_entry = self.collection[manifest_idx]

        offset = manifest_entry.offset
        if offset is None:
            offset = 0

        # Convert audio bytes to IO stream for processing (for SoundFile to read)
        audio_filestream = io.BytesIO(audio_bytes)
        features = self.featurizer.process(
            audio_filestream,
            offset=offset,
            duration=manifest_entry.duration,
            trim=self.trim,
            orig_sr=manifest_entry.orig_sr,
        )
        audio_filestream.close()

        # Audio features
        f, fl = features, torch.tensor(features.shape[0]).long()

        # Text features
        t, tl = manifest_entry.text_tokens, len(manifest_entry.text_tokens)
        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1

        return f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

    def __iter__(self):
        return self._dataset.__iter__()

    def __len__(self):
        return len(self.collection)


@experimental
class TarredAudioToCharDataset(_TarredAudioToTextDataset):
    """
    A similar Dataset to the AudioToCharDataset, but which loads tarred audio files.

    Accepts a single comma-separated JSON manifest file (in the same style as for the AudioToCharDataset),
    as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
    contain the information for one audio file, including at least the transcript and name of the audio
    file within the tarball.

    Valid formats for the audio_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].

    See the WebDataset documentation for more information about accepted data and input formats.

    If using multiple processes the number of shards should be divisible by the number of workers to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    In addition, if using mutiprocessing, each shard MUST HAVE THE SAME NUMBER OF ENTRIES after filtering
    is applied. We currently do not check for this, but your program may hang if the shards are uneven!

    Notice that a few arguments are different from the AudioToCharDataset; for example, shuffle (bool) has been
    replaced by shuffle_n (int).

    Additionally, please note that the len() of this DataLayer is assumed to be the length of the manifest
    after filtering. An incorrect manifest length may lead to some DataLoader issues down the line.

    Args:
        audio_tar_filepaths: Either a list of audio tarball filepaths, or a
            string (can be brace-expandable).
        manifest_filepath (str): Path to the manifest.
        labels (list): List of characters that can be output by the ASR model.
            For Jasper, this is the 28 character set {a-z '}. The CTC blank
            symbol is automatically added later for models using ctc.
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor
            object used to augment loaded audio
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        max_utts (int): Limit number of utterances. 0 means no maximum.
        blank_index (int): Blank character index, defaults to -1.
        unk_index (int): Unknown character index, defaults to -1.
        normalize (bool): Dataset parameter.
            Whether to use automatic text cleaning.
            It is highly recommended to manually clean text for best results.
            Defaults to True.
        trim (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        bos_id (id): Dataset parameter.
            Beginning of string symbol id used for seq2seq models.
            Defaults to None.
        eos_id (id): Dataset parameter.
            End of string symbol id used for seq2seq models.
            Defaults to None.
        pad_id (id): Token used to pad when collating samples in batches.
            If this is None, pads using 0s.
            Defaults to None.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 0.
    """

    def __init__(
        self,
        audio_tar_filepaths: Union[str, List[str]],
        manifest_filepath: str,
        labels: List[str],
        sample_rate: int,
        int_values: bool = False,
        augmentor: Optional['nemo.collections.asr.parts.perturb.AudioAugmentor'] = None,
        shuffle_n: int = 0,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_utts: int = 0,
        blank_index: int = -1,
        unk_index: int = -1,
        normalize: bool = True,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        parser: Optional[str] = 'en',
        add_misc: bool = False,
        pad_id: int = 0,
        global_rank: int = 0,
        world_size: int = 0,
    ):
        self.labels = labels

        parser = parsers.make_parser(
            labels=labels, name=parser, unk_id=unk_index, blank_id=blank_index, do_normalize=normalize
        )

        super().__init__(
            audio_tar_filepaths=audio_tar_filepaths,
            manifest_filepath=manifest_filepath,
            parser=parser,
            sample_rate=sample_rate,
            int_values=int_values,
            augmentor=augmentor,
            shuffle_n=shuffle_n,
            min_duration=min_duration,
            max_duration=max_duration,
            max_utts=max_utts,
            trim=trim,
            bos_id=bos_id,
            eos_id=eos_id,
            add_misc=add_misc,
            pad_id=pad_id,
            global_rank=global_rank,
            world_size=world_size,
        )


@experimental
class TarredAudioToBPEDataset(_TarredAudioToTextDataset):
    """
    A similar Dataset to the AudioToBPEDataset, but which loads tarred audio files.

    Accepts a single comma-separated JSON manifest file (in the same style as for the AudioToBPEDataset),
    as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
    contain the information for one audio file, including at least the transcript and name of the audio
    file within the tarball.

    Valid formats for the audio_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].

    See the WebDataset documentation for more information about accepted data and input formats.

    If using multiple processes the number of shards should be divisible by the number of workers to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    In addition, if using mutiprocessing, each shard MUST HAVE THE SAME NUMBER OF ENTRIES after filtering
    is applied. We currently do not check for this, but your program may hang if the shards are uneven!

    Notice that a few arguments are different from the AudioToBPEDataset; for example, shuffle (bool) has been
    replaced by shuffle_n (int).

    Additionally, please note that the len() of this DataLayer is assumed to be the length of the manifest
    after filtering. An incorrect manifest length may lead to some DataLoader issues down the line.

    Args:
        audio_tar_filepaths: Either a list of audio tarball filepaths, or a
            string (can be brace-expandable).
        manifest_filepath (str): Path to the manifest.
        tokenizer (TokenizerSpec): Either a Word Piece Encoding tokenizer (BERT),
            or a Sentence Piece Encoding tokenizer (BPE). The CTC blank
            symbol is automatically added later for models using ctc.
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor
            object used to augment loaded audio
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        max_utts (int): Limit number of utterances. 0 means no maximum.
        trim (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        pad_id (id): Token used to pad when collating samples in batches.
            If this is None, pads using 0s.
            Defaults to None.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 0.
    """

    def __init__(
        self,
        audio_tar_filepaths: Union[str, List[str]],
        manifest_filepath: str,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        sample_rate: int,
        int_values: bool = False,
        augmentor: Optional['nemo.collections.asr.parts.perturb.AudioAugmentor'] = None,
        shuffle_n: int = 0,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_utts: int = 0,
        trim: bool = False,
        add_misc: bool = False,
        global_rank: int = 0,
        world_size: int = 0,
    ):
        if hasattr(tokenizer, 'bos_token'):
            bos_id = tokenizer.bos_id
        else:
            bos_id = None

        if hasattr(tokenizer, 'eos_token'):
            eos_id = tokenizer.eos_id
        else:
            eos_id = None

        if hasattr(tokenizer, 'pad_token'):
            pad_id = tokenizer.pad_id
        else:
            pad_id = 0

        class TokenizerWrapper:
            def __init__(self, tokenizer):
                self._tokenizer = tokenizer

            def __call__(self, text):
                t = self._tokenizer.text_to_ids(text)
                return t

        super().__init__(
            audio_tar_filepaths=audio_tar_filepaths,
            manifest_filepath=manifest_filepath,
            parser=TokenizerWrapper(tokenizer),
            sample_rate=sample_rate,
            int_values=int_values,
            augmentor=augmentor,
            shuffle_n=shuffle_n,
            min_duration=min_duration,
            max_duration=max_duration,
            max_utts=max_utts,
            trim=trim,
            bos_id=bos_id,
            eos_id=eos_id,
            add_misc=add_misc,
            pad_id=pad_id,
            global_rank=global_rank,
            world_size=world_size,
        )
