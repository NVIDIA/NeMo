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

__all__ = ['Audio2TextDataset', 'seq_collate_fn']

import torch

from nemo.collections.asr.parts import collections, parsers
from nemo.core.classes import Dataset
from nemo.utils.decorators import experimental


@experimental
class AudioToTextDataset(Dataset):
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
        featurizer: Initialized featurizer class that converts paths of
            audio to feature tensors
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

    def __init__(
        self,
        manifest_filepath,
        labels,
        featurizer,
        max_duration=None,
        min_duration=None,
        max_utts=0,
        blank_index=-1,
        unk_index=-1,
        normalize=True,
        trim=False,
        bos_id=None,
        eos_id=None,
        load_audio=True,
        parser='en',
        add_misc=False,
    ):
        self.collection = collections.ASRAudioText(
            manifests_files=manifest_filepath.split(','),
            parser=parsers.make_parser(
                labels=labels, name=parser, unk_id=unk_index, blank_id=blank_index, do_normalize=normalize,
            ),
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
        )

        self.featurizer = featurizer
        self.trim = trim
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.load_audio = load_audio
        self._add_misc = add_misc

    # TODO: add typing decorator for datasets here
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


def seq_collate_fn(batch, token_pad_value=0):
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
            tokens_i = torch.nn.functional.pad(tokens_i, pad, value=token_pad_value)
        tokens.append(tokens_i)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None
    tokens = torch.stack(tokens)
    tokens_lengths = torch.stack(tokens_lengths)

    return audio_signal, audio_lengths, tokens, tokens_lengths
