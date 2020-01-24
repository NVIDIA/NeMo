# Copyright (c) 2019 NVIDIA Corporation
import torch
from torch.utils.data import Dataset

from nemo.collections.asr.parts import collections, parsers
from nemo.collections.asr.parts.segment import AudioSegment


class AudioOnlyDataset(Dataset):
    def __init__(
        self, manifest_filepath, n_segments=0, max_duration=None, min_duration=None, trim=False,
    ):
        """See AudioDataLayer"""
        self.collection = collections.ASRAudioText(
            manifests_files=manifest_filepath.split(','),
            parser=parsers.make_parser(),
            min_duration=min_duration,
            max_duration=max_duration,
        )
        self.trim = trim
        self.n_segments = n_segments

    def AudioCollateFunc(self, batch):
        def find_max_len(seq, index):
            max_len = -1
            for item in seq:
                if item[index].size(0) > max_len:
                    max_len = item[index].size(0)
            return max_len

        batch_size = len(batch)

        audio_signal, audio_lengths = None, None
        if batch[0][0] is not None:
            if self.n_segments > 0:
                max_audio_len = self.n_segments
            else:
                max_audio_len = find_max_len(batch, 0)

            audio_signal = torch.zeros(batch_size, max_audio_len, dtype=torch.float)
            audio_lengths = []
            for i, s in enumerate(batch):
                audio_signal[i].narrow(0, 0, s[0].size(0)).copy_(s[0])
                audio_lengths.append(s[1])
            audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)

        return audio_signal, audio_lengths

    def __getitem__(self, index):
        example = self.collection[index]
        features = AudioSegment.segment_from_file(example.audio_file, n_segments=self.n_segments, trim=self.trim,)
        features = torch.tensor(features.samples, dtype=torch.float)
        f, fl = features, torch.tensor(features.shape[0]).long()

        return f, fl

    def __len__(self):
        return len(self.collection)
