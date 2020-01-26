# Copyright (c) 2019 NVIDIA Corporation
import json
import os
import pathlib
from typing import Dict, List, Optional

import librosa
import numpy as np
import torch

# noinspection PyPep8Naming
import torch.nn.functional as F

import nemo
from nemo.backends.pytorch.nm import DataLayerNM
from nemo.collections.asr.parts import AudioDataset, WaveformFeaturizer, collections
from nemo.collections.asr.parts import features as asr_parts_features
from nemo.collections.asr.parts import parsers
from nemo.collections.tts.fastspeech import text_norm
from nemo.core.neural_types import AudioSignal, EmbeddedTextType, LengthsType, MaskType, NeuralType


class FastSpeechDataset:
    def __init__(self, audio_dataset, durs_dir):
        self._audio_dataset = audio_dataset
        self._durs_dir = durs_dir

    def __getitem__(self, index):
        audio, audio_len, text, text_len = self._audio_dataset[index]
        dur_true = torch.tensor(np.load(os.path.join(self._durs_dir, f'{index}.npy'))).long()
        return dict(audio=audio, audio_len=audio_len, text=text, text_len=text_len, dur_true=dur_true)

    def __len__(self):
        return len(self._audio_dataset)


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
    def output_ports(self) -> Optional[Dict[str, NeuralType]]:
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
        self._dataset = FastSpeechDataset(audio_dataset, durs_dir)
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
