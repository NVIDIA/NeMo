# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
# =============================================================================
"""This package contains Neural Modules responsible for ASR data layers."""

from functools import partial
from typing import Any, Dict, List, Optional

import torch

import nemo
from .parts.dataset import AudioDataset, AudioLabelDataset, KaldiFeatureDataset, TranscriptDataset, seq_collate_fn
from .parts.features import WaveformFeaturizer
from .parts.perturb import AudioAugmentor, perturbation_types
from nemo.backends.pytorch import DataLayerNM
from nemo.core import DeviceType
from nemo.core.neural_types import *
from nemo.utils.decorators import add_port_docs
from nemo.utils.misc import pad_to

__all__ = [
    'AudioToTextDataLayer',
    'KaldiFeatureDataLayer',
    'TranscriptDataLayer',
    'AudioToSpeechLabelDataLayer',
]

logging = nemo.logging


class AudioToTextDataLayer(DataLayerNM):
    """Data Layer for general ASR tasks.

    Module which reads ASR labeled data. It accepts comma-separated
    JSON manifest files describing the correspondence between wav audio files
    and their transcripts. JSON files should be of the following format::

        {"audio_filepath": path_to_wav_0, "duration": time_in_sec_0, "text": \
transcript_0}
        ...
        {"audio_filepath": path_to_wav_n, "duration": time_in_sec_n, "text": \
transcript_n}

    Args:
        manifest_filepath (str): Dataset parameter.
            Path to JSON containing data.
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
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            # 'audio_signal': NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # 'a_sig_length': NeuralType({0: AxisType(BatchTag)}),
            # 'transcripts': NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # 'transcript_length': NeuralType({0: AxisType(BatchTag)}),
            'audio_signal': NeuralType(
                ('B', 'T'),
                AudioSignal(freq=self._sample_rate)
                if self is not None and self._sample_rate is not None
                else AudioSignal(),
            ),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self,
        manifest_filepath,
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
        self._sample_rate = sample_rate
        self._featurizer = WaveformFeaturizer(sample_rate=self._sample_rate, int_values=int_values, augmentor=None)

        # Set up dataset
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
        self._dataset = AudioDataset(**dataset_params)
        self._batch_size = batch_size

        # Set up data loader
        if self._placement == DeviceType.AllGpu:
            logging.info("Parallelizing Datalayer.")
            sampler = torch.utils.data.distributed.DistributedSampler(self._dataset)
        else:
            sampler = None

        if batch_size == -1:
            batch_size = len(self._dataset)

        pad_id = 0 if pad_id is None else pad_id
        self._dataloader = torch.utils.data.DataLoader(
            dataset=self._dataset,
            batch_size=batch_size,
            collate_fn=partial(seq_collate_fn, token_pad_value=pad_id),
            drop_last=drop_last,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
        )

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self._dataloader


class KaldiFeatureDataLayer(DataLayerNM):
    """Data layer for reading generic Kaldi-formatted data.

    Module that reads ASR labeled data that is in a Kaldi-compatible format.
    It assumes that you have a directory that contains:

    - feats.scp: A mapping from utterance IDs to .ark files that
            contains the corresponding MFCC (or other format) data
    - text: A mapping from utterance IDs to transcripts
    - utt2dur (optional): A mapping from utterance IDs to audio durations,
            needed if you want to filter based on duration

    Args:
        kaldi_dir (str): Directory that contains the above files.
        labels (list): List of characters that can be output by the ASR model,
            e.g. {a-z '} for Jasper. The CTC blank symbol is automatically
            added later for models using CTC.
        batch_size (int): batch size
        eos_id (str): End of string symbol used for seq2seq models.
            Defaults to None.
        min_duration (float): All training files which have a duration less
            than min_duration are dropped. Can't be used if the `utt2dur` file
            does not exist. Defaults to None.
        max_duration (float): All training files which have a duration more
            than max_duration are dropped. Can't be used if the `utt2dur` file
            does not exist. Defaults to None.
        normalize_transcripts (bool): Whether to use automatic text cleaning.
            It is highly recommended to manually clean text for best results.
            Defaults to True.
        drop_last (bool): See PyTorch DataLoader. Defaults to False.
        shuffle (bool): See PyTorch DataLoader. Defaults to True.
        num_workers (int): See PyTorch DataLoader. Defaults to 0.
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.


        """
        return {
            # 'processed_signal': NeuralType(
            #    {0: AxisType(BatchTag), 1: AxisType(SpectrogramSignalTag), 2: AxisType(ProcessedTimeTag),}
            # ),
            # 'processed_length': NeuralType({0: AxisType(BatchTag)}),
            # 'transcripts': NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # 'transcript_length': NeuralType({0: AxisType(BatchTag)}),
            'processed_signal': NeuralType(('B', 'D', 'T'), SpectrogramType()),
            'transcripts': NeuralType(('B', 'T'), ChannelType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self,
        kaldi_dir,
        labels,
        batch_size,
        min_duration=None,
        max_duration=None,
        normalize_transcripts=True,
        drop_last=False,
        shuffle=True,
        num_workers=0,
    ):
        super().__init__()

        # Set up dataset
        dataset_params = {
            "kaldi_dir": kaldi_dir,
            "labels": labels,
            "min_duration": min_duration,
            "max_duration": max_duration,
            "normalize": normalize_transcripts,
        }
        self._dataset = KaldiFeatureDataset(**dataset_params)

        # Set up data loader
        if self._placement == DeviceType.AllGpu:
            logging.info("Parallelizing DATALAYER")
            sampler = torch.utils.data.distributed.DistributedSampler(self._dataset)
        else:
            sampler = None

        self._dataloader = torch.utils.data.DataLoader(
            dataset=self._dataset,
            batch_size=batch_size,
            collate_fn=self._collate_fn,
            drop_last=drop_last,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
        )

    @staticmethod
    def _collate_fn(batch):
        """Collate batch of (features, feature len, tokens, tokens len).
        Kaldi generally uses MFCC (and PLP) features.

        Args:
            batch: A batch of elements, where each element is a tuple of
                features, feature length, tokens, and token
                length for a single sample.

        Returns:
            The same batch, with the features and token length padded
            to the maximum of the batch.
        """
        # Find max lengths of features and tokens in the batch
        _, feat_lens, _, token_lens = zip(*batch)
        max_feat_len = max(feat_lens).item()
        max_tokens_len = max(token_lens).item()

        # Pad features and tokens to max
        features, tokens = [], []
        for feat, feat_len, tkns, tkns_len in batch:
            feat_len = feat_len.item()
            if feat_len < max_feat_len:
                pad = [0, max_feat_len - feat_len]
                feat = torch.nn.functional.pad(feat, pad)
            features.append(feat)

            tkns_len = tkns_len.item()
            if tkns_len < max_tokens_len:
                pad = [0, max_tokens_len - tkns_len]
                tkns = torch.nn.functional.pad(tkns, pad)
            tokens.append(tkns)

        features = torch.stack(features)
        feature_lens = torch.stack(feat_lens)
        tokens = torch.stack(tokens)
        token_lens = torch.stack(token_lens)

        return features, feature_lens, tokens, token_lens

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self._dataloader


class TranscriptDataLayer(DataLayerNM):
    """A simple Neural Module for loading textual transcript data.
    The path, labels, and eos_id arguments are dataset parameters.

    Args:
        pad_id (int): Label position of padding symbol
        batch_size (int): Size of batches to generate in data loader
        drop_last (bool): Whether we drop last (possibly) incomplete batch.
            Defaults to False.
        num_workers (int): Number of processes to work on data loading (0 for
            just main process).
            Defaults to 0.
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        texts:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        texts_length:
            0: AxisType(BatchTag)

        """
        return {
            # 'texts': NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # 'texts_length': NeuralType({0: AxisType(BatchTag)}),
            'texts': NeuralType(('B', 'T'), LabelsType()),
            'texts_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self,
        path,
        labels,
        batch_size,
        bos_id=None,
        eos_id=None,
        pad_id=None,
        drop_last=False,
        num_workers=0,
        shuffle=True,
    ):
        super().__init__()

        # Set up dataset
        dataset_params = {
            'path': path,
            'labels': labels,
            'bos_id': bos_id,
            'eos_id': eos_id,
        }

        self._dataset = TranscriptDataset(**dataset_params)

        # Set up data loader
        if self._placement == DeviceType.AllGpu:
            sampler = torch.utils.data.distributed.DistributedSampler(self._dataset)
        else:
            sampler = None

        pad_id = 0 if pad_id is None else pad_id

        # noinspection PyTypeChecker
        self._dataloader = torch.utils.data.DataLoader(
            dataset=self._dataset,
            batch_size=batch_size,
            collate_fn=partial(self._collate_fn, pad_id=pad_id, pad8=True),
            drop_last=drop_last,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
        )

    @staticmethod
    def _collate_fn(batch, pad_id, pad8=False):
        texts_list, texts_len = zip(*batch)
        max_len = max(texts_len)
        if pad8:
            max_len = pad_to(max_len, 8)

        texts = torch.empty(len(texts_list), max_len, dtype=torch.long)
        texts.fill_(pad_id)

        for i, s in enumerate(texts_list):
            texts[i].narrow(0, 0, s.size(0)).copy_(s)

        if len(texts.shape) != 2:
            raise ValueError(f"Texts in collate function have shape {texts.shape}," f" should have 2 dimensions.")

        return texts, torch.stack(texts_len)

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self._dataloader


# Ported from https://github.com/NVIDIA/OpenSeq2Seq/blob/master/open_seq2seq/data/speech2text/speech_commands.py
class AudioToSpeechLabelDataLayer(DataLayerNM):
    """Data Layer for general speech classification.

    Module which reads speech recognition with target label. It accepts comma-separated
    JSON manifest files describing the correspondence between wav audio files
    and their target labels. JSON files should be of the following format::

        {"audio_filepath": path_to_wav_0, "duration": time_in_sec_0, "label": \
target_label_0}
        ...
        {"audio_filepath": path_to_wav_n, "duration": time_in_sec_n, "label": \
target_label_n}

    Args:
        manifest_filepath (str): Dataset parameter.
            Path to JSON containing data.
        labels (list): Dataset parameter.
            List of target classes that can be output by the speech recognition model.
        batch_size (int): batch size
        sample_rate (int): Target sampling rate for data. Audio files will be
            resampled to sample_rate if it is not already.
            Defaults to 16000.
        int_values (bool): Bool indicating whether the audio file is saved as
            int data or float data.
            Defaults to False.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
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
        augmentor (dict): Optional dictionary of str -> kwargs (dict)
            which is parsed and used to initialize an AudioAugmentor.
            Note: It is crucial that each individual augmentation has
            a keyword `prob`, that defines a float probability in the
            the range [0, 1] of this augmentation being applied.
            If this keyword is not present, then the augmentation is
            disabled and a warning is logged.
    """

    @property
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            # 'audio_signal': NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # 'a_sig_length': NeuralType({0: AxisType(BatchTag)}),
            # 'label': NeuralType({0: AxisType(BatchTag)}),
            # 'label_length': NeuralType({0: AxisType(BatchTag)}),
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'label': NeuralType(tuple('B'), LabelsType()),
            'label_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self,
        *,
        manifest_filepath: str,
        labels: List[str],
        batch_size: int,
        sample_rate: int = 16000,
        int_values: bool = False,
        num_workers: int = 0,
        shuffle: bool = True,
        min_duration: Optional[float] = 0.1,
        max_duration: Optional[float] = None,
        trim_silence: bool = False,
        drop_last: bool = False,
        load_audio: bool = True,
        augmentor: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        super(AudioToSpeechLabelDataLayer, self).__init__()

        self._manifest_filepath = manifest_filepath
        self._labels = labels
        self._sample_rate = sample_rate

        if augmentor is not None:
            augmentor = self._process_augmentations(augmentor)

        self._featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)

        dataset_params = {
            'manifest_filepath': manifest_filepath,
            'labels': labels,
            'featurizer': self._featurizer,
            'max_duration': max_duration,
            'min_duration': min_duration,
            'trim': trim_silence,
            'load_audio': load_audio,
        }
        self._dataset = AudioLabelDataset(**dataset_params)

        # Set up data loader
        if self._placement == DeviceType.AllGpu:
            logging.info("Parallelizing Datalayer.")
            sampler = torch.utils.data.distributed.DistributedSampler(self._dataset)
        else:
            sampler = None

        self._dataloader = torch.utils.data.DataLoader(
            dataset=self._dataset,
            batch_size=batch_size,
            collate_fn=partial(seq_collate_fn, token_pad_value=0),
            drop_last=drop_last,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
        )

    def __len__(self):
        return len(self._dataset)

    def _process_augmentations(self, augmentor) -> AudioAugmentor:
        augmentations = []
        for augment_name, augment_kwargs in augmentor.items():
            prob = augment_kwargs.get('prob', None)

            if prob is None:
                logging.error(
                    f'Augmentation "{augment_name}" will not be applied as '
                    f'keyword argument "prob" was not defined for this augmentation.'
                )

            else:
                _ = augment_kwargs.pop('prob')

                try:
                    augmentation = perturbation_types[augment_name](**augment_kwargs)
                    augmentations.append([prob, augmentation])
                except KeyError:
                    logging.error(f"Invalid perturbation name. Allowed values : {perturbation_types.keys()}")

        augmentor = AudioAugmentor(perturbations=augmentations)
        return augmentor

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self._dataloader
