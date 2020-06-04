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

import copy
import io
import os
from functools import partial
from typing import Any, Dict, List, Optional, Union

import braceexpand
import torch
import webdataset as wd

from .parts.collections import ASRAudioText
from .parts.dataset import (
    AudioDataset,
    AudioLabelDataset,
    KaldiFeatureDataset,
    TranscriptDataset,
    fixed_seq_collate_fn,
    seq_collate_fn,
)
from .parts.features import WaveformFeaturizer
from .parts.parsers import make_parser
from .parts.perturb import AudioAugmentor, perturbation_types
from nemo.backends.pytorch import DataLayerNM
from nemo.core import DeviceType
from nemo.core.neural_types import *
from nemo.utils import logging
from nemo.utils.decorators import add_port_docs
from nemo.utils.misc import pad_to

__all__ = [
    'AudioToTextDataLayer',
    'TarredAudioToTextDataLayer',
    'KaldiFeatureDataLayer',
    'TranscriptDataLayer',
    'AudioToSpeechLabelDataLayer',
]


def _process_augmentations(augmenter) -> AudioAugmentor:
    """Process list of online data augmentations.

    Accepts either an AudioAugmentor object with pre-defined augmentations,
    or a dictionary that points to augmentations that have been defined.

    If a dictionary is passed, must follow the below structure:
    Dict[str, Dict[str, Any]]: Which refers to a dictionary of string
    names for augmentations, defined in `asr/parts/perturb.py`.

    The inner dictionary may contain key-value arguments of the specific
    augmentation, along with an essential key `prob`. `prob` declares the
    probability of the augmentation being applied, and must be a float
    value in the range [0, 1].

    # Example in YAML config file

    Augmentations are generally applied only during training, so we can add
    these augmentations to our yaml config file, and modify the behaviour
    for training and evaluation.

    ```yaml
    AudioToSpeechLabelDataLayer:
        ...  # Parameters shared between train and evaluation time
        train:
            augmentor:
                shift:
                    prob: 0.5
                    min_shift_ms: -5.0
                    max_shift_ms: 5.0
                white_noise:
                    prob: 1.0
                    min_level: -90
                    max_level: -46
                ...
        eval:
            ...
    ```

    Then in the training script,

    ```python
    import copy
    from ruamel.yaml import YAML

    yaml = YAML(typ="safe")
    with open(model_config) as f:
        params = yaml.load(f)

    # Train Config for Data Loader
    train_dl_params = copy.deepcopy(params["AudioToTextDataLayer"])
    train_dl_params.update(params["AudioToTextDataLayer"]["train"])
    del train_dl_params["train"]
    del train_dl_params["eval"]

    data_layer_train = nemo_asr.AudioToTextDataLayer(
        ...,
        **train_dl_params,
    )

    # Evaluation Config for Data Loader
    eval_dl_params = copy.deepcopy(params["AudioToTextDataLayer"])
    eval_dl_params.update(params["AudioToTextDataLayer"]["eval"])
    del eval_dl_params["train"]
    del eval_dl_params["eval"]

    data_layer_eval = nemo_asr.AudioToTextDataLayer(
        ...,
        **eval_dl_params,
    )
    ```

    # Registering your own Augmentations

    To register custom augmentations to obtain the above convenience of
    the declaring the augmentations in YAML, you can put additional keys in
    `perturbation_types` dictionary as follows.

    ```python
    from nemo.collections.asr.parts import perturb

    # Define your own perturbation here
    class CustomPerturbation(perturb.Perturbation):
        ...

    perturb.register_perturbation(name_of_perturbation, CustomPerturbation)
    ```

    Args:
        augmenter: AudioAugmentor object or
            dictionary of str -> kwargs (dict) which is parsed and used
            to initialize an AudioAugmentor.
            Note: It is crucial that each individual augmentation has
            a keyword `prob`, that defines a float probability in the
            the range [0, 1] of this augmentation being applied.
            If this keyword is not present, then the augmentation is
            disabled and a warning is logged.

    Returns: AudioAugmentor object
    """
    if isinstance(augmenter, AudioAugmentor):
        return augmenter

    if not type(augmenter) == dict:
        raise ValueError("Cannot parse augmenter. Must be a dict or an AudioAugmentor object ")

    augmenter = copy.deepcopy(augmenter)

    augmentations = []
    for augment_name, augment_kwargs in augmenter.items():
        prob = augment_kwargs.get('prob', None)

        if prob is None:
            raise KeyError(
                f'Augmentation "{augment_name}" will not be applied as '
                f'keyword argument "prob" was not defined for this augmentation.'
            )

        else:
            _ = augment_kwargs.pop('prob')

            if prob < 0.0 or prob > 1.0:
                raise ValueError("`prob` must be a float value between 0 and 1.")

            try:
                augmentation = perturbation_types[augment_name](**augment_kwargs)
                augmentations.append([prob, augmentation])
            except KeyError:
                raise KeyError(f"Invalid perturbation name. Allowed values : {perturbation_types.keys()}")

    augmenter = AudioAugmentor(perturbations=augmentations)
    return augmenter


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
        bos_id (id): Dataset parameter.
            Beginning of string symbol id used for seq2seq models.
            Defaults to None.
        eos_id (id): Dataset parameter.
            End of string symbol id used for seq2seq models.
            Defaults to None.
        pad_id (id): Token used to pad when collating samples in batches.
            If this is None, pads using 0s.
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
        augmentor (AudioAugmentor or dict): Optional AudioAugmentor or
            dictionary of str -> kwargs (dict) which is parsed and used
            to initialize an AudioAugmentor.
            Note: It is crucial that each individual augmentation has
            a keyword `prob`, that defines a float probability in the
            the range [0, 1] of this augmentation being applied.
            If this keyword is not present, then the augmentation is
            disabled and a warning is logged.
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
        augmentor: Optional[Union[AudioAugmentor, Dict[str, Dict[str, Any]]]] = None,
    ):
        super().__init__()
        self._sample_rate = sample_rate

        if augmentor is not None:
            augmentor = _process_augmentations(augmentor)

        self._featurizer = WaveformFeaturizer(
            sample_rate=self._sample_rate, int_values=int_values, augmentor=augmentor
        )

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


class TarredAudioToTextDataLayer(DataLayerNM):
    """Data Layer for general ASR tasks, where the audio files are tarred.

    Module which reads ASR labeled data. It accepts a single comma-separated JSON manifest file, as well as the
    path(s) to the tarball(s) with the wav files. Each line of the manifest should contain the information for one
    audio file, including at least the transcript and name of the audio file (doesn't have to be exact, only the
    basename must be the same).

    Valid formats for the audio_tar_filepaths argument include (1) a single string that can be brace-expanded,
    e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or (2) a list of file paths that will not be
    brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...]. See the WebDataset documentation for more information
    about accepted data and input formats.

    If using torch.distributed, the number of shards should be divisible by the number of workers to ensure an
    even split among workers. If it is not divisible, logging will give a warning but we will continue.

    Notice that a few arguments are different from the AudioToTextDataLayer; for example, shuffle (bool) has been
    replaced by shuffle_n (int).

    Additionally, please note that the len() of this DataLayer is assumed to be the length of the manifest. Be aware
    of this especially if the tarred audio is a subset of the samples represented in the manifest.

    Args:
        audio_tar_filepaths: Either a list of audio tarball filepaths, or a
            string (can be brace-expandable).
        manifest_filepath (str): Path to the manifest.
        labels (list): List of characters that can be output by the ASR model.
            For Jasper, this is the 28 character set {a-z '}. The CTC blank
            symbol is automatically added later for models using ctc.
        batch_size (int): batch size
        sample_rate (int): Target sampling rate for data. Audio files will be
            resampled to sample_rate if it is not already.
            Defaults to 16000.
        int_values (bool): Bool indicating whether the audio file is saved as
            int data or float data.
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
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        num_workers (int): See PyTorch DataLoader. Defaults to 0.
        augmentor (AudioAugmentor or dict): Optional AudioAugmentor or
            dictionary of str -> kwargs (dict) which is parsed and used
            to initialize an AudioAugmentor.
            Note: It is crucial that each individual augmentation has
            a keyword `prob`, that defines a float probability in the
            the range [0, 1] of this augmentation being applied.
            If this keyword is not present, then the augmentation is
            disabled and a warning is logged.
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
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
        audio_tar_filepaths,
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
        shuffle_n=0,
        num_workers=0,
        augmentor: Optional[Union[AudioAugmentor, Dict[str, Dict[str, Any]]]] = None,
    ):
        super().__init__()
        self._sample_rate = sample_rate

        if augmentor is not None:
            augmentor = _process_augmentations(augmentor)

        self.collection = ASRAudioText(
            manifests_files=manifest_filepath.split(','),
            parser=make_parser(labels=labels, name='en', do_normalize=normalize_transcripts),
            min_duration=min_duration,
            max_duration=max_duration,
            index_by_file_id=True,  # Must set this so the manifest lines can be indexed by file ID
        )

        self.featurizer = WaveformFeaturizer(sample_rate=self._sample_rate, int_values=int_values, augmentor=augmentor)

        self.trim = trim_silence
        self.eos_id = eos_id
        self.bos_id = bos_id

        # Used in creating a sampler (in Actions).
        self._batch_size = batch_size
        self._num_workers = num_workers
        pad_id = 0 if pad_id is None else pad_id
        self.collate_fn = partial(seq_collate_fn, token_pad_value=pad_id)

        # Check for distributed and partition shards accordingly
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            global_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

            if isinstance(audio_tar_filepaths, str):
                audio_tar_filepaths = list(braceexpand.braceexpand(audio_tar_filepaths))

            if len(audio_tar_filepaths) % world_size != 0:
                logging.warning(
                    f"Number of shards in tarred dataset ({len(audio_tar_filepaths)}) is not divisible "
                    f"by number of distributed workers ({world_size})."
                )

            begin_idx = (len(audio_tar_filepaths) // world_size) * global_rank
            end_idx = begin_idx + (len(audio_tar_filepaths) // world_size)
            audio_tar_filepaths = audio_tar_filepaths[begin_idx:end_idx]

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
        """Used to remove samples that have been filtered out by ASRAudioText already.
        Otherwise, we would get a KeyError as _build_sample attempts to find the manifest entry for a sample
        that was filtered out (e.g. for duration).
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
            audio_filestream, offset=offset, duration=manifest_entry.duration, trim=self.trim,
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

    def __len__(self):
        return len(self.collection)

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_iterator(self):
        return None


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
target_label_0, "offset": offset_in_sec_0}
        ...
        {"audio_filepath": path_to_wav_n, "duration": time_in_sec_n, "label": \
target_label_n, "offset": offset_in_sec_n}

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
        augmenter (AudioAugmentor or dict): Optional AudioAugmentor or
            dictionary of str -> kwargs (dict) which is parsed and used
            to initialize an AudioAugmentor.
            Note: It is crucial that each individual augmentation has
            a keyword `prob`, that defines a float probability in the
            the range [0, 1] of this augmentation being applied.
            If this keyword is not present, then the augmentation is
            disabled and a warning is logged.
        time_length (int): max seconds to consider in a batch # Pass this only for speaker recognition task
    """

    @property
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
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
        augmentor: Optional[Union[AudioAugmentor, Dict[str, Dict[str, Any]]]] = None,
        time_length: int = 0,
    ):
        super(AudioToSpeechLabelDataLayer, self).__init__()

        self._manifest_filepath = manifest_filepath
        self._labels = labels
        self._sample_rate = sample_rate

        if augmentor is not None:
            augmentor = _process_augmentations(augmentor)

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

        self.num_classes = self._dataset.num_commands
        logging.info("# of classes :{}".format(self.num_classes))
        self.labels = self._dataset.labels
        # Set up data loader
        if self._placement == DeviceType.AllGpu:
            logging.info("Parallelizing Datalayer.")
            sampler = torch.utils.data.distributed.DistributedSampler(self._dataset)
        else:
            sampler = None

        if time_length:
            collate_func = partial(fixed_seq_collate_fn, fixed_length=time_length * self._sample_rate)
        else:
            collate_func = partial(seq_collate_fn, token_pad_value=0)

        self._dataloader = torch.utils.data.DataLoader(
            dataset=self._dataset,
            batch_size=batch_size,
            collate_fn=collate_func,
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
