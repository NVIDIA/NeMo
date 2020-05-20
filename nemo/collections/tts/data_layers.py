# Copyright (c) 2019 NVIDIA Corporation
import torch

from .parts.datasets import AudioOnlyDataset
from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core import DeviceType
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType
from nemo.utils import logging
from nemo.utils.decorators import add_port_docs


class AudioDataLayer(DataLayerNM):
    """
    Data Layer for general speech tasks that loads only the audio.

    Module which reads speech data. It accepts comma-separated
    JSON manifest files describing the wav audio files and their metadata.
    JSON files should be of the following format::

        {"audio_filepath": path_to_wav_0, "duration": time_in_sec_0}
        ...
        {"audio_filepath": path_to_wav_n, "duration": time_in_sec_n}


    Args:
        manifest_filepath (str): path to JSON containing data.
        batch_size (int): batch sizelse.
        min_duration (float): All training files which have a duration less
            than min_duration are dropped. Note: Duration is read from the
            manifest JSON.
            Defaults to 0.1.
        max_duration (float): All training files which have a duration more
            than max_duration are dropped. Note: Duration is read from the
            manifest JSON.
            Defaults to None.
        trim_silence (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        drop_last (bool): See PyTorch DataLoader.
            Defaults to False.
        shuffle (bool): See PyTorch DataLoader.
            Defaults to True.
        num_workers (int): See PyTorch DataLoader.
            Defaults to 0.
        n_segments (int): Number of samples to load per audiofile.
            Defaults to 0 which indicates to load the whole file.
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            # "audio_signal": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # "a_sig_length": NeuralType({0: AxisType(BatchTag)}),
            "audio_signal": NeuralType(('B', 'T'), AudioSignal(freq=self.sample_rate)),
            "a_sig_length": NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self,
        manifest_filepath,
        batch_size,
        sample_rate,
        min_duration=0.1,
        max_duration=None,
        trim_silence=False,
        drop_last=False,
        shuffle=True,
        num_workers=0,
        n_segments=0,
    ):
        super().__init__()
        self.sample_rate = sample_rate

        self._dataset = AudioOnlyDataset(
            manifest_filepath=manifest_filepath,
            max_duration=max_duration,
            min_duration=min_duration,
            trim=trim_silence,
            n_segments=n_segments,
        )

        sampler = None
        if self._placement == DeviceType.AllGpu:
            logging.info('Parallelizing DATALAYER')
            sampler = torch.utils.data.distributed.DistributedSampler(self._dataset)

        self._dataloader = torch.utils.data.DataLoader(
            dataset=self._dataset,
            batch_size=batch_size,
            collate_fn=self._dataset.AudioCollateFunc,
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
