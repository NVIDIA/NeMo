# Copyright (c) 2019 NVIDIA Corporation
import torch
try:
    from apex import amp
except AttributeError:
    print("Unable to import APEX. Mixed precision and distributed training "
          "will not work.")

from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core import DeviceType
from nemo.core.neural_types import *
from .parts.datasets import AudioOnlyDataset


class AudioDataLayer(DataLayerNM):
    """TODO:Fix docstring
    Data Layer for general ASR tasks.

    Module which reads ASR labeled data. It accepts comma-separated
    JSON manifest files describing the correspondence between wav audio files
    and their transcripts. JSON files should be of the following format::

        {"audio_filepath": path_to_wav_0, "duration": time_in_sec_0, "text": \
transcript_0}
        ...
        {"audio_filepath": path_to_wav_n, "duration": time_in_sec_n, "text": \
transcript_n}


    Args:
        manifest_filepath (str): path to JSON containing data.
        labels (list): list of characters that can be output by the ASR model.
            For Jasper, this is the 28 character set {a-z '}. The CTC blank
            symbol is automatically added later for models using ctc.
        batch_size (int): batch size
        sample_rate (int): Target sampling rate for data. Audio files will be
            resampled to sample_rate if it is not already.
            Defaults to 16000.
        int_values (bool): Bool indicating whether the audio file is saved as
            int data or float data.
            Defaults to False.
        eos_id (str): End of string symbol used for seq2seq models.
            Defaults to None.
        min_duration (float): All training files which have a duration less
            than min_duration are dropped. Note: Duration is read from the
            manifest JSON.
            Defaults to 0.1.
        max_duration (float): All training files which have a duration more
            than max_duration are dropped. Note: Duration is read from the
            manifest JSON.
            Defaults to None.
        normalize_transcripts (bool): Whether to use automatic text cleaning.
            It is highly recommended to manually clean text ffor best results.
            Defaults to True.
        trim_silence (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        load_audio (bool): Controls whether the dataloader loads the audio
            signal and transcript or just the transcript.
            Defaults to True.
        drop_last (bool): See PyTorch DataLoader.
            Defaults to False.
        shuffle (bool): See PyTorch DataLoader.
            Defaults to True.
        num_workers (int): See PyTorch DataLoader.
            Defaults to 0.
        perturb_config (dict): Currently disabled.
    """

    @staticmethod
    def create_ports():
        input_ports = {}
        output_ports = {
            "audio_signal": NeuralType({0: AxisType(BatchTag),
                                        1: AxisType(TimeTag)}),

            "a_sig_length": NeuralType({0: AxisType(BatchTag)}),
        }
        return input_ports, output_ports

    def __init__(
            self, *,
            manifest_filepath,
            batch_size,
            min_duration=0.1,
            max_duration=None,
            trim_silence=False,
            drop_last=False,
            shuffle=True,
            num_workers=0,
            n_segments=0,
            **kwargs
    ):
        DataLayerNM.__init__(self, **kwargs)

        self._dataset = AudioOnlyDataset(
            manifest_filepath=manifest_filepath,
            max_duration=max_duration,
            min_duration=min_duration,
            trim=trim_silence,
            logger=self._logger,
            n_segments=n_segments
        )

        sampler = None
        if self._placement == DeviceType.AllGpu:
            self._logger.info('Parallelizing DATALAYER')
            sampler = torch.utils.data.distributed.DistributedSampler(
                self._dataset)

        self._dataloader = torch.utils.data.DataLoader(
            dataset=self._dataset,
            batch_size=batch_size,
            collate_fn=self._dataset.AudioCollateFunc,
            drop_last=drop_last,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers
        )

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self._dataloader
