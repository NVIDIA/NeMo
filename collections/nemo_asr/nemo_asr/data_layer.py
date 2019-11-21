# Copyright (c) 2019 NVIDIA Corporation
"""
This package contains Neural Modules responsible for ASR-related data
processing
"""
__all__ = ['AudioToTextDataLayer',
           'AudioPreprocessing',
           'MultiplyBatch',
           'SpectrogramAugmentation',
           'KaldiFeatureDataLayer',
           'TranscriptDataLayer']

from functools import partial
import torch
try:
    from apex import amp
except AttributeError:
    print("Unable to import APEX. Mixed precision and distributed training "
          "will not work.")

from nemo.backends.pytorch import DataLayerNM, TrainableNM, NonTrainableNM
from nemo.core import Optimization, DeviceType
from nemo.core.neural_types import *
from nemo.utils.misc import pad_to as nemo_pad_to
from .parts.dataset import (
        AudioDataset, seq_collate_fn, KaldiFeatureDataset, TranscriptDataset)
from .parts.features import FilterbankFeatures, WaveformFeaturizer
from .parts.spectr_augment import SpecAugment, SpecCutout


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
        eos_id (str): Dataset parameter.
            End of string symbol used for seq2seq models.
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

    @staticmethod
    def create_ports():
        input_ports = {}
        output_ports = {
            "audio_signal": NeuralType({0: AxisType(BatchTag),
                                        1: AxisType(TimeTag)}),

            "a_sig_length": NeuralType({0: AxisType(BatchTag)}),

            "transcripts": NeuralType({0: AxisType(BatchTag),
                                       1: AxisType(TimeTag)}),

            "transcript_length": NeuralType({0: AxisType(BatchTag)})
        }
        return input_ports, output_ports

    def __init__(
            self, *,
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
            # perturb_config=None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self._featurizer = WaveformFeaturizer(
            sample_rate=sample_rate, int_values=int_values, augmentor=None)

        # Set up dataset
        dataset_params = {'manifest_filepath': manifest_filepath,
                          'labels': labels,
                          'featurizer': self._featurizer,
                          'max_duration': max_duration,
                          'min_duration': min_duration,
                          'normalize': normalize_transcripts,
                          'trim': trim_silence,
                          'bos_id': bos_id,
                          'eos_id': eos_id,
                          'logger': self._logger,
                          'load_audio': load_audio}

        self._dataset = AudioDataset(**dataset_params)

        # Set up data loader
        if self._placement == DeviceType.AllGpu:
            self._logger.info('Parallelizing DATALAYER')
            sampler = torch.utils.data.distributed.DistributedSampler(
                self._dataset)
        else:
            sampler = None

        pad_id = 0 if pad_id is None else pad_id
        self._dataloader = torch.utils.data.DataLoader(
            dataset=self._dataset,
            batch_size=batch_size,
            collate_fn=partial(seq_collate_fn, token_pad_value=pad_id),
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


class AudioPreprocessing(TrainableNM):
    """
    Neural Module that does batch processing of audio files and converts them
    to spectrogram representations

    Args:
        sample_rate (int): Sample rate of the input audio data.
            Defaults to 16000
        window_size (float): Size of window for fft in seconds
            Defaults to 0.02
        window_stride (float): Stride of window for fft in seconds
            Defaults to 0.01
        n_window_size (int): Size of window for fft in samples
            Defaults to None. Use one of window_size or n_window_size.
        n_window_stride (int): Stride of window for fft in samples
            Defaults to None. Use one of window_stride or n_window_stride.
        window (str): Windowing function for fft. can be one of ['hann',
            'hamming', 'blackman', 'bartlett']
            Defaults to "hann"
        normalize (str): Can be one of ['per_feature', 'all_features']; all
            other options disable feature normalization. 'all_features'
            normalizes the entire spectrogram to be mean 0 with std 1.
            'pre_features' normalizes per channel / freq instead.
            Defaults to "per_feature"
        n_fft (int): Length of FT window. If None, it uses the smallest power
            of 2 that is larger than n_window_size.
            Defaults to None
        preemph (float): Amount of pre emphasis to add to audio. Can be
            disabled by passing None.
            Defaults to 0.97
        features (int): Number of mel spectrogram freq bins to output.
            Defaults to 64
        lowfreq (int): Lower bound on mel basis in Hz.
            Defaults to 0
        highfreq  (int): Lower bound on mel basis in Hz.
            Defaults to None
        feat_type (str): Can be one of ['logfbank', 'fbank'].
            Defaults to "logfbank"
        dither (float): Amount of white-noise dithering.
            Defaults to 1e-5
        pad_to (int): Ensures that the output size of the time dimension is
            a multiple of pad_to.
            Defaults to 16
        frame_splicing (int):
            Defaults to 1
        stft_conv (bool): If True, uses pytorch_stft and convolutions. If
            False, uses torch.stft.
            Defaults to False
        pad_value (float): The value that shorter mels are padded with.
            Defaults to 0
        mag_power (float): The power that the linear spectrogram is raised to
            prior to multiplication with mel basis.
            Defaults to 2 for a power spec
    """
    @staticmethod
    def create_ports():
        input_ports = {
            "input_signal": NeuralType({0: AxisType(BatchTag),
                                        1: AxisType(TimeTag)}),

            "length": NeuralType({0: AxisType(BatchTag)}),
        }

        output_ports = {
            "processed_signal": NeuralType({0: AxisType(BatchTag),
                                            1: AxisType(SpectrogramSignalTag),
                                            2: AxisType(ProcessedTimeTag)}),

            "processed_length": NeuralType({0: AxisType(BatchTag)})
        }
        return input_ports, output_ports

    def __init__(
            self, *,
            sample_rate=16000,
            window_size=0.02,
            window_stride=0.01,
            n_window_size=None,
            n_window_stride=None,
            window="hann",
            normalize="per_feature",
            n_fft=None,
            preemph=0.97,
            features=64,
            lowfreq=0,
            highfreq=None,
            feat_type="logfbank",
            dither=1e-5,
            pad_to=16,
            frame_splicing=1,
            stft_conv=False,
            pad_value=0,
            mag_power=2.,
            **kwargs
    ):
        if "fbank" not in feat_type:
            raise NotImplementedError("AudioPreprocessing currently only "
                                      "accepts 'fbank' or 'logfbank' as "
                                      "feat_type")
        if window_size and n_window_size:
            raise ValueError(f"{self} received both window_size and "
                             f"n_window_size. Only one should be specified.")
        if window_stride and n_window_stride:
            raise ValueError(f"{self} received both window_stride and "
                             f"n_window_stride. Only one should be specified.")
        TrainableNM.__init__(self, **kwargs)

        if window_size:
            n_window_size = int(window_size * sample_rate)
        if window_stride:
            n_window_stride = int(window_stride * sample_rate)

        self.featurizer = FilterbankFeatures(
            sample_rate=sample_rate,
            n_window_size=n_window_size,
            n_window_stride=n_window_stride,
            window=window,
            normalize=normalize,
            n_fft=n_fft,
            preemph=preemph,
            nfilt=features,
            lowfreq=lowfreq,
            highfreq=highfreq,
            dither=dither,
            pad_to=pad_to,
            frame_splicing=frame_splicing,
            stft_conv=stft_conv,
            pad_value=pad_value,
            mag_power=mag_power,
            logger=self._logger
        )
        self.featurizer.to(self._device)

        self.disable_casts = (self._opt_level == Optimization.mxprO1)

    def forward(self, input_signal, length):
        length.requires_grad_(False)
        if self.disable_casts:
            with amp.disable_casts():
                if input_signal.dim() == 2:
                    processed_signal = self.featurizer(
                        input_signal.to(torch.float), length)
                    processed_length = self.featurizer.get_seq_len(
                        length.float())
        else:
            if input_signal.dim() == 2:
                processed_signal = self.featurizer(input_signal, length)
                processed_length = self.featurizer.get_seq_len(length.float())
        return processed_signal, processed_length

    @property
    def filter_banks(self):
        return self.featurizer.filter_banks


class SpectrogramAugmentation(NonTrainableNM):
    """
    Performs time and freq cuts in one of two ways.

    SpecAugment zeroes out vertical and horizontal sections as described in
    SpecAugment (https://arxiv.org/abs/1904.08779). Arguments for use with
    SpecAugment are `freq_masks`, `time_masks`, `freq_width`, and `time_width`.

    SpecCutout zeroes out rectangulars as described in Cutout
    (https://arxiv.org/abs/1708.04552). Arguments for use with Cutout are
    `rect_masks`, `rect_freq`, and `rect_time`.

    Args:
        freq_masks (int): how many frequency segments should be cut.
            Defaults to 0.
        time_masks (int): how many time segments should be cut
            Defaults to 0.
        freq_width (int): maximum number of frequencies to be cut in one
            segment.
            Defaults to 10.
        time_width (int): maximum number of time steps to be cut in one
            segment
            Defaults to 10.
        rect_masks (int): how many rectangular masks should be cut
            Defaults to 0.
        rect_freq (int): maximum size of cut rectangles along the frequency
            dimension
            Defaults to 5.
        rect_time (int): maximum size of cut rectangles along the time
            dimension
            Defaults to 25.
    """
    @staticmethod
    def create_ports():
        input_ports = {
            "input_spec": NeuralType({0: AxisType(BatchTag),
                                      1: AxisType(SpectrogramSignalTag),
                                      2: AxisType(TimeTag)})
        }

        output_ports = {
            "augmented_spec": NeuralType({0: AxisType(BatchTag),
                                          1: AxisType(SpectrogramSignalTag),
                                          2: AxisType(ProcessedTimeTag)})
        }
        return input_ports, output_ports

    def __init__(
            self, *,
            freq_masks=0,
            time_masks=0,
            freq_width=10,
            time_width=10,
            rect_masks=0,
            rect_time=5,
            rect_freq=20,
            rng=None,
            **kwargs
    ):
        NonTrainableNM.__init__(self, **kwargs)

        if rect_masks > 0:
            self.spec_cutout = SpecCutout(
                rect_masks=rect_masks,
                rect_time=rect_time,
                rect_freq=rect_freq,
                rng=rng
            )
            self.spec_cutout.to(self._device)
        else:
            self.spec_cutout = lambda x: x

        if freq_masks + time_masks > 0:
            self.spec_augment = SpecAugment(
                freq_masks=freq_masks,
                time_masks=time_masks,
                freq_width=freq_width,
                time_width=time_width,
                rng=rng
            )
            self.spec_augment.to(self._device)
        else:
            self.spec_augment = lambda x: x

    def forward(self, input_spec):
        augmented_spec = self.spec_cutout(input_spec)
        augmented_spec = self.spec_augment(augmented_spec)
        return augmented_spec


class KaldiFeatureDataLayer(DataLayerNM):
    """Data layer for reading generic Kaldi-formatted data.

    Module that reads ASR labeled data that is in a Kaldi-compatible format.
    It assumes that you have a directory that contains:

    - feats.scp: A mapping from utterance IDs to .ark files that
            contain the corresponding MFCC (or other format) data
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

    @staticmethod
    def create_ports():
        input_ports = {}
        output_ports = {
            "processed_signal": NeuralType({0: AxisType(BatchTag),
                                            1: AxisType(SpectrogramSignalTag),
                                            2: AxisType(ProcessedTimeTag)}),

            "processed_length": NeuralType({0: AxisType(BatchTag)}),

            "transcripts": NeuralType({0: AxisType(BatchTag),
                                       1: AxisType(TimeTag)}),

            "transcript_length": NeuralType({0: AxisType(BatchTag)})
        }
        return input_ports, output_ports

    def __init__(
            self, *,
            kaldi_dir,
            labels,
            batch_size,
            min_duration=None,
            max_duration=None,
            normalize_transcripts=True,
            drop_last=False,
            shuffle=True,
            num_workers=0,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Set up dataset
        dataset_params = {'kaldi_dir': kaldi_dir,
                          'labels': labels,
                          'min_duration': min_duration,
                          'max_duration': max_duration,
                          'normalize': normalize_transcripts,
                          'logger': self._logger}
        self._dataset = KaldiFeatureDataset(**dataset_params)

        # Set up data loader
        if self._placement == DeviceType.AllGpu:
            self._logger.info('Parallelizing DATALAYER')
            sampler = torch.utils.data.distributed.DistributedSampler(
                self._dataset)
        else:
            sampler = None

        self._dataloader = torch.utils.data.DataLoader(
            dataset=self._dataset,
            batch_size=batch_size,
            collate_fn=self._collate_fn,
            drop_last=drop_last,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers
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
                pad = (0, max_feat_len - feat_len)
                feat = torch.nn.functional.pad(feat, pad)
            features.append(feat)

            tkns_len = tkns_len.item()
            if tkns_len < max_tokens_len:
                pad = (0, max_tokens_len - tkns_len)
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


class MultiplyBatch(NonTrainableNM):
    """
    Augmentation that repeats each element in a batch.
    Other augmentations can be applied afterwards.

    Args:
        mult_batch (int): number of repeats
    """
    @staticmethod
    def create_ports():
        input_ports = {
            "in_x": NeuralType({0: AxisType(BatchTag),
                                1: AxisType(SpectrogramSignalTag),
                                2: AxisType(TimeTag)}),

            "in_x_len": NeuralType({0: AxisType(BatchTag)}),

            "in_y": NeuralType({0: AxisType(BatchTag),
                                1: AxisType(TimeTag)}),

            "in_y_len": NeuralType({0: AxisType(BatchTag)})
        }

        output_ports = {
            "out_x": NeuralType({0: AxisType(BatchTag),
                                 1: AxisType(SpectrogramSignalTag),
                                 2: AxisType(TimeTag)}),

            "out_x_len": NeuralType({0: AxisType(BatchTag)}),

            "out_y": NeuralType({0: AxisType(BatchTag),
                                 1: AxisType(TimeTag)}),

            "out_y_len": NeuralType({0: AxisType(BatchTag)})
        }
        return input_ports, output_ports

    def __init__(self, *, mult_batch=1, **kwargs):
        NonTrainableNM.__init__(self, **kwargs)
        self.mult = mult_batch

    @torch.no_grad()
    def forward(self, in_x, in_x_len, in_y, in_y_len):
        out_x = in_x.repeat(self.mult, 1, 1)
        out_y = in_y.repeat(self.mult, 1)
        out_x_len = in_x_len.repeat(self.mult)
        out_y_len = in_y_len.repeat(self.mult)

        return out_x, out_x_len, out_y, out_y_len


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

    @staticmethod
    def create_ports():
        input_ports = {}
        output_ports = {
            'texts': NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),

            "texts_length": NeuralType({0: AxisType(BatchTag)})
        }
        return input_ports, output_ports

    def __init__(self,
                 path,
                 labels,
                 batch_size,
                 bos_id=None,
                 eos_id=None,
                 pad_id=None,
                 drop_last=False,
                 num_workers=0,
                 **kwargs):
        super().__init__(**kwargs)

        # Set up dataset
        dataset_params = {'path': path,
                          'labels': labels,
                          'bos_id': bos_id,
                          'eos_id': eos_id}

        self._dataset = TranscriptDataset(**dataset_params)

        # Set up data loader
        if self._placement == DeviceType.AllGpu:
            sampler = torch.utils.data.distributed.DistributedSampler(
                    self._dataset)
        else:
            sampler = None

        pad_id = 0 if pad_id is None else pad_id

        # noinspection PyTypeChecker
        self._dataloader = torch.utils.data.DataLoader(
            dataset=self._dataset,
            batch_size=batch_size,
            collate_fn=partial(self._collate_fn, pad_id=pad_id, pad8=True),
            drop_last=drop_last,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=num_workers
        )

    @staticmethod
    def _collate_fn(batch, pad_id, pad8=False):
        texts_list, texts_len = zip(*batch)
        max_len = max(texts_len)
        if pad8:
            max_len = nemo_pad_to(max_len, 8)

        texts = torch.empty(len(texts_list), max_len,
                            dtype=torch.long)
        texts.fill_(pad_id)

        for i, s in enumerate(texts_list):
            texts[i].narrow(0, 0, s.size(0)).copy_(s)

        if len(texts.shape) != 2:
            raise ValueError(
                f"Texts in collate function have shape {texts.shape},"
                f" should have 2 dimensions."
            )

        return texts, torch.stack(texts_len)

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self._dataloader
