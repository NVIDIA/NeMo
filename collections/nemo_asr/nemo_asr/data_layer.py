# Copyright (c) 2019 NVIDIA Corporation
"""
This package contains Neural Modules responsible for ASR-related data
processing
"""
import torch
from apex import amp

from nemo.backends.pytorch.nm import DataLayerNM, TrainableNM, NonTrainableNM
from nemo.core import Optimization, DeviceType
from nemo.core.neural_types import *
from .parts.dataset import AudioDataset, seq_collate_fn
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
            eos_id=None,
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
        DataLayerNM.__init__(self, **kwargs)

        self._featurizer = WaveformFeaturizer(
            sample_rate=sample_rate, int_values=int_values, augmentor=None)
        self._dataset = AudioDataset(
            manifest_filepath=manifest_filepath,
            labels=labels,
            featurizer=self._featurizer, max_duration=max_duration,
            min_duration=min_duration, normalize=normalize_transcripts,
            trim=trim_silence, logger=self._logger,
            eos_id=eos_id, load_audio=load_audio
        )

        if self._placement == DeviceType.AllGpu:
            self._logger.info('Parallelizing DATALAYER')
            sampler = torch.utils.data.distributed.DistributedSampler(
                self._dataset)
        else:
            sampler = None

        self._dataloader = torch.utils.data.DataLoader(
            dataset=self._dataset,
            batch_size=batch_size,
            collate_fn=seq_collate_fn,
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
            **kwargs
    ):
        if "fbank" not in feat_type:
            raise NotImplementedError("AudioPreprocessing currently only "
                                      "accepts 'fbank' or 'logfbank' as "
                                      "feat_type")
        TrainableNM.__init__(self, **kwargs)

        self.featurizer = FilterbankFeatures(
            sample_rate=sample_rate,
            window_size=window_size,
            window_stride=window_stride,
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
            logger=self._logger
        )
        # _pre_procesing_config = self.local_parameters
        # self.featurizer = FeatureFactory.from_config(_pre_procesing_config)
        self.featurizer.to(self._device)

        self.disable_casts = (self._opt_level == Optimization.mxprO1)

    def forward(self, input_signal, length):
        length.requires_grad_(False)
        if self.disable_casts:
            with amp.disable_casts():
                if input_signal.dim() == 2:
                    processed_signal = self.featurizer(
                        input_signal.to(torch.float), length)
                    processed_length = self.featurizer.get_seq_len(length)
        else:
            if input_signal.dim() == 2:
                processed_signal = self.featurizer(input_signal, length)
                processed_length = self.featurizer.get_seq_len(length)
        return processed_signal, processed_length


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
