# Taken straight from Patter https://github.com/ryanleary/patter
# TODO: review, and copyright and fix/add comments
import math
import librosa
import torch
import torch.nn as nn
from .perturb import AudioAugmentor
from .segment import AudioSegment
from torch_stft import STFT

CONSTANT = 1e-5


def normalize_batch(x, seq_len, normalize_type):
    if normalize_type == "per_feature":
        x_mean = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype,
                             device=x.device)
        x_std = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype,
                            device=x.device)
        for i in range(x.shape[0]):
            x_mean[i, :] = x[i, :, :seq_len[i]].mean(dim=1)
            x_std[i, :] = x[i, :, :seq_len[i]].std(dim=1)
        # make sure x_std is not zero
        x_std += CONSTANT
        return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)
    elif normalize_type == "all_features":
        x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            x_mean[i] = x[i, :, :seq_len[i].item()].mean()
            x_std[i] = x[i, :, :seq_len[i].item()].std()
        # make sure x_std is not zero
        x_std += CONSTANT
        return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1)
    else:
        return x


def splice_frames(x, frame_splicing):
    """ Stacks frames together across feature dim

    input is batch_size, feature_dim, num_frames
    output is batch_size, feature_dim*frame_splicing, num_frames

    """
    seq = [x]
    for n in range(1, frame_splicing):
        seq.append(torch.cat([x[:, :, :n], x[:, :, n:]], dim=2))
    return torch.cat(seq, dim=1)


class WaveformFeaturizer(object):
    def __init__(self, sample_rate=16000, int_values=False, augmentor=None):
        self.augmentor = augmentor if augmentor is not None else \
            AudioAugmentor()
        self.sample_rate = sample_rate
        self.int_values = int_values

    def max_augmentation_length(self, length):
        return self.augmentor.max_augmentation_length(length)

    def process(self, file_path, offset=0, duration=0, trim=False):
        audio = AudioSegment.from_file(
            file_path,
            target_sr=self.sample_rate,
            int_values=self.int_values,
            offset=offset, duration=duration, trim=trim)
        return self.process_segment(audio)

    def process_segment(self, audio_segment):
        self.augmentor.perturb(audio_segment)
        return torch.tensor(audio_segment.samples, dtype=torch.float)

    @classmethod
    def from_config(cls, input_config, perturbation_configs=None):
        if perturbation_configs is not None:
            aa = AudioAugmentor.from_config(perturbation_configs)
        else:
            aa = None

        sample_rate = input_config.get("sample_rate", 16000)
        int_values = input_config.get("int_values", False)

        return cls(sample_rate=sample_rate, int_values=int_values,
                   augmentor=aa)


class FeaturizerFactory(object):
    def __init__(self):
        pass

    @classmethod
    def from_config(cls, input_cfg, perturbation_configs=None):
        return WaveformFeaturizer.from_config(
            input_cfg,
            perturbation_configs=perturbation_configs)


class SpectrogramFeatures(nn.Module):
    def __init__(self, sample_rate=8000, window_size=0.02, window_stride=0.01,
                 n_fft=None,
                 window="hamming", normalize=True, log=True, center=True,
                 dither=CONSTANT, pad_to=8, max_duration=16.7,
                 frame_splicing=1):
        super(SpectrogramFeatures, self).__init__()
        torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'none': None,
        }
        self.win_length = int(sample_rate * window_size)
        self.hop_length = int(sample_rate * window_stride)
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))

        window_fn = torch_windows.get(window, None)
        window_tensor = window_fn(self.win_length,
                                  periodic=False) if window_fn else None
        self.window = window_tensor

        self.normalize = normalize
        self.log = log
        self.center = center
        self.dither = dither
        self.pad_to = pad_to
        self.frame_splicing = frame_splicing

    def get_seq_len(self, seq_len):
        return torch.ceil(seq_len.to(dtype=torch.float) / self.hop_length).to(
            dtype=torch.long)

    @torch.no_grad()
    def forward(self, x, seq_len):
        dtype = x.dtype
        x = x.to(torch.float)

        seq_len = self.get_seq_len(seq_len)

        # dither
        if self.dither > 0:
            x += self.dither * torch.randn_like(x)

        # do preemphasis
        if hasattr(self, 'preemph') and self.preemph is not None:
            x = torch.cat(
                (x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]),
                dim=1)

        # get spectrogram
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                       win_length=self.win_length, center=self.center,
                       window=self.window.to(torch.float))
        x = torch.sqrt(x.pow(2).sum(-1))

        # log features if required
        if self.log:
            x = torch.log(x + 1e-20)

        # frame splicing if required
        if self.frame_splicing > 1:
            x = splice_frames(x, self.frame_splicing)

        # normalize if required
        if self.normalize:
            x = normalize_batch(x, seq_len, normalize_type=self.normalize)

        # mask to zero any values beyond seq_len in batch, pad to multiple of
        # `pad_to` (for efficiency)
        max_len = x.size(-1)
        mask = torch.arange(max_len).to(seq_len.device)
        mask = mask.expand(x.size(0), max_len) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).to(device=x.device), 0)
        del mask
        if self.pad_to > 0:
            pad_amt = x.size(-1) % self.pad_to
            if pad_amt != 0:
                x = nn.functional.pad(x, (0, self.pad_to - pad_amt))

        return x.to(dtype)

    @classmethod
    def from_config(cls, cfg, log=False):
        return cls(sample_rate=cfg['sample_rate'],
                   window_size=cfg['window_size'],
                   window_stride=cfg['window_stride'],
                   n_fft=cfg['n_fft'], window=cfg['window'],
                   normalize=cfg['normalize'],
                   dither=cfg.get('dither', 1e-5), pad_to=cfg.get("pad_to", 0),
                   frame_splicing=cfg.get("frame_splicing", 1), log=log)


class FilterbankFeatures(nn.Module):
    def __init__(self, sample_rate=8000, window_size=0.02, window_stride=0.01,
                 window="hann", normalize="per_feature", n_fft=None,
                 preemph=0.97,
                 nfilt=64, lowfreq=0, highfreq=None, log=True, dither=CONSTANT,
                 pad_to=16, max_duration=16.7,
                 frame_splicing=1, stft_conv=False, logger=None):
        super(FilterbankFeatures, self).__init__()
        if logger:
            logger.info(f"PADDING: {pad_to}")
        else:
            print(f"PADDING: {pad_to}")

        self.win_length = int(sample_rate * window_size)
        self.hop_length = int(sample_rate * window_stride)
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.stft_conv = stft_conv

        if stft_conv:
            if logger:
                logger.info("STFT using conv")
            else:
                print("STFT using conv")

            # Create helper class to patch forward func for use with AMP
            class STFTPatch(STFT):
                def __init__(self, *params, **kw_params):
                    super(STFTPatch, self).__init__(*params, **kw_params)

                def forward(self, input_data):
                    return super(STFTPatch, self).transform(input_data)[0]

            self.stft = STFTPatch(self.n_fft, self.hop_length,
                                  self.win_length, window)

        else:
            print("STFT using torch")
            torch_windows = {
                'hann': torch.hann_window,
                'hamming': torch.hamming_window,
                'blackman': torch.blackman_window,
                'bartlett': torch.bartlett_window,
                'none': None,
            }
            window_fn = torch_windows.get(window, None)
            window_tensor = window_fn(self.win_length,
                                      periodic=False) if window_fn else None
            self.register_buffer("window", window_tensor)
            self.stft = lambda x: torch.stft(
                            x, n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            win_length=self.win_length,
                            center=True,
                            window=self.window.to(dtype=torch.float))

        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.nfilt = nfilt
        self.preemph = preemph
        self.pad_to = pad_to
        highfreq = highfreq or sample_rate / 2

        filterbanks = torch.tensor(
            librosa.filters.mel(sample_rate, self.n_fft, n_mels=nfilt,
                                fmin=lowfreq, fmax=highfreq),
            dtype=torch.float).unsqueeze(0)
        # self.fb = filterbanks
        # self.window = window_tensor
        self.register_buffer("fb", filterbanks)

        # Calculate maximum sequence length
        max_length = 1 + math.ceil(
            (max_duration * sample_rate - self.win_length) / self.hop_length
        )
        max_pad = 16 - (max_length % 16)
        self.max_length = max_length + max_pad

    def get_seq_len(self, seq_len):
        return torch.ceil(seq_len.to(dtype=torch.float) / self.hop_length).to(
            dtype=torch.long)

    @torch.no_grad()
    def forward(self, x, seq_len):
        seq_len = self.get_seq_len(seq_len)

        # dither
        if self.dither > 0:
            x += self.dither * torch.randn_like(x)

        # do preemphasis
        if self.preemph is not None:
            x = torch.cat(
                (x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]),
                dim=1)

        x = self.stft(x)

        # get power spectrum
        x = x.pow(2)
        if not self.stft_conv:
            x = x.sum(-1)

        # dot with filterbank energies
        x = torch.matmul(self.fb.to(x.dtype), x)

        # log features if required
        if self.log:
            x = torch.log(x + 2**-24)

        # frame splicing if required
        if self.frame_splicing > 1:
            x = splice_frames(x, self.frame_splicing)

        # normalize if required
        if self.normalize:
            x = normalize_batch(x, seq_len, normalize_type=self.normalize)

        # mask to zero any values beyond seq_len in batch, pad to multiple of
        # `pad_to` (for efficiency)
        max_len = x.size(-1)
        mask = torch.arange(max_len).to(x.device)
        mask = mask.expand(x.size(0), max_len) >= seq_len.unsqueeze(1)
        x = x.masked_fill(
            mask.unsqueeze(1).type(torch.bool).to(device=x.device), 0
        )
        del mask
        pad_to = self.pad_to
        if not self.training:
            pad_to = 16
        if pad_to == "max":
            x = nn.functional.pad(x, (0, self.max_length - x.size(-1)))
        elif pad_to > 0:
            pad_amt = x.size(-1) % pad_to
            if pad_amt != 0:
                x = nn.functional.pad(x, (0, pad_to - pad_amt))

        return x

    @classmethod
    def from_config(cls, cfg, log=False):
        return cls(sample_rate=cfg.get('sample_rate', 8000),
                   window_size=cfg.get('window_size', 0.02),
                   window_stride=cfg.get('window_stride', 0.01),
                   n_fft=cfg.get('n_fft', None),
                   nfilt=cfg.get('features', 64),
                   window=cfg.get('window', "hann"),
                   normalize=cfg.get('normalize', "per_feature"),
                   dither=cfg.get('dither', CONSTANT),
                   pad_to=cfg.get("pad_to", 16),
                   frame_splicing=cfg.get("frame_splicing", 1),
                   log=log,
                   stft_conv=cfg.get("stft_conv", False))


class FeatureFactory(object):
    featurizers = {
        "logfbank": FilterbankFeatures,
        "fbank": FilterbankFeatures,
        "stft": SpectrogramFeatures,
        "logspect": SpectrogramFeatures,
        "logstft": SpectrogramFeatures
    }

    def __init__(self):
        pass

    @classmethod
    def from_config(cls, cfg):
        feat_type = cfg.get('feat_type', "logspect")
        featurizer = cls.featurizers[feat_type]
        return featurizer.from_config(cfg, log="log" in feat_type)
