import random

import torch
import torch.nn as nn


class SpecAugment(nn.Module):
    """
    Zeroes out(cuts) random continuous horisontal or
    vertical segments of the spectrogram as described in
    SpecAugment (https://arxiv.org/abs/1904.08779).

    params:
    freq_masks - how many frequency segments should be cut
    time_masks - how many time segments should be cut
    freq_width - maximum number of frequencies to be cut in one segment
    time_width - maximum number of time steps to be cut in one segment.
        Can be a positive integer or a float value in the range [0, 1].
        If positive integer value, defines maximum number of time steps
        to be cut in one segment.
        If a float value, defines maximum percentage of timesteps that
        are cut adaptively.
    """

    def __init__(
        self, freq_masks=0, time_masks=0, freq_width=10, time_width=10, rng=None,
    ):
        super(SpecAugment, self).__init__()

        self._rng = random.Random() if rng is None else rng

        self.freq_masks = freq_masks
        self.time_masks = time_masks

        self.freq_width = freq_width
        self.time_width = time_width

        if isinstance(time_width, int):
            self.adaptive_temporal_width = False
        else:
            if time_width > 1.0 or time_width < 0.0:
                raise ValueError('If `time_width` is a float value, must be in range [0, 1]')

            self.adaptive_temporal_width = True

    @torch.no_grad()
    def forward(self, x):
        sh = x.shape

        if self.adaptive_temporal_width:
            time_width = max(1, int(sh[2] * self.time_width))
        else:
            time_width = self.time_width

        mask = torch.zeros(x.shape).byte()

        for idx in range(sh[0]):
            for i in range(self.freq_masks):
                x_left = int(self._rng.uniform(0, sh[1] - self.freq_width))

                w = int(self._rng.uniform(0, self.freq_width))

                mask[idx, x_left : x_left + w, :] = 1

            for i in range(self.time_masks):
                y_left = int(self._rng.uniform(0, sh[2] - time_width))

                w = int(self._rng.uniform(0, time_width))

                mask[idx, :, y_left : y_left + w] = 1

        x = x.masked_fill(mask.type(torch.bool).to(device=x.device), 0)

        return x


class SpecCutout(nn.Module):
    """
    Zeroes out(cuts) random rectangles in the spectrogram
    as described in (https://arxiv.org/abs/1708.04552).

    params:
    rect_masks - how many rectangular masks should be cut
    rect_freq - maximum size of cut rectangles along the frequency dimension
    rect_time - maximum size of cut rectangles along the time dimension
    """

    def __init__(self, rect_masks=0, rect_time=5, rect_freq=20, rng=None):
        super(SpecCutout, self).__init__()

        self._rng = random.Random() if rng is None else rng

        self.rect_masks = rect_masks
        self.rect_time = rect_time
        self.rect_freq = rect_freq

    @torch.no_grad()
    def forward(self, x):
        sh = x.shape

        mask = torch.zeros(x.shape).byte()

        for idx in range(sh[0]):
            for i in range(self.rect_masks):
                rect_x = int(self._rng.uniform(0, sh[1] - self.rect_freq))
                rect_y = int(self._rng.uniform(0, sh[2] - self.rect_time))

                w_x = int(self._rng.uniform(0, self.rect_time))
                w_y = int(self._rng.uniform(0, self.rect_freq))

                mask[idx, rect_x : rect_x + w_x, rect_y : rect_y + w_y] = 1

        x = x.masked_fill(mask.type(torch.bool).to(device=x.device), 0)

        return x
