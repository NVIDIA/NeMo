# Copyright (C) NVIDIA CORPORATION. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.****
"""
This file contains neural modules responsible for preprocessing audio data.
"""
__all__ = [#'AudioPreprocessing',
           'MultiplyBatch',
           'SpectrogramAugmentation']

import torch

from nemo.backends.pytorch import NonTrainableNM
from nemo.core.neural_types import *


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



