# Copyright 2020 NVIDIA. All Rights Reserved.
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
#
# Inspired by: https://github.com/r9y9/wavenet_vocoder
# Copyright (c) 2017: Ryuichi Yamamoto.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

import numpy as np
import torch
from torch.nn import functional as F


def dmld_loss(y_pred, y_true, num_classes):
    """Discretized mixture of logistic distributions loss

    https://github.com/r9y9/wavenet_vocoder/blob/master/wavenet_vocoder/mixture.py
    https://arxiv.org/pdf/1701.05517.pdf

    Args:
        y_pred (Tensor): Predicted output (B x T x C)
        y_true (Tensor): Target (B x T).
        num_classes (int): Number of classes

    Returns
        Tensor: loss

    """

    def log_sum_exp(x):
        """ numerically stable log_sum_exp implementation that prevents overflow """
        axis = len(x.size()) - 1
        m, _ = torch.max(x, dim=axis)
        m2, _ = torch.max(x, dim=axis, keepdim=True)
        return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))

    z_shape = y_pred.size(-1)
    assert z_shape % 3 == 0
    nr_mix = z_shape // 3

    # unpack parameters. (B, T, num_mixtures) x 3
    logit_probs = y_pred[:, :, :nr_mix]
    means = y_pred[:, :, nr_mix : 2 * nr_mix]
    log_scales = torch.clamp(y_pred[:, :, 2 * nr_mix : 3 * nr_mix], min=-7.0)

    # B x T -> B x T x num_mixtures
    y_true = y_true.unsqueeze(-1).expand_as(means)

    centered_y = y_true - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_y + 1.0 / (num_classes - 1))
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1.0 / (num_classes - 1))
    cdf_min = torch.sigmoid(min_in)

    # log probability for edge case of 0 (before scaling)
    # equivalent: torch.log(torch.sigmoid(plus_in))
    log_cdf_plus = plus_in - F.softplus(plus_in)

    # log probability for edge case of 255 (before scaling)
    # equivalent: (1 - torch.sigmoid(min_in)).log()
    log_one_minus_cdf_min = -F.softplus(min_in)

    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_y
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)

    inner_inner_cond = (cdf_delta > 1e-5).float()
    # noinspection PyTypeChecker
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1.0 - inner_inner_cond) * (
        log_pdf_mid - np.log((num_classes - 1) / 2)
    )
    inner_cond = (y_true > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1.0 - inner_cond) * inner_inner_out
    cond = (y_true < -0.999).float()
    log_probs = cond * log_cdf_plus + (1.0 - cond) * inner_out

    log_probs = log_probs + F.log_softmax(logit_probs, -1)

    return -log_sum_exp(log_probs)


def dmld_sample(y):
    """Sample from discretized mixture of logistic distributions.

    Args:
        y (Tensor): B x T x C

    Returns:
        Tensor: sample in range of [-1.0, 1.0].

    """

    z_shape = y.size(-1)
    assert z_shape % 3 == 0
    nr_mix = z_shape // 3

    # B x T x C
    logit_probs = y[:, :, :nr_mix]

    # sample mixture indicator from softmax
    temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-5, 1.0 - 1e-5)
    temp = logit_probs.data - torch.log(-torch.log(temp))
    _, argmax = temp.max(dim=-1)

    # (B, T) -> (B, T, nr_mix)
    one_hot = torch.zeros(argmax.size() + (nr_mix,), dtype=torch.float, device=argmax.device)
    one_hot.scatter_(len(argmax.size()), argmax.unsqueeze(-1), 1.0)

    # select logistic parameters
    means = torch.sum(y[:, :, nr_mix : 2 * nr_mix] * one_hot, dim=-1)
    log_scales = torch.sum(y[:, :, 2 * nr_mix : 3 * nr_mix] * one_hot, dim=-1)
    log_scales = torch.clamp(log_scales, min=-7.0)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = means.data.new(means.size()).uniform_(1e-5, 1.0 - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1.0 - u))

    x = torch.clamp(torch.clamp(x, min=-1.0), max=1.0)

    return x
