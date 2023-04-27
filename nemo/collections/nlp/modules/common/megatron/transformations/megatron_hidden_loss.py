# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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


import math
import os
import sys

import torch

__all__ = ["MegatronMIMHiddenLoss"]


class BaseMegatronHiddenLoss(object):
    """Base class to calculate hidden state loss"""

    def __init__(self):
        pass

    @property
    def input_names(self):
        return []

    def loss(self, hiddens_dict, **kwargs):
        """Implement your own loss calculations"""
        pass


class MegatronMIMHiddenLoss(MegatronHiddenLoss):
    # TODO: add docstring
    """
    hiddens_dict: accepts a dictionary that contains hiddens, z_mean and z_logvar
    alpha: a factor to multiply the hidden loss with.
    """

    def loss(self, hiddens_dict, alpha=1):
        # import pudb; pudb.set_trace()
        log_p_z = self._log_prob(hiddens_dict["hiddens"], 1, 0)
        log_q_z_given_x = self._log_prob(hiddens_dict["hiddens"], hiddens_dict["z_logvar"], hiddens_dict["z_mean"])
        return alpha * (log_p_z - log_q_z_given_x)

    def _log_prob(self, z, z_logvar, z_mean):
        log_scale = 0.5 * z_logvar
        var = math.exp(z_logvar)
        k = 1  ##TODO: what is k ?
        return (
            -((z - z_mean) ** 2) / (2 * var)
            - log_scale
            - k * math.log(math.sqrt(2 * math.pi))
            - 1 / 2 * math.log(z_logvar).sum(dim=-1)
        )
