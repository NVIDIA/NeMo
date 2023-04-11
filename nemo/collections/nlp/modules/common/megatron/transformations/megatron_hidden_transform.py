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


import os
import sys
import torch

__all__ = ["MegatronGaussianHiddenTransform"]


class MegatronHiddenTransform(object):
    """Base class to apply hidden state transformations"""

    def __init__(self):
        pass

    def transform(self, hiddens, **kwargs):
        """Apply your own transformations on the hiddens"""
        pass


class MegatronGaussianHiddenTransform(MegatronHiddenTransform):
    # TODO: add docstring
    def transform(self, hiddens):
        # import pudb; pudb.set_trace()
        self.hiddens = hiddens
        e = torch.randn(hiddens.shape).cuda()
        z_mean = self._hiddens_mean()
        z_logvar = self._hiddens_logvar()
        z = torch.exp(e * (z_logvar * 0.5)) + z_mean
        return {"hiddens": z, "z_mean": z_mean, "z_logvar": z_logvar}

    def _hiddens_mean(self):
        return torch.mean(self.hiddens)

    def _hiddens_logvar(self):
        return torch.log(torch.var(self.hiddens))
