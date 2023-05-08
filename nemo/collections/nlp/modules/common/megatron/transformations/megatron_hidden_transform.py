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

import torch

__all__ = ["MegatronBaseHiddenTransform", "MegatronGaussianHiddenTransform"]


class MegatronBaseHiddenTransform(torch.nn.Module):
    """Base class to apply hidden state transformations"""

    def __init__(self):
        pass

    @property
    def input_names(self):
        return ["hiddens"]

    @property
    def output_names(self):
        return ["hiddens"]

    def _validate_inputs(self, inputs):
        """Validate inputs"""
        # validate inputs
        if not set(self.input_names).isssubset(set(inputs.keys())):
            raise ValueError(f"Inputs should contain {self.input_names}, but got {inputs.keys()}")

    def _transform(self, inputs):
        """Implement your own transformations"""
        outputs = inputs

        return outputs

    def transform(self, inputs):
        """Apply a transformations on the inputs (hiddens is always assumed)"""
        # validate inputs
        self._validate_inputs(inputs)

        outputs = self._transform(inputs)

        return outputs


class MegatronGaussianHiddenTransform(MegatronBaseHiddenTransform):
    """
    Constructes a diagonal Gaussian distribution from the hidden states and samples from it using reparametrization.
    """

    def __init__(self, hidden_size, min_logvar=-8):
        # limit smaller allowed variance (for numerical stability)
        self.min_logvar = min_logvar
        self.hidden_size = hidden_size
        # project hiddens to mean and log variance
        self.hiddens_to_mean_logvar = torch.nn.Linear(hidden_size, hidden_size * 2)

    @property
    def output_names(self):
        return ["z_mean", "z_logvar", "z", "z_log_prob"]

    def _transform(self, inputs):
        """
        inputs:
            hiddens: accepts a tensor of shape (batch_size, seq_len, hidden_size)    
        
        outputs:
            z_mean: mean of Gaussian a tensor of shape (batch_size, seq_len, hidden_size)
            z_logvar: log variance of Gaussian a tensor of shape (batch_size, seq_len, hidden_size)
        """
        hiddens = inputs["hiddens"]
        # compute distribution's parameters (or use cached ones)
        if "z_mean" in inputs and "z_logvar" in inputs:
            z_mean = inputs["z_mean"]
            z_logvar = inputs["z_logvar"]
        else:
            z_mean, z_logvar = self.hiddens_to_mean_logvar(hiddens).chunk(2, dim=-1)
        # clamp logvar
        z_logvar = z_logvar.clamp(min=self.min_logvar)
        # sample z with reparametrization (or use cached one)
        if "z" in inputs:
            z = inputs["z"]
            z_log_prob = inputs.get("z_log_prob", None)
        else:
            e = torch.randn_like(hiddens)
            z = (z_logvar * 0.5).exp() * e + z_mean

        if z_log_prob is None:
            # compute log probability of z under a diagonal Gaussian distribution
            z_log_prob = -0.5 * (math.log(2 * math.pi) + z_logvar + (z - z_mean).pow(2) / z_logvar.exp())
            # sum over the last dimension (hidden_size)
            z_log_prob = z_log_prob.sum(dim=-1)

        return {
            "z": z,
            "z_mean": z_mean,
            "z_logvar": z_logvar,
            "z_log_prob": z_log_prob,
        }
