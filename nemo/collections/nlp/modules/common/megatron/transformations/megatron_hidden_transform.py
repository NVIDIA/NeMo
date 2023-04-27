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


import torch

__all__ = ["MegatronGaussianHiddenTransform"]


class MegatronBaseHiddenTransform(object):
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
            raise ValueError(
                f"Inputs should contain {self.input_names}, but got {inputs.keys()}"
            )            
        
    def transform(self, **inputs):
        """Apply your own transformations on the inputs (hiddens is always assumed)"""
        # validate inputs
        self._validate_inputs(inputs)
        
        return inputs


class MegatronGaussianHiddenTransform(MegatronBaseHiddenTransform):
    """
    Constructes a diagonal Gaussian distribution from the hidden states and samples from it using reparametrization.
    """
    def __init__(self, hidden_size, min_logvar=-8):
        # limit smaller allowed variance (for numerical stability)
        self.min_logvar = min_logvar
        self.hidden_size = hidden_size
        # project hiddens to mean and log variance
        self.hiddens_to_mean_logvar = torch.nn.Linear(hidden_size, hidden_size*2)

    @property
    def output_names(self):
        return ["z_mean", "z_logvar", "z"]

    def transform(self, **inputs):
        """
        inputs:
            hiddens: accepts a tensor of shape (batch_size, seq_len, hidden_size)    
        
        outputs:
            z_mean: mean of Gaussian a tensor of shape (batch_size, seq_len, hidden_size)
            z_logvar: log variance of Gaussian a tensor of shape (batch_size, seq_len, hidden_size)
        """
        # validate inputs
        self._validate_inputs(inputs)
        
        hiddens = inputs["hiddens"]
        # compute distribution
        z_mean, z_logvar = self.hiddens_to_mean_logvar(hiddens).chunk(2, dim=-1)
        # clamp logvar
        z_logvar = z_logvar.clamp(min=self.min_logvar)
        # sample z with reparametrization
        e = torch.randn_like(hiddens)
        z = (z_logvar * 0.5).exp() * e + z_mean

        return {"z": z, "z_mean": z_mean, "z_logvar": z_logvar}
