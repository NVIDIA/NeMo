# =============================================================================
# Copyright (c) 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

from typing import Tuple

from dataclasses import dataclass

__all__ = ['AdamConfig', 'AdamInstanceConfig', 'NovogradConfig', 'NovogradInstanceConfig']


@dataclass
class AdamConfig:
    """
    Default configuration for Adam optimizer.
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).

    ..note:
        For the details on the function/meanings of the arguments, please refer to:
        https://pytorch.org/docs/stable/optim.html?highlight=adam#torch.optim.Adam
    """
    lr: float=0.001
    betas: Tuple[float, float]=(0.9, 0.999)
    eps: float=1e-08
    weight_decay: float=0
    amsgrad: bool=False


@dataclass
class AdamInstanceConfig:
    """
    Default configuration used during automagical instantiation of Adam optimizer.
    """
    cls: str="adam" # @titu90: I honestly prefer the fullly blown: "torch.optim.Adam", let's discuss that.
    params: AdamConfig=AdamConfig()


@dataclass
class NovogradConfig:
    """
    Configuration of the Novograd optimizer.

    It has been proposed  in "Stochastic Gradient Methods with Layer-wise
    Adaptive Moments for Training of Deep Networks"
    (https://arxiv.org/abs/1905.11286)
    
    Args:
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and Beyond"
    """
    lr: float=1e-3
    betas: Tuple[float, float]=(0.95, 0.98)
    eps: float=1e-8
    weight_decay: float=0
    grad_averaging: bool=False
    amsgrad: bool=False
    luc: bool=False
    luc_trust: float=1e-3
    luc_eps: float=1e-8


@dataclass
class NovogradInstanceConfig:
    """
    Default configuration used during automagical instantiation of Novograd optimizer.
    """
    cls: str="novograd" 
    params: NovogradConfig=NovogradConfig()

