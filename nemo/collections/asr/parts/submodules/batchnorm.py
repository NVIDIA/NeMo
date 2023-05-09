# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from __future__ import annotations

from functools import reduce
from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from pytorch_lightning.pytorch.plugins.layer_sync import LayerSync


class FusedBatchNorm1d(nn.Module):
    """
    Fused BatchNorm to use in Conformer to improve accuracy in finetuning with TTS scenario
    Drop-in replacement for BatchNorm1d with simple affine projection
    """

    def __init__(self, num_features: int):
        """
        Args:
            num_features: number of channels, see original BatchNorm1d documentation
        """
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    @classmethod
    def from_batchnorm(cls, bn: nn.BatchNorm1d) -> FusedBatchNorm1d:
        """
        Construct FusedBatchNorm1d module from BatchNorm1d
        Args:
            bn: original BatchNorm module

        Returns:
            FusedBatchNorm1d module with initialized params; in eval mode result is equivalent to original BatchNorm
        """
        assert isinstance(bn, nn.BatchNorm1d)
        fused_bn = FusedBatchNorm1d(bn.num_features)
        # init projection params from original batch norm
        # so, for inference mode output is the same
        std = torch.sqrt(bn.running_var.data + bn.eps)
        fused_bn.weight.data = bn.weight.data / std
        fused_bn.bias.data = bn.bias.data - bn.running_mean.data * fused_bn.weight.data
        return fused_bn

    def forward(self, x: torch.Tensor):
        if x.dim() == 3:
            return x * self.weight.unsqueeze(-1) + self.bias.unsqueeze(-1)
        assert x.dim() == 2
        return x * self.weight + self.bias


def _get_module_by_name(module: nn.Module, full_layer_name: str) -> nn.Module:
    names = full_layer_name.split(sep='.')
    return reduce(getattr, names, module)


def replace_bn_with_fused_bn(module: nn.Module, full_layer_name: str):
    """
    Replace BatchNorm1d named `full_layer_name` in nn.Module with FusedBatchNorm1d
    Args:
        module: nn.Module instance, modified inplace
        full_layer_name: name of BatchNorm1d submodule in module to replace
    """
    bn = _get_module_by_name(module, full_layer_name)
    assert isinstance(bn, nn.BatchNorm1d)
    fused_bn = FusedBatchNorm1d.from_batchnorm(bn)
    try:
        parent_name, norm_name = full_layer_name.rsplit(".", maxsplit=1)
        setattr(_get_module_by_name(module, parent_name), norm_name, fused_bn)
    except ValueError:
        norm_name = full_layer_name
        setattr(module, norm_name, fused_bn)


def replace_bn_with_fused_bn_all(model: nn.Module) -> List[str]:
    """
    Replace BatchNorm1d with FusedBatchNorm1d in model
    Args:
        model: nn.Module instance, modified inplace

    Returns:
        list of replaced module names
    """
    replaced_module_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            replace_bn_with_fused_bn(model, name)
            replaced_module_names.append(name)
    return replaced_module_names


class SafeSyncBatchNorm(torch.nn.SyncBatchNorm):
    """
    SyncBatchNorm that works with empty inputs.
    """


    def forward(self, input: Tensor) -> Tensor:
        r"""
        Fix for NaN in inputs. (The only difference wrt original)
        """
        input = torch.nan_to_num(input)
        return super().forward(input)


    @classmethod
    def convert_safesync_batchnorm(cls, module, process_group=None):

        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = SafeSyncBatchNorm(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                process_group,
            )
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(
                name, cls.convert_safesync_batchnorm(child, process_group)
            )
        del module
        return module_output


class TorchSafeSyncBatchNorm(LayerSync):
    """A plugin that wraps all batch normalization layers of a model with synchronization logic for
    multiprocessing.
    # adapted from https://github.com/Lightning-AI/lightning/blob/master/src/lightning/pytorch/plugins/layer_sync.py
    """

    def apply(self, model: Module) -> Module:
        """Add global batchnorm for a model spread across multiple GPUs and nodes.
        Override this method to synchronize batchnorm layers between specific process groups instead
        of the whole world.
        Args:
            model: Reference to the current LightningModule
        Return:
            LightningModule with batchnorm layers synchronized within the process groups.
        """
        return SafeSyncBatchNorm.convert_safesync_batchnorm(model)

    def revert(self, model: Module) -> Module:
        """Convert the wrapped batchnorm layers back to regular batchnorm layers.
        Args:
            model: Reference to the current LightningModule
        Return:
            LightningModule with regular batchnorm layers that will no longer sync across processes.
        """
        converted_module = model
        if isinstance(model, SafeSyncBatchNorm):
            # Unfortunately, LayerSync does not store the original class - if it did
            # we could return the one that was originally created.
            converted_module = _BatchNorm(
                model.num_features, model.eps, model.momentum, model.affine, model.track_running_stats
            )
            if model.affine:
                with torch.no_grad():
                    converted_module.weight = model.weight
                    converted_module.bias = model.bias
            converted_module.running_mean = model.running_mean
            converted_module.running_var = model.running_var
            converted_module.num_batches_tracked = model.num_batches_tracked
            if hasattr(model, "qconfig"):
                converted_module.qconfig = model.qconfig
        for name, child in model.named_children():
            converted_module.add_module(name, self.revert(child))
        del model
        return 
