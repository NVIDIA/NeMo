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


import functools

import torch

from nemo.collections.nlp.modules.common.megatron.transformations.megatron_hidden_loss import MegatronBaseHiddenLoss
from nemo.collections.nlp.modules.common.megatron.transformations.megatron_hidden_transform import (
    MegatronBaseHiddenTransform,
)

__all__ = ["MegatronHiddensModule"]


class MegatronHiddensModule(torch.nn.Module):
    """
    This class jointly handles the hidden transforms and hidden loss transforms.
    It helps in validating, and applying the transforms.
    """

    def __init__(self, hidden_transforms: MegatronBaseHiddenLoss, hidden_loss_transforms: MegatronBaseHiddenTransform):
        self.hidden_transforms = hidden_transforms
        self.hidden_loss_transforms = hidden_loss_transforms

        # validate that all loss transforms are supported by hidden transforms ("hiddens" is given by default)
        hidden_outputs = set().union(*([ht.output_names for ht in self.hidden_transforms] + ["hiddens"]))
        loss_inputs = set().union(*[lt.input_names for lt in self.loss_transforms])
        if not loss_inputs.issubset(hidden_outputs):
            raise ValueError(
                f"Loss transforms {loss_inputs - hidden_outputs} are not supported by hidden transforms {hidden_outputs}"
            )

        # register all hidden / loss transforms as submodules to support learned parameters
        if self.loss_transforms is not None:
            self.loss_transforms = torch.nn.ModuleList(self.loss_transforms)
        if self.hidden_transforms is not None:
            self.hidden_transforms = torch.nn.ModuleList(self.hidden_transforms)

    @functools.cached_property
    def hidden_outputs(self):
        """Get the hidden outputs from all the hidden transforms"""
        all_output_names = [ht.output_names for ht in self.hidden_transforms] + ["hiddens"]
        output_names = set().union(*all_output_names)
        # make sure there are no duplicate output names
        if len(output_names) != len(all_output_names):
            # collect all duplicate output names
            duplicate_names = set([x for x in all_output_names if all_output_names.count(x) > 1])
            raise ValueError(f"Hidden transforms have duplicate output names: {list(duplicate_names)}")

        return output_names

    @functools.cached_property
    def loss_inputs(self):
        """Get the loss inputs from all the loss transforms"""
        hidden_outputs = set(self.hidden_outputs)
        loss_inputs = set().union(*[lt.input_names for lt in self.hidden_loss_transforms])
        if not loss_inputs.issubset(hidden_outputs):
            raise ValueError(
                f"Loss transforms {loss_inputs - hidden_outputs} are not supported by hidden transforms {hidden_outputs}"
            )

        return list(loss_inputs)
