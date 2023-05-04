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
from typing import List

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

    def __init__(
        self,
        hidden_transforms: List[MegatronBaseHiddenLoss] = [],
        hidden_loss_transforms: List[MegatronBaseHiddenTransform] = [],
    ):
        self.hidden_transforms = hidden_transforms
        self.hidden_loss_transforms = hidden_loss_transforms

        # register all hidden / loss transforms as submodules to support learned parameters
        if not all([isinstance(ht, MegatronBaseHiddenLoss) for ht in self.hidden_loss_transforms]):
            raise TypeError(
                f"hidden_loss_transforms should be a list of MegatronBaseHiddenLoss, but got {hidden_loss_transforms}"
            )
        self.loss_transforms = torch.nn.ModuleList(self.loss_transforms)
        if not all([isinstance(ht, MegatronBaseHiddenTransform) for ht in self.hidden_transforms]):
            raise TypeError(
                f"hidden_transforms should be a list of MegatronBaseHiddenTransform, but got {hidden_transforms}"
            )
        self.hidden_transforms = torch.nn.ModuleList(self.hidden_transforms)

        # validate that all loss transforms are supported by output of hidden transforms ("hiddens" is given by default)
        loss_inputs = self.loss_inputs

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

    def apply_hidden_transforms(self, inputs):
        """
        Apply hidden transforms
        Args:
            inputs: a dictionary of inputs, with "hiddens" as the default key for hidden states
        
        Returns:
            outputs: a dictionary of outputs, collecting 
        """
        outputs = inputs.copy()
        for hidden_transform in self.hidden_transforms:
            outputs.update(hidden_transform.transform(outputs))

        return outputs

    def apply_loss_transforms(self, outputs):
        """
        Apply loss transforms
        Args:
            outputs: a dictionary of outputs (after hidden transforms)
        
        Returns:
            loss_dict: a dictionary of all losses
        """
        loss_dict = {}
        joint_loss = 0.0
        for i, loss_transform in enumerate(self.loss_transforms):
            cur_loss_dict = loss_transform.loss(outputs)
            joint_loss = joint_loss + cur_loss_dict["loss"]
            cur_loss_dict.pop["loss"]
            # check if cur_loss keys are unique
            dup_keys = set(cur_loss_dict.keys()).intersection(set(loss_dict.keys()))
            if len(dup_keys):
                raise ValueError(
                    f"Loss transform ({i}) {loss_transform} is trying to override the following loss keys {list(dup_keys)}"
                )
            loss_dict.update(cur_loss_dict)

        return loss_dict
