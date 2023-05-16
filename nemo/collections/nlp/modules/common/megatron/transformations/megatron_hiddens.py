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
"""
In order to register external hidden transforms and losses please use the following methods:
* register_hidden_loss(cls_name: str, class_path: str)
* register_hidden_transform(cls_name: str, class_path: str)

This will add support to corresponding config entries:
model:
    hiddens:
        enabled: True
        enc_output_name: <name of the encoder output>
        transform:
            name: 
                cls_name: <cls_name>
                ... (all related kwargs)
        loss:
            name: 
                cls_name: <cls_name>
                ... (all related kwargs)
"""

import functools
from typing import List

import torch
from omegaconf.omegaconf import OmegaConf

from nemo.collections.nlp.modules.common.megatron.transformations.megatron_hidden_loss import MegatronBaseHiddenLoss
from nemo.collections.nlp.modules.common.megatron.transformations.megatron_hidden_transform import (
    MegatronBaseHiddenTransform,
)
from nemo.utils import logging
from nemo.utils.model_utils import import_class_by_path

__all__ = ["MegatronHiddensModule"]

# a registry of all hidden transforms (maps name to class path)
_LOSS_CLASS_REGISTRY = {
    "a_mim": "nemo.collections.nlp.modules.common.megatron.transformations.megatron_hidden_loss.MegatronMIMHiddenLoss",
    "vae": "nemo.collections.nlp.modules.common.megatron.transformations.megatron_hidden_loss.MegatronVAEHiddenLoss",
}

# a registry of all hidden losses (maps name to class path)
_TRANSFORM_CLASS_REGISTRY = {
    "cond_gaussian": "nemo.collections.nlp.modules.common.megatron.transformations.megatron_hidden_transform.MegatronGaussianHiddenTransform",
}


def get_registered_hiddens():
    """
    Return:
        A dictionary with all registered hidden transforms and losses.

    Example:
        {
            "loss": ["a-mim", "vae"],
            "transform": ["cond_gaussian"],
        }
    """
    return {
        "loss": list(_LOSS_CLASS_REGISTRY.keys()),
        "transform": list(_TRANSFORM_CLASS_REGISTRY.keys()),
    }


def register_hidden_loss(cls_name: str, class_path: str):
    """
    Register a hidden loss.

    
    Args:
        cls_name: name of the class
        class_path: path to the class (e.g., "nemo.collections.nlp.modules.common.megatron.transformations.megatron_hidden_transform.MegatronGaussianHiddenTransform")
    """
    if cls_name in _LOSS_CLASS_REGISTRY:
        raise ValueError(f"Cannot register duplicate hidden loss ({cls_name})")
    _LOSS_CLASS_REGISTRY[cls_name] = class_path
    logging.info(f"Registered hidden loss {cls_name} at {class_path}")


def register_hidden_transform(cls_name: str, class_path: str):
    """
    Register a hidden transform.
    
    Args:
        cls_name: name of the class
        class_path: path to the class (e.g., "nemo.collections.nlp.modules.common.megatron.transformations.megatron_hidden_transform.MegatronGaussianHiddenTransform")
    """
    if cls_name in _TRANSFORM_CLASS_REGISTRY:
        raise ValueError(f"Cannot register duplicate hidden transform ({cls_name})")
    _TRANSFORM_CLASS_REGISTRY[cls_name] = class_path
    logging.info(f"Registered hidden transform {cls_name} at {class_path}")


def get_hiddens_module(cfg=None):
    """Build a MegatronHiddensModule from a configuration cfg"""
    # check if we need to build a hiddens module - model.hiddens.enabled must be True
    if cfg is None or not cfg.get("enabled", False):
        return None

    # build all hidden transforms
    transform_cfg = cfg.get("transform", [])
    hidden_transforms = []
    # we expect transform_cfg to be a list of dictionaries
    for cur_list_cfg in transform_cfg.items():
        for name, cur_cfg in cur_list_cfg.items():
            cls_kwargs = OmegaConf.to_container(cur_cfg)
            if not "cls_name" in cls_kwargs:
                raise KeyError(f"Missing 'cls_name' in hidden transform {name}")

            cls_name = cls_kwargs.pop("cls_name")
            if cls_name not in _TRANSFORM_CLASS_REGISTRY:
                raise KeyError(f"Unknown hidden transform {cls_name}, available: {_TRANSFORM_CLASS_REGISTRY.keys()}")
            cur_transform = import_class_by_path(_TRANSFORM_CLASS_REGISTRY[cls_name])(**cls_kwargs)
            hidden_transforms.append(cur_transform)

    # build all hidden losses
    loss_cfg = cfg.get("loss", {})
    hidden_loss_transforms = []
    # we expect loss_cfg to be a list of dictionaries
    for cur_list_cfg in loss_cfg.items():
        for name, cur_cfg in cur_list_cfg.items():
            cls_kwargs = OmegaConf.to_container(cur_cfg)
            if not "cls_name" in cls_kwargs:
                raise KeyError(f"Missing 'cls_name' in hidden loss {name}")

            cls_name = cls_kwargs.pop("cls_name")
            if cls_name not in _LOSS_CLASS_REGISTRY:
                raise KeyError(f"Unknown hidden loss {cls_name}, available: {_LOSS_CLASS_REGISTRY.keys()}")
            cur_loss = import_class_by_path(_LOSS_CLASS_REGISTRY[cls_name])(**cls_kwargs)
            hidden_loss_transforms.append(cur_loss)

    enc_output_name = cfg.get("enc_output_name", "hiddens")

    return MegatronHiddensModule(
        hidden_transforms=hidden_transforms,
        hidden_loss_transforms=hidden_loss_transforms,
        enc_output_name=enc_output_name,
    )


class MegatronHiddensModule(torch.nn.Module):
    """
    This class jointly handles the hidden transforms and hidden loss transforms.
    It helps in validating, and applying the transforms.
    """

    def __init__(
        self,
        hidden_transforms: List[MegatronBaseHiddenLoss] = [],
        hidden_loss_transforms: List[MegatronBaseHiddenTransform] = [],
        enc_output_name: str = "hiddens",
    ):
        self.hidden_transforms = hidden_transforms
        self.hidden_loss_transforms = hidden_loss_transforms
        self.enc_output_name = enc_output_name

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

        return list(output_names)

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

        loss_dict["loss"] = joint_loss

        return loss_dict

    def get_enc_output(self, outputs):
        """
        Returns the encoder output from transformed hiddens output.
        e.g., return z for latent variable models.

        Args:
            outputs: a dictionary of outputs (after hidden transforms)
        
        Returns:
            enc_output: a tensor encoder outputs (e.g., to be used by decoder)
        """
        if torch.is_tensor(outputs):
            enc_output = outputs
        else:
            enc_output = outputs[self.enc_output_name]

        return enc_output
