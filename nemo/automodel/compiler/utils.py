# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import re
import torch
import torch.nn as nn


def extract_module_attr_name(pl_module: "pl.LightningModule") -> str:
    """Extracts the held nn.Module from a pl.LightningModule, will try "module", "model", or fail.

    Args:
        pl_module (pl.LightningModule): the LightningModule used in training.

    Raises:
        ValueError: if the pl_module has neither a .mdoel or .module

    Returns:
        str: the attr-name of the nn.Module
    """
    if hasattr(pl_module, 'module'):
        return 'module'
    elif hasattr(pl_module, 'model'):
        return 'model'
    elif isinstance(pl_module, nn.Module):
        return pl_module
    else:
        raise ValueError("Expected lightning_module to have a .model or .module attr.")


def listify(x):
    """Wraps input in a list, if not already a list.

    Args:
        x (Anything): the input, can be anything.

    Returns:
        Anything | list(Anything): Anything (if it's already a list) o/w list(Anything)
    """
    if not isinstance(x, list):
        return [x]
    return x


def get_modules_from_selector(model, module_selector):
    """Iterator over model's modules whose FQN match the module_selector.

    Args:
        model (nn.Module): the model to iterate over.
        module_selector (str): module selector, if empty or '*' will return the whole model. If
        there's an asterisk in the name will match it as a regexp.

    Raises:
        AttributeError: if the user provides an invalid selector.
        AttributeError: if user's selector selects a non-nn.Module attribute.

    Yields:
        Iterator(nn.Module): iterator over modules whose FQN matches module_selector
    """
    if module_selector is None or module_selector == '' or module_selector == '*':
        yield model
        return

    assert isinstance(module_selector, str), module_selector
    atoms: List[str] = module_selector.split('.')
    tmp = model

    for i, item in enumerate(atoms):
        if '*' in item:
            # handle wildcard selector
            # TODO(@akoumparouli): support more complex selectors e.g. net_b.*.net_c.*.conv
            for name, module in tmp.named_children():
                if re.match(item.replace('*', '.*'), name):
                    yield module
            return

        if not hasattr(tmp, item):
            raise AttributeError(tmp._get_name() + " has no " "attribute `" + item + "`")
        tmp = getattr(tmp, item)

        if not isinstance(tmp, torch.nn.Module):
            raise AttributeError("`" + item + "` is not " "an nn.Module")

    yield tmp
