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


import torch


from nemo.automodel.compiler.configs import ThunderConfig, TorchCompileConfig
from nemo.automodel.compiler.utils import extract_module_attr_name, get_modules_from_selector


def compile_module(config, module):
    """Jit-compiles an nn.Module

    Args:
        config (TorchCompileConfig, ThunderConfig): compiler config
        module (nn.Module): the module to be compiled

    Returns:
        nn.Module: the (potentially) compiled module
    """
    if isinstance(config, TorchCompileConfig):
        module.compile(**(config.kwargs or {}))
    elif isinstance(config, ThunderConfig):
        import thunder
        import thunder.dynamo
        from thunder.dev_utils.nvtx_profile_transform import NvtxProfileTransform

        # With this setting, Dynamo Graphs inline all the modules (so Dynamo FXGraph just
        # consists of `call_function` nodes only and no `call_module` node.
        # This is the default setting in PyTorch 2.5 onwards
        # (see https://github.com/pytorch/pytorch/pull/131275)
        torch._dynamo.config.inline_inbuilt_nn_modules = True

        xforms: list = [NvtxProfileTransform()] if config.profile else []
        module.compile(backend=thunder.dynamo.ThunderCompiler(transforms=xforms))
    else:
        raise ValueError("Expected config to be TorchCompileConfig or ThunderConfig")


def compile_module_from_config(config, module) -> None:
    """Jit-compiles the model at the start of the epoch.
    While other events such as on_train_start are more suitable, we use on_train_epoch_start
    since that is what is used in peft (we want to jit after adding the adapters).

    Args:
        module (nn.Module): the nn.Module to compile.
    """
    if config is None:
        return
    if not isinstance(config, (TorchCompileConfig, ThunderConfig)):
        return

    attr_name = extract_module_attr_name(module)
    model = getattr(module, attr_name)

    if getattr(module, '_compiled', False) == True:
        return

    # TODO(@akoumparouli): you want to concatenate (via regex OR-operator) all expressions
    # and trigger the compile if anyone matches, instead of iterating over all O(N^2).
    compiled = False
    for module in get_modules_from_selector(model, config.module_selector):
        compile_module(config, module)
        compiled = True

    setattr(module, '_compiled', compiled)
