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

from typing import Optional

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.trainer import Trainer
from nemo.lightning.io.mixin import IOMixin


def extract_module_attr_name(pl_module: "pl.LightningModule") -> str:
    if hasattr(pl_module, 'module'):
        return 'module'
    elif hasattr(pl_module, 'model'):
        return 'model'
    else:
        raise ValueError("Expected lightning_module to have a .model or .module attr.")

def listify(x):
    if not isinstance(x, list):
        return [x]
    return x

class JitConfig:
    target_module: str = ''
    use_torch_compile: bool = False
    use_thunder: bool = False
    profile_thunder: bool = False

class JitTransform(Callback, IOMixin):
    """
    Apply JIT-compling on PyTorch model

    Args:
        config (JitConfig): The jit-compiler config to use.

    Example:
        >>> from nemo.lightning.pytorch.callbacks import JitTransform
        >>> trainer = Trainer(callbacks=[JitTransform(JitConfig(use_torch_compile=True))])
    """

    def __init__(self, config: JitConfig = None):
        self.config = config

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.config is None:
            return
        attr_name = extract_module_attr_name(pl_module)
        model = getattr(pl_module, attr_name)

        if getattr(pl_module, '_compiled', False) == False:
            return
        pl_module._compiled = True

        for config in listify(self.config):
            if isinstance(config.target_module, str) and config.target_module != '':
                module = model
            module = model
            if self.backend == 'torch':
                model.compile()
            elif self.backend == 'thunder':
                import thunder
                import thunder.dynamo
                from thunder.dev_utils.nvtx_profile_transform import NvtxProfileTransform
                # With this setting, Dynamo Graphs inline all the modules (so Dynamo FXGraph just
                # consists of `call_function` nodes only and no `call_module` node.
                # This is the default setting in PyTorch 2.5 onwards
                # (see https://github.com/pytorch/pytorch/pull/131275)
                torch._dynamo.config.inline_inbuilt_nn_modules = True

                xforms: list = [NvtxProfileTransform()] if config.profile_thunder else []
                be = thunder.dynamo.ThunderCompiler(transforms=xforms)
                model.compile(backend=be)
            else:
                raise ValueError("got unexpected backend")
