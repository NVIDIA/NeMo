# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from contextlib import nullcontext
from typing import Any, Optional

import torch
from torch.nn.parallel import DistributedDataParallel

from nemo.automodel.llm.causal_lm import AutoModelForCausalLMConfig
from nemo.lightning.pytorch.callbacks.jit_transform import (
    JitConfig,
    compile_module,
    get_modules_from_selector,
    listify,
)
from nemo.tron.utils.common_utils import get_rank_safe

logger = logging.getLogger(__name__)


def get_model_from_config(
    model_config: AutoModelForCausalLMConfig,
    use_torch_fsdp2: bool = False,
    wrap_with_ddp: bool = False,
    ddp_kwargs: Optional[dict[str, Any]] = None,
):
    model = model_config.configure_model()

    # Print number of parameters.
    if get_rank_safe() == 0:
        logger.info(model)

    # GPU allocation.
    model.cuda(torch.cuda.current_device())
    if model_config.jit_config is not None:
        jit_compile_model(model, model_config.jit_config)

    if wrap_with_ddp:
        if use_torch_fsdp2:
            ...
        else:
            device_ids = [torch.cuda.current_device()]
            ctx = torch.cuda.stream(torch.cuda.Stream()) if device_ids is not None else nullcontext()
            with ctx:
                model = DistributedDataParallel(module=model, device_ids=device_ids, **(ddp_kwargs or {}))
    return model


def jit_compile_model(model: torch.nn.Module, jit_config: JitConfig):
    """Jit-compiles the model at the start of the epoch.
    While other events such as on_train_start are more suitable, we use on_train_epoch_start
    since that is what is used in peft (we want to jit after adding the adapters).

    Args:
        trainer (pl.Trainer): PTL trainer
        pl_module (pl.LightningModule): PTL module
    """
    if jit_config is None:
        return
    if not jit_config.use_thunder and not jit_config.use_torch:
        return

    if getattr(model, "_compiled", False):
        return

    # TODO(@akoumparouli): you want to concatenate (via regex OR-operator) all expressions
    # and trigger the compile if anyone matches, instead of iterating over all O(N^2).
    compiled = False
    for config in listify(jit_config):
        for module in get_modules_from_selector(model, config.module_selector):
            compiled |= compile_module(config, module)

    setattr(model, "_compiled", compiled)
