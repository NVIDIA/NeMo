w# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import Any, Callable, Generator, List, Literal, Tuple, TypeVar, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.plugins.precision import MixedPrecision
from torch.nn import Module
from torch.optim import Optimizer

from nemo.lightning._strategy_lib import GradScaler

AnyT = TypeVar("AnyT")


def get_optim_config(optimizer: Optimizer):
    try:
        return optimizer.mcore_optimizer.config
    except:
        raise ValueError("Failed to extract optimizer config from module.")

@dataclass
class DtypeConfig:
    fp32: bool = False
    fp16: bool = False
    bf16: bool = False
    params_dtype: torch.dtype = None
    pipeline_dtype: torch.dtype = None
    autocast_dtype: torch.dtype = None
    autocast_enabled: bool = False
    grad_reduce_in_fp32: bool = True
    # fp8 related
    fp8: str = None
    fp8_margin: int = 0
    fp8_interval: int = 1
    fp8_amax_history_len: int = 1
    fp8_amax_compute_algo: str = "most_recent"
    fp8_wgrad: bool = True
    fp8_dot_product_attention: bool = False
    fp8_multi_head_attention: bool = False


class MegatronMixedPrecision(MixedPrecision):
    def __init__(
        self,
        precision: Literal["16-mixed", "bf16-mixed", "32"],
        params_dtype: torch.dtype = None,
        pipeline_dtype: torch.dtype = None,
        autocast_dtype: torch.dtype = None,
        autocast_enabled: bool = False,
        grad_reduce_in_fp32: bool = True,
        # fp8 related,
        fp8: str = None,
        fp8_margin: int = 0,
        fp8_interval: int = 1,
        fp8_amax_history_len: int = 1,
        fp8_amax_compute_algo: str = "most_recent",
        fp8_wgrad: bool = True,
        fp8_dot_product_attention: bool = False,
        fp8_multi_head_attention: bool = False,
        device: str = "cuda",
    ) -> None:

        if isinstance(precision, int):
            precision = str(precision)

        dtype = torch.bfloat16 if precision in ['bf16', 'bf16-mixed'] else torch.float32
        self.dtype_config = DtypeConfig(
            fp32=precision in ['fp32', '32'],
            fp16=precision in ['fp16', 'fp16-mixed', '16'],
            bf16=precision in ['bf16', 'bf16-mixed'],
            params_dtype=params_dtype or torch.float32,
            pipeline_dtype=pipeline_dtype or dtype,
            autocast_dtype=autocast_dtype or dtype,
            autocast_enabled=autocast_enabled,
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            fp8=fp8,
            fp8_margin=fp8_margin,
            fp8_interval=fp8_interval,
            fp8_amax_history_len=fp8_amax_history_len,
            fp8_amax_compute_algo=fp8_amax_compute_algo,
            fp8_wgrad=fp8_wgrad,
            fp8_dot_product_attention=fp8_dot_product_attention,
            fp8_multi_head_attention=fp8_multi_head_attention,
        )
        scaler = None
        if self.dtype_config.fp16:
            scaler = GradScaler(init_scale=2**32, growth_interval=1000, hysteresis=2)
        super().__init__(self.dtype_config, device, scaler)

    def convert_module(self, module: Module) -> Module:
        """Convert the module parameters to the precision type this plugin handles.

        This is optional and depends on the precision limitations during optimization.

        """
        from megatron.core.transformer.module import Float16Module
        from megatron.core.utils import get_model_config

        if self.dtype_config.fp16 or self.dtype_config.bf16:
            # Patch config options
            config = get_model_config(module.module)
            config.fp16 = self.dtype_config.fp16
            config.bf16 = self.dtype_config.bf16
            if hasattr(module, 'module'):
                module.module = Float16Module(config, module.module)
            else:
                module = Float16Module(config, module)

        return module

    def convert_optimizer(self, optimizer: Optimizer) -> Optimizer:
        """Convert the optimizer parameters to the precision type this plugin handles.

        This is optional and depends on the precision limitations during optimization.

        """
        optim_config = get_optim_config(optimizer)
        assert optim_config.bf16 == (self.precision == "bf16-mixed"), "BF16 enabled on model but not on optimizer"
        assert optim_config.fp16 == (self.precision == "fp16-mixed"), "BF16 enabled on model but not on optimizer"
        return optimizer

    def convert_input(self, data: AnyT) -> AnyT:
        """Convert model inputs (forward) to the floating point precision type of this plugin.

        Note: MegatronStrategy will take care of only doing this when:
            parallel_state.is_pipeline_first_stage()

        """
        return data

    def convert_output(self, data: AnyT) -> AnyT:
        """Convert outputs to the floating point precision type expected after model's forward.

        Note: MegatronStrategy will take care of only doing this when:
            parallel_state.is_pipeline_last_stage()

        """
        return data

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        model: Union["pl.LightningModule", torch.nn.Module],
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> None:
        from nemo.core.optim import MainParamsOptimizerWrapper

        if not isinstance(optimizer, MainParamsOptimizerWrapper):
            return super().optimizer_step(optimizer, model, closure, **kwargs)

        if self.scaler is None:
            assert optimizer.fp32_grad_accumulation, "BF16 uses FP32 grad accumulation"
            _ = closure()
            self._after_closure(model, optimizer)
            return optimizer.step(**kwargs)

        assert not optimizer.fp32_grad_accumulation, "FP16 uses FP16 grad accumulation"
        closure_result = closure()

        # TODO: Add an option for merged all-reduce

        # cast fp16 grads to fp32 and copy to main grads, which are used for unscale and param update
        optimizer.copy_model_grads_to_main_grads()
        # `unscale` after the closure is executed but before the `on_before_optimizer_step` hook.
        # unscale main (fp32) gradients
        self.scaler.unscale_(optimizer)
        self._after_closure(model, optimizer)
        skipped_backward = closure_result is None
        # in manual optimization, the closure does not return a value
        if not isinstance(model, pl.LightningModule) or not model.automatic_optimization or not skipped_backward:
            # note: the scaler will skip the `optimizer.step` if nonfinite gradients are found
            self.scaler.step(optimizer, **kwargs)
            self.scaler.update()

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """No explicit precision casting. Inputs are supposed to be manually casted."""
        try:
            yield
        finally:
            pass


def patch_dtype_config(dtype_config, config):
    for field in fields(dtype_config):
        if not hasattr(config, field.name):
            continue
        # If we overwrote a value, throw a warning.
        old_val = getattr(config, field.name)
        new_val = getattr(dtype_config, field.name)
        if old_val != new_val:
            setattr(config, field.name, new_val)
            print(f"Overwrote {type(config).__name__}.{field.name}  {old_val} -> {new_val}")
    return config


__all__ = ["MegatronMixedPrecision"]
