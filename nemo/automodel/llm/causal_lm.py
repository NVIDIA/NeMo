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

import inspect
import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional

import torch
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

from nemo.automodel.loss.linear_ce import HAVE_LINEAR_LOSS_CE
from nemo.automodel.loss.masked_ce import masked_cross_entropy
from nemo.lightning.pytorch.accelerate.transformer_engine import TEConfig
from nemo.lightning.pytorch.callbacks.jit_transform import (
    JitConfig,
    compile_module,
    get_modules_from_selector,
    listify,
)
from nemo.tron.utils.common_utils import get_rank_safe
from nemo.utils.import_utils import safe_import

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class AutoModelForCausalLMConfig:
    """
    Configuration for the HFAutoModelForCausalLM wrapper.
    """

    model_name: str = "gpt2"
    load_pretrained_weights: bool = True
    loss_fn: Optional[Any] = partial(masked_cross_entropy, reduction="sum")  # Target callable for loss function
    model_accelerator: Optional[TEConfig] = None  # Target callable for model acceleration (e.g., TE)
    trust_remote_code: bool = False
    default_dtype: str = "bfloat16"  # e.g., "float32", "bfloat16", "float16"
    load_in_4bit: bool = False
    attn_implementation: str = "sdpa"  # e.g., "sdpa", "eager", "flash_attention_2"
    use_liger_kernel: bool = False
    enable_grad_ckpt: bool = False
    device_map: str = "cpu"
    use_linear_ce_loss: bool = True
    make_vocab_size_divisible_by: int = 128
    jit_config: Optional[JitConfig] = None
    ddp_kwargs: Optional[dict[str, Any]] = None
    calculate_per_token_loss: bool = True
    barrier_with_L1_time: bool = False

    hf_config: Optional[AutoConfig] = field(default=None, init=False)

    def __post_init__(self):
        if self.use_linear_ce_loss and not HAVE_LINEAR_LOSS_CE:
            logger.warning(
                "Dependency for linear CE loss is not available. \
                    Please refer to https://github.com/apple/ml-cross-entropy."
            )
            self.use_linear_ce_loss = False
        logger.info(f"use_linear_ce_loss: {self.use_linear_ce_loss}")

    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.default_dtype, torch.bfloat16)

    def _configure_model(self, attn_implementation):
        """Helper method to initialize and configure the model."""
        # create all your layers here
        auto_cls = AutoModelForCausalLM
        if self.use_liger_kernel:
            liger_kernel_trf, HAS_LIGER_KERNEL = safe_import("liger_kernel.transformers")
            if not HAS_LIGER_KERNEL:
                logger.warning("Asked to use Liger Kernel, but could not import")
            else:
                auto_cls = liger_kernel_trf.AutoLigerKernelForCausalLM

        quantization_config = None
        torch_dtype = self.get_torch_dtype()

        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch_dtype,
            )

        if self.load_pretrained_weights:
            m = auto_cls.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=None if self.load_in_4bit else self.device_map,
                trust_remote_code=self.trust_remote_code,
                attn_implementation=attn_implementation,
                quantization_config=quantization_config,
            )
            self.hf_config = m.config
            return m
        else:
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
            dtype = getattr(config, "torch_dtype", torch_dtype)
            self.hf_config = config
            return auto_cls.from_config(
                config,
                torch_dtype=dtype,
                trust_remote_code=self.trust_remote_code,
                attn_implementation=attn_implementation,
            )

    def configure_model(self):
        """
        Configure and initialize the Hugging Face model.

        This method loads a pretrained model or creates a model from configuration
        based on the config settings. It handles attention implementation fallbacks,
        Liger kernel application, model acceleration, and gradient checkpointing.

        Returns:
            The configured model instance.

        Raises:
            Exception: If model configuration fails.
        """
        try:
            model = self._configure_model(attn_implementation=self.attn_implementation)
            logger.info(f"Configuring model with attn_implementation: {self.attn_implementation}")
        except ValueError as e:
            # 'does not support an attention implementation through torch.nn.functional.scaled_dot_product_attention'
            if "does not support an attention" in str(e):
                logger.warning("Falling back to 'eager' attention implementation.")
                model = self._configure_model(attn_implementation="eager")
            else:
                raise e

        if self.use_liger_kernel:
            from liger_kernel.transformers import _apply_liger_kernel_to_instance

            _apply_liger_kernel_to_instance(model=model)

        if self.model_accelerator is not None:
            from nemo.lightning.pytorch.accelerate.transformer_engine import te_accelerate

            te_accelerate(model, self.model_accelerator.fp8_autocast)

        if self.enable_grad_ckpt:
            if getattr(model, "supports_gradient_checkpointing", False):
                model.gradient_checkpointing_enable()
            else:
                logger.warning("Asked to use gradient checkpoint, but model does not support it")

        model.train()
        return model

    def setup(
        self,
        use_torch_fsdp2: bool = False,
        wrap_with_ddp: bool = False,
        ddp_kwargs: Optional[dict[str, Any]] = None,
    ):
        model = self.configure_model()

        # Print number of parameters.
        if get_rank_safe() == 0:
            logger.info(model)

        # GPU allocation.
        model.cuda(torch.cuda.current_device())
        if self.jit_config is not None:
            jit_compile_model(model, self.jit_config)

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
