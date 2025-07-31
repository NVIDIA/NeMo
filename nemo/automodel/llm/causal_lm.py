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
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Iterable, Optional, Protocol

import torch
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

from nemo.automodel.loss.linear_ce import (HAVE_LINEAR_LOSS_CE,
                                           fused_linear_cross_entropy)
from nemo.automodel.loss.masked_ce import masked_cross_entropy
from nemo.lightning.pytorch.accelerate.transformer_engine import TEConfig
from nemo.lightning.pytorch.callbacks.jit_transform import JitConfig
from nemo.tron.llm.utils import get_batch_from_iterator
from nemo.tron.state import GlobalState
from nemo.tron.utils.common_utils import get_world_size_safe
from nemo.utils.import_utils import safe_import

logger = logging.getLogger(__name__)


class FinalizeModelGradsFnProtocol(Protocol):
    def __call__(
        self,
        model: torch.nn.Module,
        total_num_tokens: Optional[torch.Tensor] = None,
    ) -> None: ...


def finalize_model_grads(model: torch.nn.Module, total_num_tokens: Optional[torch.Tensor] = None):
    if total_num_tokens is None:
        return

    group = None
    # This is the size of the data parallel group, since DDP all reduces grads via averaging
    group_size = get_world_size_safe()
    # TODO: Add support to select data parallel group for FSDP etc

    num_tokens = total_num_tokens.clone().detach()
    torch.distributed.all_reduce(num_tokens, group=group)
    scaling_factor = group_size / num_tokens
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.mul_(scaling_factor)


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
    finalize_model_grads_func: Optional[FinalizeModelGradsFnProtocol] = finalize_model_grads
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
            from liger_kernel.transformers import \
                _apply_liger_kernel_to_instance

            _apply_liger_kernel_to_instance(model=model)

        if self.model_accelerator is not None:
            from nemo.lightning.pytorch.accelerate.transformer_engine import \
                te_accelerate

            te_accelerate(model, self.model_accelerator.fp8_autocast)

        if self.enable_grad_ckpt:
            if getattr(model, "supports_gradient_checkpointing", False):
                model.gradient_checkpointing_enable()
            else:
                logger.warning("Asked to use gradient checkpoint, but model does not support it")

        model.train()
        return model


def model_forward(model, batch, num_logits_to_keep=None):
    """
    Perform a forward pass of the model.

    Args:
        batch (dict): A dictionary of inputs that the model expects.
        num_logits_to_keep (int, optional): The number of logits to keep. 0 means all logits are kept.
    Returns:
        ModelOutput: The output of the underlying Hugging Face model.
    """
    if num_logits_to_keep is None:
        return model(**batch)
    # Check if num_logits_to_keep parameter exists in model's forward method
    model_forward_params = inspect.signature(model.forward).parameters
    if "num_logits_to_keep" in model_forward_params:
        return model(**batch, num_logits_to_keep=num_logits_to_keep)
    if "logits_to_keep" in model_forward_params:
        return model(**batch, logits_to_keep=num_logits_to_keep)
    return model(**batch)


def forward_with_loss_no_cp(model, batch, labels, loss_mask, config: AutoModelForCausalLMConfig):
    batch["output_hidden_states"] = True if config.use_linear_ce_loss else False  # Enable hidden states output

    if not config.use_linear_ce_loss:
        outputs = model_forward(model, batch)
        # Prepare for loss calculation
        logits = outputs.logits
        n_cls = logits.shape[-1]
        logits = logits.view(-1, n_cls)
        labels = labels.view(-1)
        assert logits.shape[-2] == labels.shape[-1], "Expected logits & labels to have the same length"
        loss = config.loss_fn(logits, labels, loss_mask)
    else:
        # use num_logits_to_keep=1 to avoid full logits matrix in memory
        # TODO: test CE with CP enabled
        outputs = model_forward(model, batch, num_logits_to_keep=1)
        hidden_states = outputs.hidden_states[-1]
        lm_head = model.get_output_embeddings().weight  # Get the weight matrix
        if loss_mask is not None:
            # Replace labels with -100 where mask is 0 (don't compute loss for these positions)
            # -100 is the default ignore index in PyTorch's cross entropy loss
            labels = labels.masked_fill(loss_mask == 0, -100)
        num_items_in_batch = torch.count_nonzero(labels != -100).item()
        logit_softcapping = 0
        loss = fused_linear_cross_entropy(
            hidden_states=hidden_states,
            lm_weight=lm_head,
            labels=labels,
            num_items_in_batch=num_items_in_batch,
            logit_softcapping=logit_softcapping,
        )

    return loss


def forward_step(
    state: GlobalState, data_iterator: Iterable, model: torch.nn.Module, config: AutoModelForCausalLMConfig
):
    timers = state.timers
    straggler_timer = state.straggler_timer

    timers("batch-generator", log_level=2).start()
    with straggler_timer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch_from_iterator(data_iterator).values()
    timers("batch-generator").stop()

    batch = {}
    batch["input_ids"] = tokens
    batch["attention_mask"] = loss_mask.bfloat16()
    # if attention_mask is not None:
    # Change to HF Transformer format
    # batch["attention_mask"] = torch.logical_not(attention_mask).bfloat16()
    # batch["position_ids"] = position_ids
    # TODO(@boxiangw): Refractor. Needed for SP support
    batch["position_ids"] = torch.arange(0, batch["input_ids"].shape[1]).unsqueeze(0).cuda(non_blocking=True)

    # batch = _remove_extra_batch_keys(batch)
    with straggler_timer:
        output_tensor = forward_with_loss_no_cp(model, batch, labels, loss_mask, config)

    return output_tensor, loss_mask.sum()
