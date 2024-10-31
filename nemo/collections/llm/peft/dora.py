import re
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import torch
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    gather_from_tensor_model_parallel_region,
)
from megatron.core.utils import make_sharded_tensor_for_checkpoint, make_tp_sharded_tensor_for_checkpoint
from torch import nn

from nemo.collections.llm.peft.lora import LinearAdapter
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import ParallelLinearAdapter
from nemo.lightning.pytorch.callbacks.peft import PEFT, AdapterWrapper
from nemo.utils import logging
from nemo.utils.import_utils import safe_import_from

TEColumnParallelLinear, HAVE_TE_COL_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine", "TEColumnParallelLinear"
)
TELayerNormColumnParallelLinear, HAVE_TE_LN_COL_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine",
    "TELayerNormColumnParallelLinear",
)
TERowParallelLinear, HAVE_TE_ROW_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine", "TERowParallelLinear"
)
HAVE_TE = all((HAVE_TE_COL_LINEAR, HAVE_TE_LN_COL_LINEAR, HAVE_TE_ROW_LINEAR))


class ParallelLinearDoRAAdapter(ParallelLinearAdapter):
    def init_weight_magnitude(self, value):
        # weight_magnitude has shape (d,) where d is the output dim of the linear layer
        self.weight_magnitude = nn.Parameter(value, requires_grad=True)

    def get_weight_magnitude(self):
        return self.weight_magnitude

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)

        magnitude_key = f"{prefix}weight_magnitude"
        if self.input_is_parallel:
            # RPL output is gathered, so weight_magnitude is not sharded for TP
            magnitude_sharded_tensor = make_sharded_tensor_for_checkpoint(
                self.weight_magnitude, magnitude_key, prepend_offsets=sharded_offsets
            )
        else:
            # CPL output is sharded, so weight_magnitude is sharded for TP
            magnitude_sharded_tensor = make_tp_sharded_tensor_for_checkpoint(
                self.weight_magnitude, magnitude_key, 0, prepend_offsets=sharded_offsets
            )
        sharded_state_dict[magnitude_key] = magnitude_sharded_tensor

        return sharded_state_dict


class DoRALinear(AdapterWrapper):
    """TODO"""

    def __init__(self, to_wrap: nn.Module, adapter: ParallelLinearDoRAAdapter):
        super().__init__(to_wrap, adapter)
        self.adapter: ParallelLinearDoRAAdapter
        self.scaling = adapter.alpha / adapter.dim
        self.adapter.init_weight_magnitude(self._get_weight_norm())

    def _get_weight_norm(self):
        if self.adapter.input_is_parallel:
            linear_out_weight = gather_from_tensor_model_parallel_region(self.adapter.linear_out.weight.T).T
            linear_in_weight = self.adapter.linear_in.weight
        else:
            linear_out_weight = self.adapter.linear_out.weight
            linear_in_weight = gather_from_tensor_model_parallel_region(self.adapter.linear_in.weight.T).T

        weight = self.to_wrap.weight + self.scaling * linear_out_weight @ linear_in_weight
        return torch.linalg.norm(weight, dim=1).to(weight.dtype).detach()

    def forward(self, x):
        linear_output, bias, layernorm_output = self.base_linear_forward(x)
        adapter_output = self.adapter(layernorm_output.contiguous())

        # mag_norm_scale is  ||W_0 + B_0 A_0|| / ||W_0 + B A||  (scaling in front of BA not shown)
        mag_norm_scale = (self.adapter.get_weight_magnitude() / self._get_weight_norm()).view(1, 1, -1)
        """
          mag_norm_scale * (linear_output + adapter_output)
        = ||W_0 + B_0 A_0|| / ||W_0 + B A|| * (W_0 x + B A x)
        = ||W_0 + B_0 A_0|| ((W_0 + B A) / ||W_0 + B A||) x
        = m ((W_0 + B A) / ||W_0 + B A||) x
        = equation 5 in DoRA paper
        """
        return mag_norm_scale * (linear_output + adapter_output), bias


@dataclass
class DoRA(PEFT):
    """
    TODO
    """

    target_modules: List[str] = field(
        default_factory=lambda: ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2']
    )
    dim: int = 32
    alpha: int = 32
    dropout: float = 0.0
    dropout_position: Literal['pre', 'post'] = 'post'
    lora_A_init_method: str = "xavier"
    lora_B_init_method: str = "zero"

    def transform(self, m: nn.Module, name=None, prefix=None):
        """
        Applies DoRA to a specific module within the model architecture.

        Args:
            m (nn.Module): The module to apply DoRA to.
            name (str, optional): Name of the module (if applicable). Defaults to None.
            prefix (str, optional): Prefix for the module name (if applicable). Defaults to None.

        Returns:
            nn.Module: The modified module with DoRA applied, or the original module if not a target.
        """

        def wildcard_match(pattern, key):
            if key is None:
                return None
            regex_pattern = re.compile("^" + pattern.replace("*", "(.*)") + "$")
            match = regex_pattern.match(key)
            return match is not None

        full_name = f"{prefix}.{name}" if prefix else name
        if name in self.target_modules or any(wildcard_match(pattern, full_name) for pattern in self.target_modules):
            if HAVE_TE and isinstance(m, TEColumnParallelLinear) or isinstance(m, TELayerNormColumnParallelLinear):
                input_is_parallel = False
                # m.in_features and m.out_features are divided by tp_size already,
                # but in_features and out_features passed to ParallelLinearAdapter are not.
                tp_size = parallel_state.get_tensor_model_parallel_world_size()
                in_features = m.in_features
                out_features = m.out_features * tp_size
                # DoRA is applied after layernorm, so layernorm output must be returned
                m.return_layernorm_output = True
                # perf optimization for DoRA + SP (to check!)
                if m.config.sequence_parallel and not m.ub_overlap_ag:
                    m.return_layernorm_output_gathered = True
            elif HAVE_TE and isinstance(m, TERowParallelLinear):
                input_is_parallel = True
                tp_size = parallel_state.get_tensor_model_parallel_world_size()
                in_features = m.in_features * tp_size
                out_features = m.out_features
            elif isinstance(m, ColumnParallelLinear):
                input_is_parallel = False
                in_features = m.input_size
                out_features = m.output_size
            elif isinstance(m, RowParallelLinear):
                input_is_parallel = True
                in_features = m.input_size
                out_features = m.output_size
            elif isinstance(m, nn.Linear):
                return LinearAdapter(
                    m, dim=self.dim, alpha=self.alpha, dropout=self.dropout, lora_A_init_method=self.lora_A_init_method
                )
            else:
                raise NotImplementedError(f"Layer type is unrecognized for LoRA: {type(m)}")

            logging.info(f"Adding DoRA to: {full_name}")
            adapter = ParallelLinearDoRAAdapter(
                in_features,
                out_features,
                self.dim,
                activation='identity',
                norm_position=None,
                norm_type=None,
                column_init_method=self.lora_A_init_method,
                row_init_method=self.lora_B_init_method,
                gather_output=False,
                input_is_parallel=input_is_parallel,
                dropout=self.dropout,
                dropout_position=self.dropout_position,
                model_parallel_config=getattr(m, "config", None),
                alpha=self.alpha,
            )
            return DoRALinear(m, adapter)
        return m
