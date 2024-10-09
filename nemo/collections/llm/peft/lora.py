# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass, field
from typing import List, Literal

from megatron.core import parallel_state
from torch import nn

from nemo.lightning.pytorch.callbacks.peft import PEFT, AdapterWrapper
from nemo.utils import logging
from nemo.lightning.io.mixin import IOMixin
from pytorch_lightning.trainer.states import TrainerFn
import torch

from nemo import lightning as nl
from nemo.collections import llm
from typing import Any, Dict, List
import torch
from nemo.lightning.io import load_context, ModelConnector
from nemo.lightning.megatron_parallel import MegatronParallel
from nemo.utils.get_rank import is_global_rank_zero
from pathlib import Path
from nemo.utils import logging


class AdapterParallelAdd(AdapterWrapper):
    """An adapter wrapper that adds the output of the adapter to the output of the wrapped module.

    This class is designed to be used with LoRA (Low-Rank Adaptation) and similar techniques
    where the adapter's output is added to the main module's output. It extends the AdapterWrapper
    class to provide a specific implementation of the forward method.
    """

    def forward(self, x):
        linear_output = self.to_wrap(x)
        assert isinstance(
            linear_output, tuple
        ), f"{self.to_wrap} should return a tuple but instead returns {linear_output}"
        """ Four cases for the wrapped module's return values
        1. nothing: (out, None)
        2. return_bias: (out, bias)
        2. return_layernorm_output: ((out, ln_out), None)
        3. both: (out, bias, ln_out)
        """
        if len(linear_output) == 2:
            linear_output, bias = linear_output
            if isinstance(linear_output, tuple) and len(linear_output) == 2:
                linear_output, layernorm_output = linear_output
                x = layernorm_output
        elif len(linear_output) == 3:
            linear_output, bias, layernorm_output = linear_output
            x = layernorm_output

        adapter_output = self.adapter(x.contiguous())
        return linear_output + adapter_output, bias


@dataclass
class LoRA(PEFT):
    """
    Implements the LoRA (Low-Rank Adaptation) module for parameter-efficient fine-tuning.

    LoRA uses a low-rank projection to adapt the weights of a pre-trained model to a new downstream task.
    This class facilitates the application of LoRA to specific modules within the model architecture.

    Args:
        target_modules (List[str], optional): A list of module names to apply LoRA to.
            Defaults to all linear layers ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2'].
                - 'linear_qkv': Apply LoRA to the fused linear layer used for query, key, and value projections
                                in self-attention modules.
                - 'linear_proj': Apply LoRA to the linear layer used for projecting the output of self-attention modules.
                - 'linear_fc1': Apply LoRA to the first fully-connected layer in MLP.
                - 'linear_fc2': Apply LoRA to the second fully-connected layer in MLP.
        dim (int): Dimension of the low-rank projection space. Defaults to 32.
        alpha (int): Weighting factor for the low-rank projection. Defaults to 32.
        dropout (float): Dropout rate for the low-rank projection. Defaults to 0.0.
        dropout_position (Literal['pre', 'post'], optional): Position for applying dropout.
            Can be 'pre' (before the low-rank projection) or 'post' (after). Defaults to 'post'.

    Example:
    --------
        >>> from nemo.collections import llm
        >>> lora = llm.peft.LoRA(target_modules=['linear_qkv', 'linear_proj'], dim=32)
        >>> model = llm.Mistral7BModel(model_transform=lora)
        >>> # (set up trainer and data)
        >>> trainer.fit(model, data)

    References:
    -----------
        Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021).
        LoRA: Low-Rank Adaptation of Large Language Models. arXiv preprint arXiv:2106.09685.
        https://arxiv.org/abs/2106.09685

    )
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
        Applies LoRA to a specific module within the model architecture.

        Args:
            m (nn.Module): The module to apply LoRA to.
            name (str, optional): Name of the module (if applicable). Defaults to None.
            prefix (str, optional): Prefix for the module name (if applicable). Defaults to None.

        Returns:
            nn.Module: The modified module with LoRA applied, or the original module if not a target.
        """
        from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import ParallelLinearAdapter

        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if name in self.target_modules:
            # m.in_features and m.out_features are divided by tp_size already,
            # but in_features and out_features passed to ParallelLinearAdapter are not.
            if name in ['linear_qkv', 'linear_fc1']:
                # Column Parallel Linear
                input_is_parallel = False
                in_features = m.in_features
                out_features = m.out_features * tp_size
                # LoRA is applied after layernorm, so layernorm output must be returned
                m.return_layernorm_output = True
                # perf optimization for LoRA + SP
                if m.config.sequence_parallel and not m.ub_overlap_ag:
                    m.return_layernorm_output_gathered = True
            else:  # name in ['linear_proj', 'linear_fc2']
                # Row Parallel Linear
                input_is_parallel = True
                in_features = m.in_features * tp_size
                out_features = m.out_features

            logging.info(f"Adding lora to: {prefix}.{name}")
            adapter = ParallelLinearAdapter(
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
            return AdapterParallelAdd(m, adapter)
        return m
    
    def apply_transform(self, trainer):
        super().apply_transform(trainer)
        import pdb; pdb.set_trace()
        if trainer.state.fn == TrainerFn.PREDICTING:            
            base_sharded_dict = {k:v for k,v in trainer.model.state_dict().items() if 'adapter' not in k and 'extra_state' not in k }
            lora_sharded_dict = {k:v.data.data for k, v in trainer.model.sharded_state_dict().items() if 'adapter' in k and 'extra_state' not in k}
            merged_weights = self._merge_lora_weights(base_model_state_dict = base_sharded_dict, 
                                     lora_state_dict = lora_sharded_dict, 
                                     num_layers = trainer.model._modules['0'].config.num_layers, 
                                     tp_size = trainer.strategy.tensor_model_parallel_size,
                                     rank =torch.distributed.get_rank())
            trainer.model.load_state_dict(merged_weights)

    def _merge_lora_weights(self, base_model_state_dict: Dict[str, Any],
          lora_state_dict: Dict[str, Any],
          num_layers: int,
          tp_size: int,
          rank: int):
        mcore_layer_to_lora = {}
        """
        'self_attention.linear_qkv.adapter.linear_in.weight' 
        'self_attention.linear_qkv.adapter.linear_out.weight', 
        'self_attention.linear_proj.adapter.linear_in.weight'
        'self_attention.linear_proj.adapter.linear_out.weight',
        'mlp.linear_fc1.adapter.linear_in.weight',
        'mlp.linear_fc1.adapter.linear_out.weight', 
        'mlp.linear_fc2.adapter.linear_in.weight',
        'mlp.linear_fc2.adapter.linear_out.weight', 
        """

        mcore_layer_to_lora["attention_qkv"] = {
            "base_model_layer": "self_attention.linear_qkv.weight",
            "lora_in": "self_attention.linear_qkv.adapter.linear_in.weight",
            "lora_out": "self_attention.linear_qkv.adapter.linear_out.weight",
        }
        mcore_layer_to_lora["attention_dense"] = {
            "base_model_layer": "self_attention.linear_proj.weight",
            "lora_in": "self_attention.linear_proj.adapter.linear_in.weight",
            "lora_out": "self_attention.linear_proj.adapter.linear_out.weight",
        }
        mcore_layer_to_lora["mlp_fc1"] = {
            "base_model_layer": "mlp.linear_fc1.weight",
            "lora_in": "mlp.linear_fc1.adapter.linear_in.weight",
            "lora_out": "mlp.linear_fc1.adapter.linear_out.weight",
        }
        mcore_layer_to_lora["mlp_fc2"] = {
            "base_model_layer": "mlp.linear_fc2.weight",
            "lora_in": "mlp.linear_fc2.adapter.linear_in.weight",
            "lora_out": "mlp.linear_fc2.adapter.linear_out.weight",
        }

        for nl in range(num_layers):
            for key in mcore_layer_to_lora.keys():
                ##TODO: prefix should be model or module or 0.module?
                key_base = f'0.module.decoder.layers.{nl}.{mcore_layer_to_lora[key]["base_model_layer"]}'
                key_lora_in = f'module.decoder.layers.{nl}.{mcore_layer_to_lora[key]["lora_in"]}'
                key_lora_out = f'module.decoder.layers.{nl}.{mcore_layer_to_lora[key]["lora_out"]}'
                if key_lora_in in lora_state_dict and key_lora_out in lora_state_dict:
                    if tp_size > 1:
                        gathered_lora_in = [torch.zeros_like(lora_state_dict[key_lora_in]) for _ in range(tp_size)]
                        gathered_lora_out = [torch.zeros_like(lora_state_dict[key_lora_out]) for _ in range(tp_size)]
                        torch.distributed.all_gather(gathered_lora_in, lora_state_dict[key_lora_in])
                        torch.distributed.all_gather(gathered_lora_out, lora_state_dict[key_lora_out])

                        if is_global_rank_zero():
                            print(f"RANK{torch.distributed.get_rank()} has {key_lora_in} shape {lora_state_dict[key_lora_in].shape}") #gathered lorain{gathered_lora_in}")
                            print(f"RANK{torch.distributed.get_rank()} has {key_lora_out} shape {lora_state_dict[key_lora_out].shape}") #gathered loraout {gathered_lora_out}")
                        ## TODO: Who decides what dim they split?
                        tp_dim_lora_in = 1 if key in ["attention_dense", 'mlp_fc2'] else 0
                        wt_lora_in = torch.cat(gathered_lora_in, dim=tp_dim_lora_in).float()
                        wt_lora_out = torch.cat(gathered_lora_out, dim=0).float()
                        wt_lora = wt_lora_out @ wt_lora_in
                        tp_dim_base = 0 if key in ["attention_qkv", "mlp_fc1"] else 1
                        wt_lora_current_rank = torch.chunk(wt_lora, tp_size, dim=tp_dim_base)[rank]
                    else: #when tp==1
                        wt_lora_in = lora_state_dict[key_lora_in]
                        wt_lora_out = lora_state_dict[key_lora_out]
                        wt_lora = wt_lora_out @ wt_lora_in
                        wt_lora_current_rank = wt_lora

                    wt_base = base_model_state_dict[key_base]
                    logging.info(f"Full {key_base} wt_lora_in {wt_lora_in.shape}, wt_lora_out {wt_lora_out.shape}, wt_lora {wt_lora.shape}, wt_base {wt_base.shape}")

                    
                    base_model_state_dict[key_base] = (wt_base.float() + wt_lora_current_rank.to(wt_base.device)).type_as(wt_base)
                    logging.info(f'merging for weight {key_base}')

        return base_model_state_dict
