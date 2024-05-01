import inspect
from typing import Callable

import torch


from torch import nn
import torch.nn.functional as F
from torchao.dtypes.nf4tensor import NF4Tensor
from megatron.core.transformer import ModuleSpec, TransformerConfig
from nemo.collections.nlp.parts.peft_config import get_target_modules



class LinearNF4(torch.autograd.Function):
    @staticmethod

    #  inconsistently.

    def forward(ctx, input: torch.Tensor, weight: NF4Tensor):
        """Save the quantized nf4 weight for backward pass"""
        ctx.nf4_weight = weight
        return F.linear(input, weight.to(input.dtype))

    @staticmethod

    #  inconsistently.

    def backward(ctx, grad_output):
        """The nf4 weight will never require grad so we can just return the grad_output @ weight.to(grad_output.dtype)"""
        weight: NF4Tensor = ctx.nf4_weight
        return grad_output @ weight.to(grad_output.dtype), None


def linear_nf4(input: torch.Tensor, weight: NF4Tensor) -> torch.Tensor:
    """Apply a linear operation with the NF4Tensor weight

    Args:
        input: Input tensor
        weight: NF4Tensor weight
    """
    return LinearNF4.apply(input, weight)


def to_nf4(tensor, block_size: int = 64, scaler_block_size: int = 256):
    tensor1 = tensor.to(torch.bfloat16)
    return NF4Tensor.from_tensor(tensor1, block_size, scaler_block_size)


class NF4Linear(nn.Module):
    """
    NF4 Linear Layer for QLoRA as introduced in `QLORA: Efficient Finetuning of Quantized LLMs <https://arxiv.org/abs/2305.14314>`_.
    Serves as a replacement of megatron.core.transformer.custom_layers.transformer_engine.TERowParallelLinear

    Args: follow TERowParallelLinear
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str = None,
    ):
        super().__init__()
        assert not bias, "NF4 linear layer does not support bias"
        assert not is_expert, "NF4 linear layer does not support MoE"
        weight = to_nf4(nn.Parameter(torch.empty((output_size, input_size), device=torch.cuda.current_device())))
        # self.register_parameter("weight", UncastableParameter(weight))
        self.register_parameter("weight", nn.Parameter(weight))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): input tensor with shape ``(..., in_dim)``

        Returns:
            Tensor: output tensor with shape ``(..., out_dim)``

        """
        return linear_nf4(input=x, weight=self.weight), None

    def _apply(self, fn, recurse=True):
        if 'bfloat16' in inspect.getsource(fn):
            # don't cast weights in this layer to bf16
            return self
        else:
            super()._apply(fn, recurse=recurse)

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """ Sharding along axis 1, bias not sharded """
        state_dict = self.state_dict(prefix=prefix, keep_vars=True)
        return state_dict
class NF4LayernormLinear(nn.Module):
    """
    Layernorm + NF4 Layer for QLoRA.
    This class only combines the two layers for compatibility with TE's LayernormLinear layer, so that
    the implementation for LoRA and QLoRA can share the same code path.
    It does NOT fuse the two operations like TE does.
    Serves as a replacement of megatron.core.transformer.custom_layers.transformer_engine.TELayerNormColumnParallelLinear

    Args: follow TELayerNormColumnParallelLinear
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: str = None,
    ):
        super().__init__()
        assert not bias, "NF4 linear layer does not support bias"
        assert not is_expert, "NF4 linear layer does not support MoE"
        assert not skip_weight_param_allocation, "NF4 linear layer does not support `skip_weight_param_allocation`"

        layer_norm_weight = torch.nn.Parameter(torch.empty(input_size))
        self.register_parameter("layer_norm_weight", nn.Parameter(layer_norm_weight))
        if config.normalization != "RMSNorm":
            layer_norm_bias = torch.nn.Parameter(torch.empty(input_size))
            self.register_parameter('layer_norm_bias', layer_norm_bias)
            nn.init.zeros_(self.layer_norm_bias)
        else:
            self.layer_norm_bias = None

        self.zero_centered_gamma = config.layernorm_zero_centered_gamma
        self.normalization = config.normalization
        self.layer_norm_fn = self._create_layer_norm_fn()

        weight = to_nf4(nn.Parameter(torch.empty((output_size, input_size), device=torch.cuda.current_device())))
        # todo revert temporary change
        # weight = nn.Parameter(torch.empty((input_size, output_size), device=torch.cuda.current_device()))
        # import math
        # torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        ### end temporary change
        # self.register_parameter("weight", UncastableParameter(weight))
        self.register_parameter("weight", nn.Parameter(weight))
        self.te_return_bias = False

    def _create_layer_norm_fn(self):
        '''
        create the layernorm function signature in TE. Assume this layer is already running without gradients
        since this is for QLoRA.
        '''
        if self.normalization == 'LayerNorm':
            from transformer_engine.pytorch.module.layernorm import _LayerNorm
            layer_norm_fn = _LayerNorm.forward
        elif self.normalization == 'RMSNorm':
            from transformer_engine.pytorch.module.rmsnorm import _RMSNorm
            layer_norm_fn = _RMSNorm.forward
        else:
            raise ValueError("Unsupported normalization type:", self.normalization)

        return layer_norm_fn

    def forward(self, x):
        layer_norm_args = [
            None,  # ctx
            x,  # inp
            self.layer_norm_weight,
            1e-5,  # self.eps,
            0,  # self.fwd_rmsnorm_sm_margin,
            0,  # self.bwd_rmsnorm_sm_margin,
            self.zero_centered_gamma,
            False,  # torch.is_grad_enabled(),
            x.dtype,  # self.activation_dtype,
        ]
        if self.normalization == "LayerNorm":
            layer_norm_args.insert(3, self.layer_norm_bias)
        layernorm_output = self.layer_norm_fn(*layer_norm_args)
        linear_output = linear_nf4(input=layernorm_output, weight=self.weight)
        return (linear_output, layernorm_output), None

    def _apply(self, fn, recurse=True):
        # don't cast model weight to bf16
        if 'bfloat16' in inspect.getsource(fn):
            return self
        else:
            super()._apply(fn, recurse=recurse)

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """ Sharding along axis 1, bias not sharded """
        state_dict = self.state_dict(prefix=prefix, keep_vars=True)
        return state_dict

def get_gpt_layer_with_QLoRA_spec(cfg_peft) -> ModuleSpec:
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
    from megatron.core.transformer import TransformerLayer, TransformerLayerSubmodules
    from megatron.core.transformer.attention import SelfAttentionSubmodules, SelfAttention
    from megatron.core.transformer.custom_layers.transformer_engine import TELayerNormColumnParallelLinear, \
        TEDotProductAttention, TERowParallelLinear
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.mlp import MLP, MLPSubmodules

    qlora_targets = get_target_modules(cfg_peft.lora_tuning)

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=NF4LayernormLinear if 'attention_qkv' in qlora_targets else TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=NF4Linear if 'attention_dense' in qlora_targets else TERowParallelLinear,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp= ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=NF4LayernormLinear if 'mlp_fc1' in qlora_targets else TELayerNormColumnParallelLinear,
                    linear_fc2=NF4Linear if 'mlp_fc2' in qlora_targets else TERowParallelLinear,
                ),
        ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


def cast_checkpoint_to_nf4(state_dict, cfg_peft):
    import re
    assert cfg_peft['peft_scheme'] == 'qlora'
    qlora_targets = get_target_modules(cfg_peft.lora_tuning)
    patterns = []
    if 'attention_qkv' in qlora_targets:
        patterns.append(r".*.layers.[0-9]*.self_attention.linear_qkv.weight")
    if 'attention_dense' in qlora_targets:
        patterns.append(r".*.layers.[0-9]*.self_attention.linear_proj.weight")
    if 'mlp_fc1' in qlora_targets:
        patterns.append(r".*.layers.[0-9]*.mlp.linear_fc1.weight")
    if 'mlp_fc2' in qlora_targets:
        patterns.append(r".*.layers.[0-9]*.mlp.linear_fc2.weight")
    pattern = "|".join(patterns)
    nf4_state_dict = {}
    for key, val in state_dict.items():
        if re.match(pattern, key):
            nf4_state_dict[key] = to_nf4(val)

    state_dict.update(nf4_state_dict)
    return state_dict


#
# class QLorALinear(nn.Module, AdapterModuleUtil):
#     """
#     QLoRA linear layer

#     Args:
#         # TODO
#     """
#
#     def __init__(
#         self,
#         in_features: int,
#         out_features: int,
#         dim: int,
#         alpha: Optional[float] = None,
#         dropout: float = 0.0,
#         dropout_position: str = 'post',
#         model_parallel_config: Optional[ModelParallelConfig] = None,
#     ):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.dim = dim
#         self.alpha = alpha
#
#         linear = nn.Linear(in_features, out_features, bias=False)
#         weight = to_nf4(linear.weight)
#
#         self.register_parameter("weight", nn.Parameter(weight))
#
#         self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
#         self.dropout_position = dropout_position
#
#         self.adapter_linear_in = nn.Linear(in_features, dim, bias=False)
#         self.adapter_linear_out = nn.Linear(dim, out_features, bias=False)
#         nn.init.kaiming_uniform_(self.adapter_linear_in.weight, a=math.sqrt(5))
#         nn.init.zeros_(self.adapter_linear_out.weight)
#
#         # cast all parameters when using amp O2 training
#         if model_parallel_config.bf16:
#             self.bfloat16()
#         elif model_parallel_config.fp16:
#             self.half()
#
#         # Setup adapter strategy
#         # TODO what does this line do?
#         self.setup_adapter_strategy(adapter_mixin_strategies.ReturnResultAdapterStrategy())
#
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x (Tensor): input tensor with shape ``(..., in_dim)``
#
#         Returns:
#             Tensor: output tensor with shape ``(..., out_dim)``
#
#         """
#         out = linear_nf4(input=x, weight=self.weight)
#
#         if self.dropout is not None and self.dropout_position == 'pre':
#             x = self.dropout(x)
#         x = self.adapter_linear_in(x)
#         x = self.adapter_linear_out(x)
#         if self.dropout is not None and self.dropout_position == 'post':
#             x = self.dropout(x)
#
#         lora_out = x * (self.alpha / self.dim)
#         return out + lora_out
#
#
# class QLoraKQVAdapter(QLorALinear):
#     pass
#
# class QLoraDenseAttentionAdapter(QLorALinear):
#     pass
#
# class QLoraHto4HAdapter(QLorALinear):
#     pass
#
# class QLora4HtoHAdapter(QLorALinear):
#     pass
#
# @dataclass
# class QLoraKQVAdapterConfig(ParallelLinearAdapterConfig):
#     _target_: str = "{0}.{1}".format(QLoraKQVAdapter.__module__, QLoraKQVAdapter.__name__)
#
# @dataclass
# class QLoraDenseAttentionAdapterConfig(ParallelLinearAdapterConfig):
#     _target_: str = "{0}.{1}".format(QLoraDenseAttentionAdapter.__module__, QLoraDenseAttentionAdapter.__name__)
#
# @dataclass
# class QLoraHto4HAdapterConfig(ParallelLinearAdapterConfig):
#     _target_: str = "{0}.{1}".format(QLoraHto4HAdapter.__module__, QLoraHto4HAdapter.__name__)
#
# @dataclass
# class QLora4HtoHAdapterConfig(ParallelLinearAdapterConfig):
#     _target_: str = "{0}.{1}".format(QLora4HtoHAdapter.__module__, QLora4HtoHAdapter.__name__)
