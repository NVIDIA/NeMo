# Implementation of Hyena operator
#
# Poli, Michael and Massaroli, Stefano and Nguyen, Eric and Fu, Daniel Y and Dao, Tri and Baccus, Stephen and
# Bengio, Yoshua and Ermon, Stefano and Re, Christopher,
# Hyena Hierarchy: Towards Larger Convolutional Language Models
# 2023, https://arxiv.org/abs/2302.10866
#
# Based on the reference implementation from:
# https://github.com/HazyResearch/safari/blob/main/src/models/sequence/hyena.py

import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn

from einops import rearrange

from nemo.collections.common.parts.utils import activation_registry
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.utils.metaclasses import Singleton
from nemo.utils import logging

from megatron.core.transformer import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.custom_layers.transformer_engine import (
    TELayerNormColumnParallelLinear, TERowParallelLinear
)

try:
    from .fftconv_wrapper import fftconv_ref, fftconv_func
    HAVE_FFTCONV = True
except ImportError:
    HAVE_FFTCONV = False

try:
    from flashfftconv import FlashFFTConv as FlashFFTConvImpl
    HAVE_FLASHFFTCONV = True

    class FlashFFTConv(metaclass=Singleton):
        # Recommendation is to create single instance per model
        # https://github.com/HazyResearch/flash-fft-conv?tab=readme-ov-file#example-model
        def __init__(self, seqlen, dtype):
            self.flashfftconv = FlashFFTConvImpl(seqlen, dtype)

except ImportError:
    HAVE_FLASHFFTCONV = False

try:
    from causal_conv1d import causal_conv1d_fn
    HAVE_CAUSAL_CONV1D = True
except ImportError:
    HAVE_CAUSAL_CONV1D = False


@dataclass
class HyenaOperatorSubmodules:
    in_proj: Union[ModuleSpec, type] = IdentityOp
    short_filter: Union[ModuleSpec, type] = IdentityFuncOp
    implicit_filter: Union[ModuleSpec, type] = IdentityOp
    out_proj: Union[ModuleSpec, type] = IdentityOp


@dataclass
class HyenaFilterSubmodules:
    positional_embedding: Union[ModuleSpec, type] = IdentityOp
    linear: Union[ModuleSpec, type] = IdentityOp
    activation: Union[ModuleSpec, type] = IdentityOp
    modulation: Union[ModuleSpec, type] = IdentityOp


def auto_assign_attrs(cls, **kwargs):
    for k, v in kwargs.items():
        setattr(cls, k, v)


# TODO: Possibly remove
# @torch.jit.script
# def mul_sum(q, y):
#     return (q * y).sum(dim=1)


def register(module: nn.Module, name: str, tensor: torch.Tensor, learnable: bool):
    if learnable:
        module.register_parameter(name, nn.Parameter(tensor))
    else:
        module.register_buffer(name, tensor)


class Sin(nn.Module):
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim)) if train_freq else w * torch.ones(1, dim)

    def forward(self, x):
        return torch.sin(self.freq * x)


class CausalDepthWiseConv1d(nn.Module):
    def __init__(self, channels, width, bias=True):
        if not HAVE_CAUSAL_CONV1D:
            raise ImportError("Missing causal-conv1d library, please run 'pip install causal-conv1d'")

        super().__init__()
        self.channels = channels
        self.width = width
        self._conv_1d = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=width,
            padding=width - 1,
            groups=channels,
            bias=bias
        )

    def forward(self, x):
        return causal_conv1d_fn(x, self._conv_1d.weight.squeeze(1), self._conv_1d.bias)


class PositionalEmbedding(nn.Module):
    def __init__(
            self,
            emb_dim: int,
            seq_len: int,
            learn_z: bool = True,
            **kwargs):
        """Complex exponential positional embeddings for Hyena filters."""
        super().__init__()

        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]  # 1, L, 1

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
            # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        register(self, "z", z, learnable=learn_z)
        register(self, "t", t, learnable=False)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(nn.Module):
    def __init__(
            self,
            d_model,
            fast_decay_pct=0.3,
            slow_decay_pct=1.5,
            target=1e-2,
            learn_modulation: bool = False,
            modulate: bool = True,
            shift: float = 0.0,
            **kwargs
    ):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        register(self, "deltas", deltas, learnable=learn_modulation)

    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs())
            x = x * (decay + self.shift)
        return x


class HyenaFilter(nn.Module):
    def __init__(
            self,
            d_model,
            submodules: HyenaFilterSubmodules = None,
            fftconv_type='safari',
            precision='bf16',
            emb_dim=3,  # dim of input to MLP, augments with positional encoding
            order=16,  # width of the implicit MLP
            seq_len=1024,
            num_heads=1,
            learn_pos_emb_z=True,
            dropout=0.0,
            w=1,  # frequency of periodic activations
            bias=True,
            num_inner_mlps=2,
            normalized=False,
            **modulation_args
    ):
        """
        Implicit long filter with modulation.

        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP

        Note:
            filter_dropout is not implemented
        """
        super().__init__()

        if submodules is None:
            submodules = HyenaFilterSubmodules(
                positional_embedding=PositionalEmbedding,
                linear=nn.Linear,
                activation=Sin,
                modulation=ExponentialModulation
            )

        self.d_model = d_model
        self.use_bias = bias
        self.fftconv_type = fftconv_type
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)

        act = build_module(submodules.activation, dim=order, w=w)
        self.emb_dim = emb_dim
        if emb_dim % 2 == 0 or emb_dim < 3:
            raise ValueError("emb_dim must be odd and greater or equal to 3 (time, sine and cosine)")
        self.seq_len = seq_len
        self.num_heads = num_heads

        self.pos_emb = build_module(
            submodules.positional_embedding, emb_dim, seq_len, learn_pos_emb_z
        )

        # uses a variable number of inner linear layers
        self.implicit_filter = nn.Sequential(
            build_module(submodules.linear, emb_dim, order),
            act,
        )
        for i in range(num_inner_mlps):
            self.implicit_filter.append(build_module(submodules.linear, order, order))
            self.implicit_filter.append(act)
        # final linear layer
        self.implicit_filter.append(build_module(submodules.linear, order, d_model, bias=False))

        self.modulation = build_module(submodules.modulation, d_model, **modulation_args)

        self.normalized = normalized

        if self.fftconv_type == 'flash':
            self.flashfftconv = FlashFFTConv(2 * self.seq_len, torch_dtype_from_precision(precision)).flashfftconv
        else:
            self.flashfftconv = None

    def filter(self, L):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)

        h = self.modulation(t, h)

        if self.normalized:
            h = h / torch.norm(h, dim=-1, p=1, keepdim=True)

        return h

    def forward(self, x, k=None, L=None, bias=None, output_hbl_layout=False, v=None, q=None):
        if k is None:
            if L is None:
                raise ValueError('Must pass filter length L if kernel k is None')
            k = self.filter(L)
            # Ensure compatibility with filters that return a tuple
            k = k[0] if type(k) is tuple else k

        if bias is None:
            bias = self.bias
        bias = bias if self.use_bias else 0 * bias

        k = k.to(dtype=torch.float32)

        if self.fftconv_type == 'flash':
            x = x.contiguous()
            y = self.flashfftconv(x, k) + x * bias.unsqueeze(dim=1)
        else:  # fftconv_type == 'safari'
            bias = bias.to(dtype=torch.float32)

            y = fftconv_func(
                x, k, bias, dropout_mask=None, gelu=False,
                output_hbl_layout=output_hbl_layout, head_dim=self.num_heads, v=v, q=q
            )
        # TODO: Possibly remove
        # else:
        #     y = fftconv_ref(x, k, bias, dropout_mask=None, gelu=False)

        return y

class HyenaOperator(nn.Module):
    def __init__(
            self,
            config: TransformerConfig,
            d_model,
            l_max,
            submodules: HyenaOperatorSubmodules = None,
            order=2,
            filter_order=64,
            num_heads=1,
            dropout=0.0,
            filter_dropout=0.0,
            jit_filter=False,
            short_filter_order=3,
            activation="identity",
            **filter_args,
    ):
        r"""
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf

        Args:
            d_model (int): Dimension of the input and output embeddings (width of the layer)
            l_max: (int): Maximum input sequence length. Defaults to None
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            filter_order: (int): Width of the FFN parametrizing the implicit filter. Defaults to 64
            num_heads: (int): Number of heads. Defaults to 1
            dropout: (float): Dropout probability. Defaults to 0.0
            filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
            jit_filter: (bool): Whether JIT the implicit filter function. Defaults to False
            short_filter_order: (int): Length of the explicit input convolutional filter. Defaults to 3
            activation: (str): type of act between kernel output and FF (default identity)
        """
        super().__init__()

        if submodules is None:
            submodules = HyenaOperatorSubmodules(
                in_proj=TELayerNormColumnParallelLinear,
                short_filter=CausalDepthWiseConv1d,
                implicit_filter=HyenaFilter,
                out_proj=TERowParallelLinear
            )

        if d_model % num_heads != 0:
            raise ValueError(f'Model dimension {d_model} must be divisible by num heads {num_heads}')
        head_dim = d_model // num_heads

        auto_assign_attrs(
            self, d_model=d_model, order=order, l_max=l_max, num_heads=num_heads,
            head_dim=head_dim, filter_order=filter_order,
            short_filter_order=short_filter_order, filter_dropout=filter_dropout,
            jit_filter=jit_filter, activation=activation, mcore_config=config
        )
        self.activation = activation_registry[activation]()
        self.dropout = nn.Dropout(dropout)
        self.setup_projections(submodules)
        self.setup_filters(submodules, filter_args)

    def setup_projections(self, submodules: HyenaOperatorSubmodules):
        "Initializes input and output projections (over the width dimension)"
        self.out_proj = build_module(
            submodules.out_proj,
            self.d_model,
            self.d_model,
            config=self.mcore_config,
            init_method=self.mcore_config.output_layer_init_method,
            bias=True,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='out_proj',
        )

        self.in_proj = build_module(
            submodules.in_proj,
            self.d_model,
            (self.order + 1) * self.d_model,
            config=self.mcore_config,
            init_method=self.mcore_config.init_method,
            gather_output=False,
            bias=True,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='in_proj',
        )

    def setup_filters(self, submodules: HyenaOperatorSubmodules, filter_args):
        "Initializes the explicit and implicit filters"
        if self.order < 2:
            raise ValueError(f'Order must be at least 2, (got {self.order})')
        if self.num_heads > 1 and self.order > 2:
            raise ValueError(f'Multi-head supported only with order == 2 '
                             f'(got num_heads {self.num_heads} and order {self.order})')
        total_width = self.d_model * (self.order + 1)

        # TODO: Replace with causal_conv1d
        self.short_filter = build_module(
            submodules.short_filter,
            total_width,
            self.short_filter_order
        )

        if self.num_heads > 1:
            fftconv_type = 'safari'
            filter_channels = self.head_dim
        else:
            # TODO: Verify safari-fftconv vs flashfftconv for L <= 8192
            # if self.l_max <= 8192 and HAVE_FFTCONV:
            #     # safari-fftconv supports seq-len <= 8192 and is a bit faster vs. flashfftconv
            #     fftconv_type = 'safari'
            # else:
            #     fftconv_type = 'flash'
            fftconv_type = 'flash'
            filter_channels = self.head_dim * (self.order - 1)

        if fftconv_type == 'safari' and not HAVE_FFTCONV:
            raise ImportError('fftconv library not found. Please see README at <tbd> for instructions.')
        if fftconv_type == 'flash' and not HAVE_FLASHFFTCONV:
            raise ImportError('flashfftconv library not found. Please see README at <tbd> for instructions.')

        logging.info(f'Hyena fftconv_type: {fftconv_type}')

        self.filter_fn = build_module(
            submodules.implicit_filter,
            filter_channels,
            fftconv_type=fftconv_type,
            order=self.filter_order,
            seq_len=self.l_max,
            num_heads=self.num_heads,
            channels=1,
            dropout=self.filter_dropout,
            **filter_args
        )
        if self.jit_filter: self.filter_fn = torch.jit.script(self.filter_fn, self.L)

    def forward(self, u, *args, **kwargs):
        l = u.size(0)
        l_filter = min(l, self.l_max)
        u = self.in_proj(u)
        u = u[0] if isinstance(u, tuple) else u
        u = rearrange(u, 'l b d -> b d l')  # In MCore the leading dimension is the sequence dimension

        k = self.filter_fn.filter(l_filter)
        # `c` is always 1 by default
        k = rearrange(k, 'c l v -> c v l', v=self.head_dim)[0]

        uc = self.short_filter(u)[..., :l_filter]

        if self.num_heads == 1:
            y = self.conv_single_head(uc, k)
        else:
            y = self.conv_multi_head(uc, k)

        y = self.activation(y)
        y = self.out_proj(y)
        if isinstance(y, tuple):
            y, bias = y
        else:
            bias = None

        # Convert back to sequence-first for MCore
        y = rearrange(y, 'b l h -> l b h')

        # MCore TransformerLayer expects tuple where 2nd element represents the bias, it can be None
        return y, bias

    def conv_single_head(self, uc, k):
        k = rearrange(k, '(o v) l -> o v l', v=self.head_dim, o=self.order - 1)

        *x, v = uc.split(self.d_model, dim=1)
        bias = rearrange(self.filter_fn.bias, '(v o) -> o v', v=self.head_dim, o=self.order - 1)
        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.filter_fn(v, k=k[o], bias=bias[o])

        y = v * x[0]
        y = rearrange(y, 'b d l -> b l d')
        return y

    def conv_multi_head(self, uc, k):
        x1, x2, v = uc.split(self.d_model, dim=1)
        x1 = x1.contiguous()
        x2 = x2.contiguous()
        v = v.contiguous()
        bias = self.filter_fn.bias

        y = self.filter_fn(v, k, bias=bias, output_hbl_layout=True, v=x2, q=x1)
        y = rearrange(y, 'b d l -> b l d')
        return y


    @property
    def d_output(self):
        return self.d_model
